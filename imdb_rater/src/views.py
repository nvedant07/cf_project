# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.conf import settings
from django.template.defaulttags import register

import os
from django.conf import settings
from models import Movie, UserRating, Trust, Feedback, DislikedFilms
from forms import RatingForm, TrustForm, AlgoForm
from urllib import urlopen, urlretrieve
from bs4 import BeautifulSoup
from operator import itemgetter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

import argparse
import networkx as nx
from node2vec import Graph
from gensim.models import Word2Vec

import pandas as pd
import numpy as np

BASEURL = "http://www.imdb.com/title/tt%s/"

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

def scrape_imdb():
	items = pd.read_csv(os.path.join(settings.BASE_DIR, "movies.csv"), usecols=[0,1], index_col = 0)
	links = pd.read_csv(os.path.join(settings.BASE_DIR, "links.csv"), index_col=0, usecols=[0,1], dtype={'imdbId':str})
	merged_df = pd.merge(items, links, left_index=True, right_index=True, how='inner')

	for idx, row in merged_df.iterrows():
		if int(row['title'][-5:].rstrip(')')) >= 2010:
			try:
				if not os.path.exists(os.path.join(settings.BASE_DIR,"movie_posters/%s.jpg"%str(idx))):
					html = urlopen(BASEURL%row['imdbId'])
					soup = BeautifulSoup(html.read())
					
					if float(soup.findAll("span", {"itemprop":"ratingValue"})[0].text) > 8.0:
						image_url = soup.findAll("div", { "class" : "poster" })[0].findAll("img")[0]["src"]
						print image_url
						urlretrieve(image_url, os.path.join(settings.BASE_DIR,"movie_posters/%s.jpg"%str(idx)))
						m = Movie(movielens_id=idx, name=row['title'], url=BASEURL%row['imdbId'], image="%s.jpg"%str(idx))
						m.save()
			except:
				print "couldn't fetch from " + BASEURL%row['imdbId']
				# m = Movie(movielens_id=idx, name=row['title'], url=BASEURL%row['imdbId'])
			# m.save()

def make_user_item_matrix():
	all_ratings = UserRating.objects.all()
	users = []
	movies = []
	for rating in all_ratings:
		if rating.corresponding_user not in users:
			users.append(rating.corresponding_user)
		if rating.movie not in movies:
			movies.append(rating.movie)
	user_mapping_forward = dict(enumerate(users))
	user_mapping = {v:k for k,v in user_mapping_forward.items()}
	item_mapping_forward = dict(enumerate(movies))
	item_mapping = {v:k for k,v in item_mapping_forward.items()}
	user_item_matrix = np.zeros((len(users), len(movies)))
	for rating in all_ratings:
		user_item_matrix[user_mapping[rating.corresponding_user]][item_mapping[rating.movie]] = int(rating.rating)
	return user_item_matrix, user_mapping, item_mapping, user_mapping_forward, item_mapping_forward

def user_user(current_user):
	user_item_matrix, user_mapping, item_mapping, user_mapping_forward, item_mapping_forward = make_user_item_matrix()
	item_user_matrix = user_item_matrix.T
	mask = np.ones(item_user_matrix.shape)
	mask[item_user_matrix.nonzero()] = 0
	masked_matrix = np.ma.masked_array(item_user_matrix, mask=mask)
	similarity_matrix = np.ma.corrcoef(masked_matrix, rowvar=False).data

	y_pred = []
	i = user_mapping[current_user]
	for j in range(user_item_matrix.shape[1]):
		user_sum = np.sum(similarity_matrix[i]) - 1
		if user_sum == 0:
			predicted_rating = np.mean(user_item_matrix[i,:][user_item_matrix[i,:].nonzero()])
		elif len(user_item_matrix[i,:][user_item_matrix[i,:].nonzero()]) == 0:
			predicted_rating = np.mean(user_item_matrix[:,j][user_item_matrix[:,j].nonzero()])
		else:
			numerator = 0
			for user2, k in user_mapping.items():
				if k != i and user_item_matrix[k][j] != 0:
					u_a_mean = np.mean(user_item_matrix[k,:][user_item_matrix[k,:].nonzero()])
					numerator += similarity_matrix[i][k] * (user_item_matrix[k][j] - u_a_mean)
			predicted_rating = np.mean(user_item_matrix[i,:][user_item_matrix[i,:].nonzero()]) + numerator/user_sum
		y_pred.append(predicted_rating)
	keep_indices = np.where(user_item_matrix[i,:].flatten() == 0)[0]
	print keep_indices
	y_pred = list(enumerate(y_pred))
	y_pred = sorted(y_pred, key=itemgetter(1))
	y_pred = np.array([k for k in np.array(zip(*y_pred)[0]) if k in keep_indices])
	y_pred = y_pred[-10:]
	print user_item_matrix[i,:]
	name_to_path_dicts = []
	name_to_path = {}
	name_to_url = {}
	name_to_id = {}
	count = 1
	for val in y_pred:
		movie = item_mapping_forward[val]
		name_to_path[movie.name] = movie.image
		name_to_url[movie.name] = movie.url
		name_to_id[movie.name] = movie.movielens_id
		if count % 4 == 0:
			name_to_path_dicts.append(name_to_path)
			name_to_path = {}
		count += 1
	name_to_path_dicts.append(name_to_path)
	return name_to_path_dicts, name_to_url, name_to_id

def item_item(current_user):
	user_item_matrix, user_mapping, item_mapping, user_mapping_forward, item_mapping_forward = make_user_item_matrix()
	mask = np.ones(user_item_matrix.shape)
	mask[user_item_matrix.nonzero()] = 0
	masked_matrix = np.ma.masked_array(user_item_matrix, mask=mask)
	similarity_matrix = np.ma.corrcoef(masked_matrix, rowvar=False).data
	print similarity_matrix.shape

	y_pred = []
	i = user_mapping[current_user]
	for j in range(user_item_matrix.shape[1]):
		item_sum = np.sum(similarity_matrix[j]) - 1
		if item_sum == 0:
			predicted_rating = np.mean(user_item_matrix[:,j][user_item_matrix[:,j].nonzero()])
			y_pred.append(predicted_rating)
		else:
			predicted_rating = np.mean(user_item_matrix[:,j][user_item_matrix[:,j].nonzero()]) + np.dot(similarity_matrix[j].flatten(), user_item_matrix[i].flatten())/item_sum
        	y_pred.append(predicted_rating)
	keep_indices = np.where(user_item_matrix[i,:].flatten() == 0)[0]
	print keep_indices
	y_pred = list(enumerate(y_pred))
	print y_pred
	y_pred = sorted(y_pred, key=itemgetter(1))
	y_pred = np.array([k for k in np.array(zip(*y_pred)[0]) if k in keep_indices])
	y_pred = y_pred[-10:]
	print user_item_matrix[i,:]
	name_to_path_dicts = []
	name_to_path = {}
	name_to_url = {}
	name_to_id = {}
	count = 1
	for val in y_pred:
		movie = item_mapping_forward[val]
		name_to_path[movie.name] = movie.image
		name_to_url[movie.name] = movie.url
		name_to_id[movie.name] = movie.movielens_id
		if count % 4 == 0:
			name_to_path_dicts.append(name_to_path)
			name_to_path = {}
		count += 1
	name_to_path_dicts.append(name_to_path)
	return name_to_path_dicts, name_to_url, name_to_id


'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

def parse_args(input_, output_, dimensions):
	'''
	Parses the node2vec arguments.
	'''
	variables = {
		"input": input_,
		"output": output_,
		"dimensions": dimensions,
		"directed": True,
		"walk_length": 80,
		"num_walks":10,
		"window_size":10,
		"iter":1,
		"workers":8,
		"p":1,
		"q":1,
		"weighted":False
	}
	return variables

def read_graph(variables):
	'''
	Reads the input network in networkx.
	'''
	if variables["weighted"]:
		G = nx.read_edgelist(variables["input"], nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(variables["input"], nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1
	# print G.nodes()
	for i in range(len(User.objects.all())):
		if i not in G.nodes():
			G.add_node(i)
	# print G[1][32]['weight']
	if not variables["directed"]:
		G = G.to_undirected()

	return G

def learn_embeddings(walks, variables):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=variables["dimensions"], window=variables["window_size"], min_count=0, sg=1, workers=variables["workers"], iter=variables["iter"])
	model.wv.save_word2vec_format(variables["output"])
	
	return

def some_other_main(variables):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph(variables)
	print nx_G
	G = Graph(nx_G, variables["directed"], variables["p"], variables["q"])
	G.preprocess_transition_probs()
	walks = G.simulate_walks(variables["num_walks"], variables["walk_length"])
	learn_embeddings(walks, variables)


def deep_nn(current_user):
	user_item_matrix, user_mapping, item_mapping, user_mapping_forward, item_mapping_forward = make_user_item_matrix()
	trusts = Trust.objects.all()
	f = open('edges.txt','w')
	for t in trusts:
		f.write('%d %d\n'%(t.issuing_user.id, t.trusted_user.id))
	f.close()
	variables = parse_args('edges.txt', 'embeddings.txt', 512)
	some_other_main(variables)
	embeddings = np.loadtxt('embeddings.txt',skiprows=1)
	np.sort(embeddings, axis=0)
	embeddings = embeddings[:,1:]

	user_item_list = []
	rating_list = []
	for i in range(user_item_matrix.shape[0]):
		for j in range(user_item_matrix.shape[1]):
			if user_item_matrix[i][j] != 0:
				user_item_list.append((i,j))
				rating_list.append(user_item_matrix[i][j])
	X = np.array(user_item_list)
	Y = np.array(rating_list)
	X_final = np.empty((X.shape[0],558))
	var = np.arange(46)
	one_hot = LabelBinarizer().fit_transform(var)
	c = 0
	for u,i in X:
		temp = np.hstack((embeddings[u], one_hot[i]))
		X_final[c] = temp
		c += 1
	Y_final = LabelBinarizer().fit_transform(Y)
	clf = MLPClassifier(solver='adam',hidden_layer_sizes=(100,50),verbose=True)
	clf.fit(X_final, Y_final)
	
	user_id = user_mapping[current_user]
	test_embedding = embeddings[user_id]
	X_test = np.empty((46,558))
	for i in range(X_test.shape[0]):
		temp = np.hstack((test_embedding, one_hot[i]))
		X_test[i] = temp
	y_pred = clf.predict(X_test)
	y_pred = [np.argmax(p) + 1 for p in y_pred]

	i = user_mapping[current_user]
	keep_indices = np.where(user_item_matrix[i,:].flatten() == 0)[0]
	print keep_indices
	y_pred = list(enumerate(y_pred))
	print y_pred
	y_pred = sorted(y_pred, key=itemgetter(1))
	y_pred = np.array([k for k in np.array(zip(*y_pred)[0]) if k in keep_indices])
	y_pred = y_pred[-10:]
	print user_item_matrix[i,:]
	name_to_path_dicts = []
	name_to_path = {}
	name_to_url = {}
	name_to_id = {}
	count = 1
	for val in y_pred:
		movie = item_mapping_forward[val]
		name_to_path[movie.name] = movie.image
		name_to_url[movie.name] = movie.url
		name_to_id[movie.name] = movie.movielens_id
		if count % 4 == 0:
			name_to_path_dicts.append(name_to_path)
			name_to_path = {}
		count += 1
	name_to_path_dicts.append(name_to_path)
	return name_to_path_dicts, name_to_url, name_to_id


@login_required
def home(request):
	# scrape_imdb()
	u = request.user
	rated_items = UserRating.objects.filter(corresponding_user=u)
	movies = Movie.objects.all()
	name_to_path_dicts = []
	name_to_path = {}
	name_to_url = {}
	name_to_form = {}
	name_to_id = {}
	count = 1
	blacklisted_movies = []
	for rated_item in rated_items:
		if rated_item.movie in movies:
			blacklisted_movies.append(rated_item.movie)
	for movie in movies:
		if movie not in blacklisted_movies:
			rating_form = RatingForm(request.POST or None)
			name_to_form[movie.name] = rating_form
			name_to_path[movie.name] = movie.image
			name_to_url[movie.name] = movie.url
			name_to_id[movie.name] = movie.movielens_id
			if count % 4 == 0:
				name_to_path_dicts.append(name_to_path)
				name_to_path = {}
			count += 1
	name_to_path_dicts.append(name_to_path)
	if len(rated_items) < 10:
		heading = "Please rate atleast %d more item(s)"%(10 - len(rated_items))
		button_status = 'disabled'
	else:
		heading = "You may proceed"
		button_status = ''
	context= {
		'heading': heading,
		'name_to_path_dicts': name_to_path_dicts,
		'name_to_url': name_to_url,
		'name_to_form': name_to_form,
		'name_to_id': name_to_id,
		'button_status' : button_status
	}
	return render(request, 'rate.html', context)

def post_rating(request):
	if request.method == 'POST':
		movielens_id = int(request.META['PATH_INFO'].lstrip('/home_'))
		corresponding_movie = Movie.objects.get(movielens_id=movielens_id)
		rating = request.POST.get('rating')
		u = User.objects.get(username = request.user.username)
		rated_object = UserRating(movie=corresponding_movie, corresponding_user=u, rating=rating)
		rated_object.save()
	return redirect('home')

def trust(request):
	u = request.user
	rated_items = UserRating.objects.filter(corresponding_user=u)
	if len(rated_items) < 10:
		return redirect('home')
	all_users = User.objects.all()
	already_rated = Trust.objects.filter(issuing_user=u)
	username_to_path_dicts = []
	username_to_name = {}
	username_to_id = {}
	username_to_form = {}
	username_to_path = {}
	count = 1
	blacklisted_users = []
	for rated_item in already_rated:
		blacklisted_users.append(rated_item.trusted_user)
	for user in all_users:
		if user not in blacklisted_users and user != request.user:
			rating_form = TrustForm(request.POST or None)
			username_to_form[user.username] = rating_form
			# username_to_path[user.username] = '%s.jpg'%(request.user.username)
			username_to_path[user.username] = 'dummy.jpg'
			username_to_id[user.username] = user.id
			username_to_name[user.username] = user.get_full_name()
			if count % 4 == 0:
				username_to_path_dicts.append(username_to_path)
				username_to_path = {}
			count += 1
	username_to_path_dicts.append(username_to_path)
	heading = "Issue trust ratings"
	context= {
		'heading': heading,
		'username_to_path_dicts': username_to_path_dicts,
		'username_to_form': username_to_form,
		'username_to_id': username_to_id,
		'username_to_name': username_to_name
	}
	return render(request,'trust.html',context)

def post_trust(request):
	if request.method == 'POST':
		userid = int(request.META['PATH_INFO'].lstrip('/post_trust_'))
		issuing_user = request.user
		trusted_user = User.objects.get(id=userid)
		# value = int(request.POST.get(''))
		# if value == 2:
		# 	value = -1
		value = 1
		trust = Trust(issuing_user=issuing_user,trusted_user=trusted_user,trust_value=value)
		trust.save()
	return redirect('trust')

def display(request):
	form = AlgoForm(request.POST or None)
	context = {
		'algo_form': form,
		'heading': 'Choose an algo to display recommendations'
	}
	return render(request, 'display.html', context)

def algo(request):
	u = request.user
	rated_items = UserRating.objects.filter(corresponding_user=u)
	if len(rated_items) < 10:
		return redirect('home')
	if request.method == 'GET':
		return redirect('display')

	algo = int(request.POST.get('algo'))
	if algo == 1:
		heading = "Algo A"
		name_to_path_dicts, name_to_url, name_to_id = user_user(request.user)
	elif algo == 2:
		heading = "Algo B"
		name_to_path_dicts, name_to_url, name_to_id = item_item(request.user)
	else:
		heading = "Algo C"
		name_to_path_dicts, name_to_url, name_to_id = deep_nn(request.user)

	all_disliked_films = DislikedFilms.objects.filter(user=request.user)
	all_disliked_films = [f.movie.name for f in all_disliked_films]
	for i in range(len(name_to_path_dicts)):
		name_to_path = name_to_path_dicts[i]
		for name in name_to_path.keys():
			if name in all_disliked_films:
				name_to_path.pop(name)
		name_to_path_dicts[i] = name_to_path

	form = AlgoForm(request.POST or None)
	context = {
		'algo_form': form,
		'heading': heading,
		'algo':algo,
		'name_to_path_dicts': name_to_path_dicts,
		'name_to_url': name_to_url,
		'name_to_id': name_to_id
	}
	return render(request,'display.html',context)

def user_reaction(request):
	if request.method == 'POST':
		algo, movie_id = (request.META['PATH_INFO'].lstrip('/user_reaction_')).split('_')
		algo, movie_id = int(algo), int(movie_id)

		d = DislikedFilms(user=request.user, movie=Movie.objects.get(movielens_id=movie_id))
		d.save()

		feedbacks = Feedback.objects.filter(algorithm=algo)
		if len(feedbacks) == 0:
			f = Feedback(algorithm=algo, count=1)
			f.save()
		else:
			feedbacks.update(count = int(feedbacks[0].count) + 1)
		if algo == 1:
			heading = "Algo A: Feedback Submitted"
			name_to_path_dicts, name_to_url, name_to_id = user_user(request.user)
		elif algo == 2:
			heading = "Algo B: Feedback Submitted"
			name_to_path_dicts, name_to_url, name_to_id = item_item(request.user)
		else:
			heading = "Algo C: Feedback Submitted"
			name_to_path_dicts, name_to_url, name_to_id = deep_nn(request.user)

		all_disliked_films = DislikedFilms.objects.filter(user=request.user)
		all_disliked_films = [f.movie.name for f in all_disliked_films]
		for i in range(len(name_to_path_dicts)):
			name_to_path = name_to_path_dicts[i]
			for name in name_to_path.keys():
				if name in all_disliked_films:
					name_to_path.pop(name)
			name_to_path_dicts[i] = name_to_path


		form = AlgoForm(request.POST or None)
		context= {
			'algo_form': form,
			'heading': heading,
			'algo':algo,
			'name_to_path_dicts': name_to_path_dicts,
			'name_to_url': name_to_url,
			'name_to_id': name_to_id
		}
		return render(request,'display.html',context)			
	else:
		return redirect('display')