import pandas as pd
import numpy as np
import math, os, time, json
import h5py

def load_data():
	ratings = pd.read_csv("ratings_train.txt", sep="\t", header=None,
							names=["object_id", "member_id", "rating", "status", "creation", "last_modified", "type", "vertical_id"],
							dtype={"object_id" : int, "member_id" : int, "rating" : int})
	binary_trusts = pd.read_csv("user_rating.txt", sep="\t", header=None, names=["my_id", "other_id", "value", "creation"],
								dtype={"my_id" : int, "other_id" : int, "value" : int})
	article_to_user = pd.read_csv("mc.txt", sep="|", header=None, names=["content_id", "author_id", "subject_id"],
									dtype={"content_id" : int, "author_id" : int})
	ratings.replace(to_replace={"rating" : { 6 : 5 }}, inplace=True)
	
	all_object_ids = article_to_user['content_id'].unique()
	rated_object_ids = ratings['object_id'].unique()

	user_authors = article_to_user['author_id'].unique()
	user_raters = ratings['member_id'].unique()
	trust_statement_users = binary_trusts['my_id'].unique()
	trusted_users = binary_trusts['other_id'].unique()

	all_users = np.array(list(set(np.concatenate((user_authors, user_raters, trust_statement_users, trusted_users)))))
	all_items = np.array(list(set(np.concatenate((all_object_ids, rated_object_ids)))))
	if not os.path.exists('./user_mapping.json'):
		# map every user id to an integer in the range 0, len(all_users); makes sure every user who rated lies between 0 and len(user_raters)
		user_mapping = {}
		for i in range(len(user_raters)):
			user_mapping[user_raters[i]] = i
		remaining_users = list(set(all_users) - set(user_raters))
		for i in range(len(remaining_users)):
			user_mapping[remaining_users[i]] = i + len(user_raters)
		assert len(user_mapping.keys()) == len(all_users)
		f=open('user_mapping.json','w')
		f.write(json.dumps(user_mapping))
		f.close()
	else:
		f=open('user_mapping.json')
		user_mapping = json.loads(f.read())
		f.close()
	if not os.path.exists('./item_mapping.json'):
		item_mapping = {}
		for i in range(len(rated_object_ids)):
			item_mapping[rated_object_ids[i]] = i
		remaining_items = list(set(all_items) - set(rated_object_ids))
		for i in range(len(remaining_items)):
			item_mapping[remaining_items[i]] = i + len(rated_object_ids)
		assert len(item_mapping.keys()) == len(all_items)
		f=open('item_mapping.json','w')
		f.write(json.dumps(item_mapping))
		f.close()
	else:
		f=open('item_mapping.json')
		item_mapping = json.loads(f.read())
		f.close()
	if not os.path.exists('./data_stats.txt'):
		f=open("data_stats.txt","w")
		f.write("Total items(reviews and essays): %d\n"%(len(all_items)))
		f.write("Total rated items: %d\n"%(len(rated_object_ids)))
		f.write("Total users: %d\n"%len(all_users))
		f.write("Users who rated: %d\n"%len(user_raters))
		f.write("Users who gave trust statements: %d\n"%len(trust_statement_users))
		f.write("Total trust ratings: %d (%d +ve, %d -ve)"%(len(binary_trusts),
				len(binary_trusts[binary_trusts["value"] == 1]), len(binary_trusts[binary_trusts["value"] == -1])) )
		f.close()
	return ratings, binary_trusts, article_to_user, user_mapping, item_mapping

def get_matrices(ratings,binary_trust,user_mapping,item_mapping):
	if os.path.exists('./user_item_matrix.txt'):
		user_item_matrix = np.matrix(np.loadtxt(open('user_item_matrix.txt')))
	else:
		x = len(ratings['member_id'].unique())
		y = len(ratings['object_id'].unique())
		print x,y
		user_item_matrix = np.zeros((x, y))
		user_index = 0
		item_index = 0
		start = time.clock()
		for index, row in ratings.iterrows():
			try:
				user_item_matrix[ user_mapping[str(row['member_id'])] ][ item_mapping[str(row['object_id'])] ] = row['rating']
			except:
				print user_mapping[str(row['member_id'])], item_mapping[str(row['object_id'])]
			if index % 1000000 == 0:
				print str(time.clock() - start)
				start = time.clock()
		print user_item_matrix.shape
	np.savetxt('user_item_matrix.txt', user_item_matrix)
	return user_item_matrix

if __name__=="__main__":
	ratings, binary_trusts, article_to_user, all_items, all_users = load_data()	
