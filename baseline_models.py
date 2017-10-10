import numpy as np
import pandas as pd
import math, time, h5py
from data_analysis import load_data, get_matrices
from scipy import io, sparse

def find_pcc(user_item_matrix, user_max, significance_threshold=None):
	similarity_matrix = np.zeros((user_max,user_max))
	
	count = 0
	for i in range(user_max):
		user1_movies = np.array(user_by_movie[i].T).flatten()
		if i not in user_means:
			nonzero = user1_movies.nonzero()
			mean_1 = np.average(user1_movies[nonzero])
			user_means[i] = mean_1
		else:
			mean_1 = user_means[i]

		start = time.clock()

		for j in range(1, user_max+1):
			if i != j:
				user2_movies = np.array(user_by_movie[j].T).flatten()
				bitmap1 = user1_movies.nonzero()[0]
				bitmap2 = user2_movies.nonzero()[0]
				if bitmap2.shape[0] == 0:
					continue
				if bitmap1.shape[0] > bitmap2.shape[0]:
					bitmap2 = np.concatenate( (bitmap2, np.array(np.zeros( (1, bitmap1.shape[0] - bitmap2.shape[0]) ) ).flatten() ), axis = 0)
				else:
					bitmap1 = np.concatenate( (bitmap1, np.array(np.zeros( (1, bitmap2.shape[0] - bitmap1.shape[0]) ) ).flatten() ), axis = 0)
				bitmap = np.intersect1d(np.array(bitmap1).flatten(), np.array(bitmap2).flatten())
				common_ratings_user1 = np.array(user1_movies[bitmap.astype(int)]).flatten()
				common_ratings_user2 = np.array(user2_movies[bitmap.astype(int)]).flatten()
				
				if len(bitmap) > 0:
					if j not in user_means:
						mean_2 = np.average(user2_movies[user2_movies.nonzero()])
						user_means[j] = mean_2
					else:
						mean_2 = user_means[j]
					try:
						common_ratings_user1 -= mean_1
						common_ratings_user2 -= mean_2
						sq_1 = np.array(np.square(common_ratings_user1)).flatten()
						sq_2 = np.array(np.square(common_ratings_user2)).flatten()
						s_score = np.sum( common_ratings_user1 * common_ratings_user2 )/ np.sqrt( np.sum(sq_1) * np.sum(sq_2) )
						if significance_threshold is not None and len(bitmap) < significance_threshold:
							s_score *= (len(bitmap)*s_score/significance_threshold)
						similarity_matrix[i][j] = s_score
					except Exception as e:
						similarity_matrix[i][j] = 0	
				else:
					similarity_matrix[i][j] = 0

		print 'Time Taken: ' + str(time.clock() - start)

		count += 1
		# if count % 10 == 0:
		print '\t%d/%d'%(count,user_max)
	return similarity_matrix

if __name__ == "__main__":
	ratings, binary_trusts, article_to_user, user_mapping, item_mapping = load_data()
	print 'data loaded'
	user_item_matrix = get_matrices(ratings, binary_trusts, user_mapping, item_mapping)
	print 'matrix made'
	# user_item_matrix = sparse.coo_matrix(np.matrix(user_item_matrix))
	# io.mmwrite('user_item_matrix',user_item_matrix)
	# np.savez_compressed('user_item_matrix',data=user_item_matrix)