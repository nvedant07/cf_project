import numpy as np
import h5py
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("rating_path")

args = parser.parse_args()
rating_data = np.loadtxt(args.rating_path, delimiter=' ', dtype=int)
# uniq = np.unique(rating_data[:, 0])

# path = "final_data"
# data = h5py.File(path, "r")
# users = data['users'][:]
# items = data['items'][:]

# users = np.sort(users)
# items = np.sort(items)

# user_mapping = {}
# for idx, u in enumerate(users):
# 	user_mapping[u] = idx


# item_mapping = {}
# for idy, i in enumerate(items):
# 	if idy %1000 == 0:
# 		print idy
# 	item_mapping[i] = idy

user_item_mat = np.zeros((49290, 139738), dtype=np.int8)
c = 0
for x, y, r in rating_data:
	c += 1
	if c %10000 == 0:
		print c
	# 	print x
	# 	print x in users
	# 	print user_mapping[x]
	# if y in items:
		# if x == top:
			# print(user_mapping[x])
		# print(x)
	user_item_mat[x-1][y-1] = r

# print(user_item_mat[40].shape)
# print(len(np.where(user_item_mat[40] == 0)[0]))
l = []
for i in range(user_item_mat.shape[0]):
	xx = np.where(user_item_mat[i] == 0)
	l.append(len(xx[0]))

print(min(l))
print(max(l))

save = h5py.File("user_item_mat_all_users_all_items", "w")
save.create_dataset('matrix', data = user_item_mat)
save.close()

# with open('item_mapping', 'w') as outfile:
#     pickle.dump(item_mapping, outfile, protocol=pickle.HIGHEST_PROTOCOL)
# save.create_dataset('user_mapping', data = user_mapping)
# save.create_dataset('item_mapping', data = item_mapping)