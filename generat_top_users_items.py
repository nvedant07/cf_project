import numpy as np
import h5py
import argparse
import operator

parser = argparse.ArgumentParser()
parser.add_argument("rating_path")
parser.add_argument("trust_path")

args = parser.parse_args()
rating_data = np.loadtxt(args.rating_path, delimiter=' ', dtype=int)

path = "final_data"
save = h5py.File(path, "w")
user_dic = {}

for user, item, r in rating_data:
	if user not in user_dic.keys():
		user_dic[user] = 1
	else:
		user_dic[user] += 1
print("made user dic")
top_users = sorted(user_dic.items(), key=operator.itemgetter(1), reverse=True)[:5000]
top_users_number = [x for (x, y) in top_users]
save.create_dataset('users', data = top_users_number)
print("sorted user dic")

item_dic = {}
for user, item, r in rating_data:
	if item not in item_dic.keys():
		item_dic[item] = 1
	else:
		item_dic[item] += 1

print("made item dic")
top_items = sorted(item_dic.items(), key=operator.itemgetter(1), reverse=True)[:10000]
top_items_number = [x for (x, y) in top_items]
save.create_dataset('items', data = top_items_number)
print("sorted item dic")
save.close()

# trust_data = np.loadtxt(args.trust_path, delimiter=' ', dtype=int, usecols=(1,2,3))
# print(trust_data)