import pandas as pd

ratings = pd.read_csv("rating.txt", sep="\t", header=None,
							names=["object_id", "member_id", "rating", "status", "creation", "last_modified", "type", "vertical_id"],
							dtype={"object_id" : int, "member_id" : int, "rating" : int})

userids = set(ratings['member_id'])

ratings.set_index(['member_id','object_id'], inplace=True)
print ratings.head()

train = open('ratings_train.txt', 'a')
test = open('ratings_test.txt', 'a')
for user in userids:
	user_ratings = ratings.loc[user].reset_index()
	start = 0
	end = int(0.2 * len(user_ratings))
	to_write = user_ratings.iloc[start:end, ]
	if len(to_write) > 0:
		to_write.loc[:,'member_id'] = user
		to_write = to_write[["object_id", "member_id", "rating", "status", "creation", "last_modified", "type", "vertical_id"]]
		to_write.to_csv(test, header=False, sep='\t', index=False)

	if start > 0:
		to_write = user_ratings.iloc[0:start, ]
		to_write.loc[:,'member_id'] = user
		to_write = to_write[["object_id", "member_id", "rating", "status", "creation", "last_modified", "type", "vertical_id"]]
		to_write.to_csv(train, header=False, sep='\t', index=False)
	if end < len(user_ratings):
		to_write = user_ratings.iloc[end:len(user_ratings), ]
		to_write.loc[:,'member_id'] = user
		to_write = to_write[["object_id", "member_id", "rating", "status", "creation", "last_modified", "type", "vertical_id"]]
		to_write.to_csv(train, header=False, sep='\t', index=False)

train.close()
test.close()