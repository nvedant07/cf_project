import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error as mae

item_features = np.loadtxt('./epinions_subset/item_features')
print item_features.shape

trust_embedding = np.loadtxt("./epinions_subset/epinions_embed", skiprows=1)
user_rows = np.array(trust_embedding[:, 0], dtype=np.int16)

x = np.arange(1000)
one_hot = LabelBinarizer().fit_transform(x)
user_item_mat = np.loadtxt("./epinions_subset/user_item_matrix.txt")

user_item_list = []
rating_list = []
for i in range(user_item_mat.shape[0]):
	for j in range(user_item_mat.shape[1]):
		if user_item_mat[i][j] != 0:
			user_item_list.append((i, j))
			rating_list.append(user_item_mat[i][j])

ready_to_split_X = np.array(user_item_list)
ready_to_split_Y = np.array(rating_list)

X_train, X_test, y_train, y_test = train_test_split(ready_to_split_X, ready_to_split_Y, test_size=0.20, )

final_classification_matrix = np.empty((X_train.shape[0], 1128))
c = 0
for (u, i) in X_train:
	row = np.where(user_rows == u)[0]
	if len(row) != 0:
		r = row[0]
		temp = np.hstack((trust_embedding[r][1:], one_hot[i]))
	else:
		unknown_user = np.ones(128)*0.5
		temp = np.hstack((unknown_user, one_hot[i]))
	final_classification_matrix[c] = temp
	c += 1
train_labels = LabelBinarizer().fit_transform(y_train)

c = 0
final_test_matrix = np.empty((X_test.shape[0], 1128))
for (u, i) in X_test:
	row = np.where(user_rows == u)[0]
	if len(row) != 0:
		r = row[0]
		temp = np.hstack((trust_embedding[r][1:], one_hot[i]))
	else:
		unknown_user = np.ones(128)*0.5
		temp = np.hstack((unknown_user, one_hot[i]))
	final_test_matrix[c] = temp
	c += 1

architecture = (500, 100, 50, 10,)
clf = MLPClassifier(solver='adam', hidden_layer_sizes=architecture, verbose=True)
clf.fit(final_classification_matrix, train_labels)

predicted = clf.predict(final_test_matrix)

print predicted[0]

predicted = [ np.argmax(p)+1 for p in predicted]

print mae(y_test, predicted)

print architecture