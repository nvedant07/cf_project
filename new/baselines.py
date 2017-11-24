
# coding: utf-8

# In[9]:

import h5py
import pandas as pd
import numpy as np
import json, time
from sklearn.metrics import mean_absolute_error as MAE


# In[2]:

f = open('./epinions_subset/user_mapping.json')
user_mapping = json.loads(f.read())
f.close()
f = open('./epinions_subset/item_mapping.json')
item_mapping = json.loads(f.read())
f.close()


# In[3]:

def five_fold_split(matrix):
    split_matrices = []
    valid_indices = matrix.nonzero()
    test_set_size = int(0.2 * len(valid_indices[0]))
    for i in range(5):
        start = int(i * 0.2 * len(valid_indices[0]))
        end = int((i+1) * 0.2 * len(valid_indices[0]))
        test_indices = (valid_indices[0][start:end], valid_indices[1][start:end])
        
        if start > 0 and end < len(valid_indices[0]):
            train_indices = (np.append(valid_indices[0][0:start], valid_indices[0][end:len(valid_indices[0])]),
                                       np.append(valid_indices[1][0:start], valid_indices[1][end:len(valid_indices[0])]) )
        elif start > 0:
            train_indices = (valid_indices[0][0:start], valid_indices[1][0:start])
        elif end < len(valid_indices[0]):
            train_indices = (valid_indices[0][end:len(valid_indices[0])], valid_indices[1][end:len(valid_indices[0])])
        
        train_matrix = np.zeros(matrix.shape)
        train_matrix[train_indices] = matrix[train_indices]
        test_matrix = np.zeros(matrix.shape)
        test_matrix[test_indices] = matrix[test_indices]
        split_matrices.append((train_matrix, test_matrix))
    count = 0
    for tup in split_matrices:
        print (len(valid_indices[0]), len(tup[0].nonzero()[0]) + len(tup[1].nonzero()[0]))
    return split_matrices


# In[4]:

def train_item_based(user_by_item):
    print (user_by_item.shape)
    mask = np.ones(user_by_item.shape)
    mask[user_by_item.nonzero()] = 0
    masked_matrix = np.ma.masked_array(user_by_item, mask=mask)
    return np.ma.corrcoef(masked_matrix, rowvar=False).data


# In[5]:

def train_user_based(user_by_item):
    item_by_user = user_by_item.T
    print (item_by_user.shape)
    mask = np.ones(item_by_user.shape)
    mask[item_by_user.nonzero()] = 0
    masked_matrix = np.ma.masked_array(item_by_user, mask=mask)
    return np.ma.corrcoef(masked_matrix, rowvar=False).data


# In[6]:

user_item_matrix = np.loadtxt('./epinions_subset/user_item_matrix.txt')
split_matrices = five_fold_split(user_item_matrix)


# In[7]:

for i in range(5):
    print (i)
    item_similarity = train_item_based(split_matrices[i][0])
    np.savetxt('./epinions_subset/item_similarity_%d.txt'%i, item_similarity)


# In[22]:

def nmae_item_item(test_matrix, train_matrix, similarity_matrix):
    indices = test_matrix.nonzero()
    y_true = []
    y_pred = []
    for i,j in zip(indices[0], indices[1]):
        y_true.append(test_matrix[i][j])
        item_sum = np.sum(similarity_matrix[j]) - 1
        if item_sum == 0:
            predicted_rating = np.mean(train_matrix[:,j][train_matrix[:,j].nonzero()])
        else:
            predicted_rating = np.mean(train_matrix[:,j][train_matrix[:,j].nonzero()]) + np.dot(similarity_matrix[j].flatten(), train_matrix[i].flatten())/item_sum
        y_pred.append(np.round(predicted_rating))
    return MAE(y_true, y_pred)/4


# In[23]:

##prediction item item
for i in range(5):
    similarity = np.loadtxt('./epinions_subset/item_similarity_%d.txt'%i)
    print (nmae_item_item(split_matrices[i][1], split_matrices[i][0], similarity))


# In[ ]:

for i in range(5):
    print (i)
    user_similarity = train_user_based(split_matrices[i][0])
    np.savetxt('./epinions_subset/user_similarity_%d.txt'%i, user_similarity)


# In[ ]:

def nmae_user_user(test_matrix, train_matrix, similarity_matrix):
    indices = test_matrix.nonzero()
    y_true = []
    y_pred = []
    for i,j in zip(indices[0], indices[1]):
        y_true.append(test_matrix[i][j])
        user_sum = np.sum(similarity_matrix[i]) - 1
        if user_sum == 0:
            predicted_rating = np.mean(train_matrix[i,:][train_matrix[i,:].nonzero()])
        else:
            numerator = 0
            for k in range(test_matrix.shape[0]):
                if k != i and train_matrix[k][j] != 0:
                    u_a_mean = np.mean(train_matrix[k,:][train_matrix[k,:].nonzero()])
                    numerator += similarity_matrix[i][k] * (train_matrix[k][j] - u_a_mean)
            predicted_rating = np.mean(train_matrix[i,:][train_matrix[i,:].nonzero()]) + numerator/user_sum
        y_pred.append(np.round(predicted_rating))
    return MAE(y_true, y_pred)/4


# In[ ]:

##prediction user user
for i in range(5):
    similarity = np.loadtxt('./epinions_subset/user_similarity_%d.txt'%i)
    print (nmae_user_user(split_matrices[i][1], split_matrices[i][0], similarity))


# In[ ]:



