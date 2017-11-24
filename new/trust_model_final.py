import numpy as np
import json
import math
import networkx as nx
from datetime import datetime as DT
import h5py
from scipy.optimize import minimize
import pickle
from numpy import linalg as LA

IDtoUser = dict()
with open('./epinions_subset/user_mapping.json') as f:
	IDtoUser = json.load(f)
userToID = {v: k for k, v in IDtoUser.iteritems()}

# print(IDtoUser)
# print(max(map(int,IDtoUser.keys())))
# exit(0)

IDtoItem = dict()
with open('./epinions_subset/item_mapping.json') as f:
	IDtoItem = json.load(f)
itemToID = {v: k for k, v in IDtoItem.iteritems()}

R = np.loadtxt('./epinions_subset/user_item_matrix.txt')
R_mask = np.copy(R)
R_mask[ np.where(R_mask>0)[0] ] = 1

edges = np.loadtxt('./epinions_subset/trust_data.txt')

print(DT.now().time())

edgeList = [ str(userToID[int(ed[0])])+','+str(userToID[int(ed[1])]) for ed in edges if int(ed[0]) in userToID and int(ed[1]) in userToID ]
g = nx.parse_edgelist( edgeList, delimiter=',', create_using=nx.DiGraph(), nodetype=int)

missing_nodes = [ int(uid) for uid in IDtoUser if int(uid) not in g.nodes() ]
g.add_nodes_from(missing_nodes)

adjMat = nx.adjacency_matrix(g)

degCent = nx.in_degree_centrality(g)
degCent = [ degCent[int(uid)] for uid in IDtoUser ]
eigenCent = nx.eigenvector_centrality(g)
eigenCent = [ eigenCent[int(uid)] for uid in IDtoUser ]

# simMat = [ float(np.dot(R[int(i),:],R[int(j),:])) / (1+math.sqrt(np.sum(R[int(i),:][np.where(R[int(j),:]>0)[0]]**2)*np.sum(R[int(j),:][np.where(R[int(i),:]>0)[0]]**2))) for uid in IDtoUser.keys() for i,j in zip( [uid]*len(IDtoUser.keys()), IDtoUser.keys() ) ]
# print(len(simMat))

# simMat = np.reshape(simMat, (len(IDtoUser.keys()),len(IDtoUser.keys())))

# print(simMat.shape)

# with h5py.File('sim_pairs.h5', 'w') as hf:
# 	hf.create_dataset('S', data=simMat)

simMat = None
with h5py.File('sim_pairs.h5', 'r') as hf:
	simMat = hf['S'][:]


# beta = 0.5

# trustMat = np.array( [ [ beta*(float(simMat[int(i)][int(k)])/np.sum(simMat[int(i),:])) + (1-beta)*(float(eigenCent[int(k)])/sum(eigenCent)), beta*(float(simMat[int(i)][int(k)])/np.sum(simMat[int(i),:])) + (1-beta)*(float(degCent[int(k)])/sum(degCent)) ] for uid in IDtoUser for i,k in zip( [uid]*len(IDtoUser.keys()), IDtoUser.keys() ) ] )
# trustMat_E = trustMat[:,0]
# trustMat_D = trustMat[:,1]

# print( len(trustMat_E), len(trustMat_E) )

# trustMat_E = np.reshape( trustMat_E, (len(IDtoUser.keys()),len(IDtoUser.keys())) )
# trustMat_D = np.reshape( trustMat_D, (len(IDtoUser.keys()),len(IDtoUser.keys())) )

# print(trustMat_E.shape)
# print(trustMat_D.shape)

# with h5py.File('trust_pairs.h5', 'w') as hf:
# 	hf.create_dataset('E', data=trustMat_E)
# 	hf.create_dataset('D', data=trustMat_D)

trustMat_E = None
trustMat_D = None
with h5py.File('trust_pairs.h5', 'r') as hf:
	trustMat_E = hf['E'][:]
	trustMat_D = hf['D'][:]

print(DT.now().time())

def sigmoid(Z):
	return 1 / (1.0 + np.exp(-Z))

def sig_grad(Z):
	return np.multiply( sigmoid(Z),1-sigmoid(Z) )
def relu(Z):
	copy = np.array(Z)
	copy[Z == 0] = 0
	return copy
def relu_grad(Z):
	print Z
	copy = np.array(Z)
	copy[Z<0] = 0
	copy[Z>0] = 1
	return copy

T = np.array( [ np.nonzero(adjMat[i,:]) for i in range(adjMat.shape[0]) ] )
phi = np.array( [ np.nonzero(adjMat[:,i]) for i in range(adjMat.shape[1]) ] )

# print(T[0])
# exit(0)

m = len(IDtoUser.keys())
n = len(IDtoItem.keys())
lf = 4
U = np.random.randn(m, lf)
V = np.random.randn(lf, n)
alpha = 0.5
u_lam = 0.001
v_lam = 0.001
t_lam = 0.001
lr = 0.5

# *args: R, trustMat
def F_cost(params, *args):
	U, V = params[:m*lf].reshape(m,lf), params[m*lf:].reshape(lf,n)
	R_new = np.matmul(U, V)
	return np.sum(np.multiply( R_mask, (args[0] - sigmoid(alpha*R_new + (1-alpha)*np.matmul(args[1],R_new)))**2 )) + (u_lam/2)*LA.norm(U) + (v_lam/2)*LA.norm(V)

# res = minimize(fun=F_cost, x0=np.hstack([U.flatten(),V.flatten()]), args=(R,trustMat_D), options={'disp':True})

# print(res['X'])
# print(res['message'])

# exit(0)

# for i in range(200):
# 	R_new = np.matmul(U,V)
# 	print("MAE = ", np.mean(np.abs(R_new-R)))

# 	U_grad = [ alpha*R_mask[i][j]*sig_grad( alpha*np.dot(U[i,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[i][k]*np.dot(U[k,:],V[:,j]) for k in T[i] ])) )*(sigmoid( alpha*np.dot(U[i,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[i][k]*np.dot(U[k,:],V[:,j]) for k in T[i] ])) )-R[i][j])*V[:,j] for i in range(m) for j in range(n) ]
# 	U_grad += [ (1-alpha)*R_mask[p][j]*sig_grad( alpha*np.dot(U[p,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[p][k]*np.dot(U[k,:],V[:,j]) for k in T[p] ])) )*(sigmoid( alpha*np.dot(U[p,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[p][k]*np.dot(U[k,:],V[:,j]) for k in T[p] ])) )-R[p][j])*trustMat_D[p][i]*V[:,j] for i in range(m) for p in phi[i] for j in range(n) ]
# 	U_grad += np.array( [ u_lam*U[i,:] for i in range(m) ] )

# 	V_grad = [ R_mask[i][j]*sig_grad( alpha*np.dot(U[i,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[i][k]*np.dot(U[k,:],V[:,j]) for k in T[i] ])) )*(sigmoid( alpha*np.dot(U[i,:],V[:,j])+(1-alpha)*(np.sum([ trustMat_D[i][k]*np.dot(U[k,:],V[:,j]) for k in T[i] ])) )-R[i][j])*( alpha*U[i,:] + (1-alpha)*(np.sum([ trustMat_D[i][k]*U[k,:] for k in T[i] ])) ) for j in range(n) for i in range(m) ]
# 	V_grad += np.array( [ v_lam*V[:,j] for j in range(n) ] )
# 	V_grad = V_grad.T

# 	U -= lr*U_grad
# 	V -= lr*V_grad

for i in range(200):
	R_new = np.matmul(U,V)
	print("MAE = ", np.mean(np.abs(R_new-R)))

	for p in range(n):
		V_p_grad = np.copy(U)
		for q in range(lf):
			temp = np.matmul(V[:,p], U.T)
			V_p_grad[:,q] *= R_mask[:,p] * relu_grad(temp) * (relu(temp) - R_mask[:,p])
		
		V_p_grad += v_lam * V[:,p]

		V_p_grad = np.sum(V_p_grad, axis=0)
		V[:,p] -= lr*V_p_grad

	print(len(np.where(V==np.inf)[0]))
	print(np.sum(np.isnan(V)))

	for p in range(m):
		U_p_grad = np.copy(V)
		for q in range(lf):
			temp = np.matmul(U[i,:],V)
			# print(lf, p)
			U_p_grad[q,:] *= R_mask[p,:] * relu_grad(temp) * (relu(temp) - R[p,:])
		
		t1 = U[p,:]
		for v in T[p][1]:
			# print( t1.shape, U[v,:].shape, v, p )
			t1 -= trustMat_D[p][v] * U[v,:]

		t2 = np.zeros(t1.shape)
		for v in phi[p][0]:
			t3 = U[v,:]
			for w in T[v][1]:
				t3 -= trustMat_D[v][w] * U[w,:]
			t2 += trustMat_D[v][p] * t3

		# print(t1.shape, t2.shape)

		U_p_grad = np.sum(U_p_grad, axis=1)

		U_p_grad += u_lam*U[p,:] + t_lam*(t1-t2)

		U[p,:] -= lr*U_p_grad

	print(len(np.where(V==np.inf)[0]))
	print(np.sum(np.isnan(V)))

