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
with open('user_mapping.json') as f:
	IDtoUser = json.load(f)
userToID = {v: k for k, v in IDtoUser.iteritems()}

# print(IDtoUser)
# print(max(map(int,IDtoUser.keys())))
# exit(0)

IDtoItem = dict()
with open('item_mapping.json') as f:
	IDtoItem = json.load(f)
itemToID = {v: k for k, v in IDtoItem.iteritems()}

R = np.loadtxt('user_item_matrix.txt')
R = R/100
R_mask = np.copy(R)
R_mask[ np.where(R_mask>0)[0] ] = 1

edges = np.loadtxt('trust_data.txt')

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

for i in range(trustMat_D.shape[0]):
	trustMat_D[i][np.isnan(trustMat_D[i])] = 1e-2		
	trustMat_E[i][np.isnan(trustMat_E[i])] = 1e-2		

if np.sum(np.isnan(trustMat_D)) > 0:
	print("((")
	exit(0)

print(DT.now().time())

def sigmoid(Z):
	return 1 / (1.0 + np.exp(-Z))
	# return Z

def sig_grad(Z):
	return np.multiply( sigmoid(Z),1-sigmoid(Z) )
	# return Z

T = np.array( [ np.nonzero(adjMat[i,:]) for i in range(adjMat.shape[0]) ] )
phi = np.array( [ np.nonzero(adjMat[:,i]) for i in range(adjMat.shape[1]) ] )

# print(T[0])
# exit(0)

m = len(IDtoUser.keys())
n = len(IDtoItem.keys())
lf = 4
U = np.random.randn(m, lf)
# U = np.random.uniform(0,5, size=(m, lf))
V = np.random.randn(lf, n)
# V = np.random.uniform(0,5, size=(lf, n))

alpha = 0.0
u_lam = 0.001
v_lam = 0.001
t_lam = 0.001
lr = 0.001

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
	mae = 0
	c = 0
	for a in range(R.shape[0]):
		for b in range(R.shape[1]):
			if R[a][b] != 0:
				mae += abs(R[a][b]-R_new[a][b])
				c += 1
	mae = mae/float(c)
	print("MAE = ", mae)

	for p in range(m):
		U_p_grad = np.copy(V)

		# print("j1", U_p_grad)
		
		for q in range(lf):
			temp = np.matmul(U[p,:],V)
			# print(lf, p)
			U_p_grad[q,:] *= R_mask[p,:] * sig_grad(temp) * (sigmoid(temp) - R[p,:])
			if np.sum(np.isnan(U_p_grad)) > 0:
				print("**")
				exit(0)
		
		# print("j2", U_p_grad)

		t1 = U[p,:]
		for v in T[p][1]:
			# print( t1.shape, U[v,:].shape, v, p )
			t1 -= trustMat_E[p][v] * U[v,:]

		t2 = np.zeros(t1.shape)
		for v in phi[p][0]:
			t3 = U[v,:]
			for w in T[v][1]:
				t3 -= trustMat_E[v][w] * U[w,:]
			t2 += trustMat_E[v][p] * t3
			# print("t2", t2, trustMat_D[v][p], )

		# print(t1.shape, t2.shape)

		U_p_grad = np.sum(U_p_grad, axis=1)

		# print("j3", U_p_grad)
		
		U_p_grad += u_lam*U[p,:] + t_lam*(t1-t2)

		# print("j4", U_p_grad)
		# print("j4", t1)
		# print("j4", t2)
		
		U[p,:] -= lr*(U_p_grad/m)

		if np.sum(np.isnan(U)) > 0:
			print("##", U[p,:], p)
			print(U_p_grad)
			exit(0)

	print("U = ", U)

	# exit(0)

	for p in range(n):
		V_p_grad = np.copy(U)

		# print("vp1", V_p_grad)
		for q in range(lf):
			temp = np.matmul(V[:,p], U.T)
			# print(sig_grad(temp), sigmoid(temp))
			V_p_grad[:,q] *= R_mask[:,p] * sig_grad(temp) * (sigmoid(temp) - R_mask[:,p])
		
		
		V_p_grad += v_lam * V[:,p]

		# print("vp2", V_p_grad)

		V_p_grad = np.sum(V_p_grad, axis=0)

		# print("vp3", V_p_grad)

		V[:,p] -= lr*(V_p_grad/n)

		# print("vp4", V_p_grad)

		# exit(0)

	if np.sum(np.isnan(V)) > 0:
		print("((")
		exit(0)

	print(len(np.where(V==np.inf)[0]))
	print(np.sum(np.isnan(V)))

	print("V = ", V)