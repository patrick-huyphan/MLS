#!/usr/bin/python
# MLS

import numpy as np
from scipy import spatial
from numpy import arange,array,ones,linalg
from pylab import plot,show
from scipy import sparse

def run(name):
    print("hello" +name)

    
'''
 *
 * @author patrick_huy update in X update in rho theta
 * SCC is quadratic programming:
 * minimize->x: (1/2)x_TPx+qTx+r
 * Gx ≤ h 
 * Ax = b
 * where P ∈ S^+n, G ∈ R^mXn and A ∈ R^pXn
 * 
 * For convex clustering min fx + gx +h
 * fx: ||A - X||, cost of node, min of sum square b.w node X to control node A. A is inference centroid node. if Ai == Aj, Xi and Xj has same centroid and in same cluster.
 * gx: lambda wij||Xi - Xj||, cost of edge, the edge cost is a sum of norms of differences of the adjacent edge variables
 * h:  lambda2 u||X|| for sparse data
 *
 * should design matrix data and vector (row and column of matrix)to easy process with sparse data: <<indexR, indexC>, value>
'''
def cosine_similarity(v1,v2):
   return spatial.distance.cosine(v1, v2)
   #return ( np.sum(vector*matrix,axis=1) / ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )[::-1]

def initEdge(data, d):
    edge = []
    
    for i in range(0,d):
        for j in range(i+1,d):
            sim = cosine_similarity(data[:,i],data[:,j])
            if sim > 0.13:
                edge.append([i,j,sim])
    #print(edge)
    '''
    for v1 in data:
        for v2 in data:
            sim = cosine_similarity(v1,v2)
            if sim > 0.13:
                edge.append([v1,v2,sim])
    '''
    return edge

def init(data):
    A = []
	return A
'''
U and V has [des, src, [nodedata]]
'''
def initUV(edges, matrix):
    u = []
    v = []
    for egde in edges:
        u.append([egde[0],egde[1],[0]])
        u.append([egde[1],egde[0],[0]])
        v.append([egde[0],egde[1],[matrix[egde[0]]]])
        v.append([egde[1],egde[0],[matrix[egde[1]]]])
    return (u,v)

def initV(edges):
    return 0
'''

''' 
def updateX(d, edge, V, U, A, B):
    x = []
	for i in range(0,d[0]):
		for(e in edge):
			if e[0]:
				x[i] = [0];
			if e[1]:
				x[i] = [0];
	return x

'''

'''     
def updateUV(edge, U,V,X):
    u = []
    v = []
    for(e in edge):
		u = U
		v = V
    return u,v
    
def updateV():
    return 0
    
def updateU():
    return 0


def primalResidual():
	return 0
	
def dualResidual():
	return 0
	
'''

''' 
def checkStop(edge, X0, U0, V0, V):
    return 0
    

'''

''' 
def getCluster(edge, X0):
    return 0
    
'''

''' 
def getPresentMat(edge, X0):
    return 0
'''
 * with convex optimization, set start point and solve problem with linear
 * or quadratic programming:
 * min 1/2sum||A-X|| + lambda sum(rho||X-X||) + lambda_2 sum (u||X||) 
 * Lagrange = sum() + sum() 
 * X = min fx + sum (1/2||x-v+u||) 
 * V = min lambda *w ||v-v|| + rho/2(||x-z+u||+||x-z+u||) 
 * U = u + rho(x-z)
 *
 * @param _Matrix
 * @param _lambda: in range 1->2
 * @param _lambda2
 * @param _rho
 * @param _e1
 * @param _e2
 * @throws IOException
'''
def SCC(data):
    sA = sparse.csr_matrix(data)
    #print(sA)
    sD = sparse.csr_matrix.todense(sA)
    #print(sD)
    d = np.shape(data)
    print(str(d) +" "+ str(d[0]) +" "+str(d[1]))
    '''
	edge: sim b.w 2 node
	'''
	edge = initEdge(data, d[0])
	'''
	init data: 
    '''
	A = init(data)
    V0 = [] # List of edge 
    U0 = [] # List of edge
    X0 = [] # matrix for get cluster
    B = []
	U,V = initUV(edge, data)
    
    loop = 0
    maxloop = 100
    while(loop< maxloop):
        X = X0
        U = U0
        V = V0
		
        X = updateX(d, edge, V, U, A, B) 

        U,V = updateUV(edge, V, U, X)
        #V = updateV(V, U)
        #U = updateU(U, V)
        
        if (checkStop(edge, X0, U0, V0, V) and (loop > 1)):
            print(" SCC STOP at " + loop)
            break

        loop = loop+1
    
    getCluster(edge, X)
    getPresentMat(edge, X)

def updateU2():
    return 0
    
def updateV2():
    return 0

'''

''' 	
def updateUV2(edge, V, U, X):
    u = []
    v = []
    for(e in edge):
		u = U
		v = V
		
    return u,v
'''
enhence SCC, speed up to reach stop condition
'''
def FSCC(data):
    sA = sparse.csr_matrix(data)
    #print(sA)
    sD = sparse.csr_matrix.todense(sA)
    #print(sD)
    d = np.shape(data)
    print(str(d) +" "+ str(d[0]) +" "+str(d[1]))
    edge = initEdge(data, d[0])
    A = init(data)
    V0 = []
    U0 = []
    X0 = []
    U,V = initUV(edge, data)
    #V = initV()
    B = []
    loop = 0
    maxloop = 100
    while(loop< maxloop):
        X = X0
        U = U0
        V = V0
        
        X = updateX(d, edge, V, U, A, B) 

        U,V = updateUV(edge, V, U, X)
        #V = updateV(V, U)
        #U = updateU(U, V)
        
        if (checkStop(edge, X0, U0, V0, V) and (loop > 1)):
            print(" SCC STOP at " + loop)
            break

        U,V = updateUV2(V, U, X)
        loop = loop+1
    
    getCluster()
    getPresentMat()    

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w