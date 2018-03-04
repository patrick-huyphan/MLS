#!/usr/bin/python
# MLS

import numpy as np
from scipy import spatial

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

def initEdge(data):
    edge = []

    for i in range(0,numberOfVertices):
        for j in range(i+1,numberOfVertices):
            sim = cosine_similarity(data[i],data[j])
            if sim > 0.13:
                edge.append([j,j,sim])
    '''
    for v1 in data:
        for v2 in data:
            sim = cosine_similarity(v1,v2)
            if sim > 0.13:
                edge.append([v1,v2,sim])
    '''
    return edge

def init(data):

def initUV(edges, matrix):
    u = []
    v = []
    for egde in edges:
        u.append([egde[0],egde[1],[0]])
        u.append([egde[1],egde[0],[0]])
        v.append([egde[0],egde[1],[matrix[egde[0]]]])
        v.append([egde[1],egde[0],[matrix[egde[1]]]])
    return u,v

def initV(edges):

def calcD():

def updateX():

def updateUV(U,V,X):
    u = []
    v = []
    
    return u,v
    
def updateV():

def updateU():

def checkStop():

def getCluster():

def getPresentMat():

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
    edge = initEdge(data);
    init(data)
    V0 = []
    U0 = []
    U,V = initUV(edge, data)
    #V = initV()
    B = []
    
    while(loop< maxloop):
        X = X0
        U = U0
        V = V0
        
        for i in range(0,numberOfVertices):
            X = updateX(i, V, U,B) 

        U,V = updateUV(V, U, X)
        #V = updateV(V, U)
        #U = updateU(U, V)
        
        if (checkStop(X0, U0, V0, V) && (loop > 1)):
        {
            print(" SCC STOP at " + loop)
            break
        }
        loop++
    
    getCluster()
    getPresentMat()

def updateU2():

def updateV2():
    
def FSCC(data):
    initEdge(data);
    init(data)
    V0 = []
    U0 = []
    U = initU()
    V = initV()
    while(loop< maxloop):
        X = X0
        U = U0
        V = V0
        for (int i = 0; i < numberOfVertices; i++):
            D = calcD(i, V, U)
            updateX(i, D) 
        }
        V = updateV(V, U)
        U = updateU(U, V)
        
        if (checkStop(X0, U0, V0, V) && (loop > 1)):
        {
            print(" SCC STOP at " + loop)
            break
        }
        updateU2()
        updateV2()
        loop++
    
    getCluster()
    getPresentMat()    
