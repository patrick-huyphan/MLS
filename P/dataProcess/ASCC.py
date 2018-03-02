#!/usr/bin/python
# MLS

import numpy as np

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

def init():

def initU():

def initV():

def calcD():

def updateX():

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
def SCC():
    init()
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
        loop++
    
    getCluster()
    getPresentMat()

def updateU2():

def updateV2():
    
def FSCC():
    init()
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