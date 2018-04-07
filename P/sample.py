#!/home/hduser/Downloads/Or/bin
# MLS
import sys
sys.path.insert(0, '/home/hduser/workspace/MLS/P/')
import os
import time
#import pathlib
import config.config as cf
import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.ASCC as ascc
import dataProcess.fpgrowth as fpg

from numpy import arange,array,ones,linalg
from pylab import plot,show
from scipy import sparse

def lnr():
    xi = arange(0,9)
    A = array([ xi, ones(9)])
    # linearly generated sequence
    y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
    w = linalg.lstsq(A.T,y)[0] # obtaining the parameters

    # plotting the line
    line = w[0]*xi+w[1] # regression line
    plot(xi,line,'r-',xi,y,'o')
    show()


def runFPtreeMerge(transactions1,transactions2):

    startTime = time.time()

    #for tran in transactions:
    #    print(" transaction: "+ str(tran))
    
    rootTree1 = fpg.buildFPTree(transactions1, 2)
    
    rootTree1.printTree()
    
    rootTree2 = fpg.buildFPTree(transactions2, 2)
    rootTree2.printTree()
    
    rootTree3 = rootTree1.mergeTree(rootTree2)
    rootTree3.printTree()
    
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime)) 
    '''
    patterns1 = find_frequent_patterns(tree3, 2)
    for patte in patterns:
        print(" pattern: "+ str(patte))
        
    rules = fpg.generate_association_rules(patterns1, 0.7)
    for rule in rules:
        print(" rule: " + str(rule))
    
    #fpTree1_ = fpg.find_frequent_patterns(transactions, 2)
    #patterns2 = fpg.find_frequent_patterns(transactions, 2)
    
    #patterns3 = fpg.mergeTree(patterns1 ,patterns2)
    
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    '''
    return 0
    
def runBathcMerge(transactions1,transactions2):
    startTime = time.time()
    
    batch1 = fpg.find_frequent_patterns_batch(transactions1, 2)
    
    batch2 = fpg.find_frequent_patterns_batch(transactions2, 2)
    
    batch3 = fpg.mergeBatch(batch1, batch2)
    
    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    
    for patte in batch3.batch:
        print(" batch3: "+ str(patte.value) +" "+ str(patte.count))


    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    return 0

def runADMM():
    mat = ior.rawData2matrix(path+"data_694_446.dat",0, 446, 696)
        #print(mat[:, [0,2]]) # get column 0,2
    ascc.ASCC(mat)

def linuxDataPath():
    return "/home/hduser/workspace/MLS/data/"
def winDataPath():
    return "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
    
if __name__ == "__main__":
    print("main start")

    iow.write("test write")
    ior.read("test read")
    admm.run("ADMM")
    path = ""

    if cf.get_platform() == "linux":
        path = linuxDataPath()
    else:
        path = winDataPath()
    
    #lnr()
    '''
    transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
    '''
    transactions2 = ior.read2RawData(path+"mushroom.dat",0, 100, 100)
    
    #runBathcMerge(transactions2, transactions2)
    
    runFPtreeMerge(transactions2, transactions2)
    
    #if cf.get_platform() == "linux":
        #data = ior.read2SparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.csv")
        #ior.saveSparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.dat",data)

    current_dir = os.getcwd() #pathlib.Path("/../../data/data_694_446.dat").parent
    print(current_dir)
    #current_file = pathlib.Path(__file__)
    #print(current_file)
        #sA = sparse.csr_matrix(mat)
        #print(sA)
        #sD = sparse.csr_matrix.todense(sA)
        #print(sD)
    #ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
