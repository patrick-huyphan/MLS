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

def fpgr(transactions):
    '''
    for tran in transactions:
        print(" transaction: "+ str(tran))
    
    patterns = fpg.find_frequent_patterns(transactions, 2)
    for patte in patterns:
        print(" pattern: "+ str(patte))
        
    rules = fpg.generate_association_rules(patterns, 0.7)
    for rule in rules:
        print(" rule: " + str(rule))
    '''
    patterns = fpg.find_frequent_patterns_batch(transactions, 2)
    #for patte in patterns.batch:
        #print(" pattern2: "+ str(patte.value) +" "+ str(patte.count))
    return patterns

def runFPtreeMerge(transactions)
    startTime = time.time()
    
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    return 0
    
def runBathcMerge(transactions)
    startTime = time.time()
    
    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    return 0



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
    transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
    transactions2 = ior.read2RawData(path+"mushroom.dat",0, 500, 100)
    
    runBathcMerge(transactions2)
    
    runFPtreeMerge(transactions2)
    
    batch1 = fpgr(transactions2)
    transactions3 = ior.read2RawData(path+"mushroom.dat",150, 500, 50)
    batch2 = fpgr(transactions3)
    #for patte in batch1.batch:
    #    print(" pattern1: "+ str(patte.value) +" "+ str(patte.count))
    #for patte in batch2.batch:
    #    print(" pattern2: "+ str(patte.value) +" "+ str(patte.count))
    
    batch3 = fpg.mergeBatch(batch1, batch2)
    for patte in batch3.batch:
        print(" batch3: "+ str(patte.value) +" "+ str(patte.count))
        
    #if cf.get_platform() == "linux":
        #data = ior.read2SparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.csv")
        #ior.saveSparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.dat",data)
    mat = ior.rawData2matrix(path+"data_694_446.dat",0, 446, 696)
        #print(mat[:, [0,2]]) # get column 0,2
    ascc.ASCC(mat)
    current_dir = os.getcwd() #pathlib.Path("/../../data/data_694_446.dat").parent
    print(current_dir)
    #current_file = pathlib.Path(__file__)
    #print(current_file)
        #sA = sparse.csr_matrix(mat)
        #print(sA)
        #sD = sparse.csr_matrix.todense(sA)
        #print(sD)
    #ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
