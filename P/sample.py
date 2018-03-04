#!/home/hduser/Downloads/Or/bin
# MLS
import sys
sys.path.insert(0, '/home/hduser/workspace/MLS/P/')

import config.config as cf
import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
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
    for patte in patterns.batch:
        print(" pattern2: "+ str(patte.value) +" "+ str(patte.count))

def linuxDataPath():
    return "/home/hduser/workspace/MLS/data/"
def winDataPath():
    return ""
    
if __name__ == "__main__":
    print("main start")

    iow.write("test write")
    ior.read("test read")
    admm.run("ADMM")
    
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
    #transactions2 = ior.read2RawData("/home/hduser/workspace/MLS/data/mushroom.dat",0, 500, 100)
    #transactions3 = ior.read2RawData("/home/hduser/workspace/MLS/data/mushroom.dat",150, 500, 50)
    #fpgr(transactions2)
    #fpgr(transactions3)
    
    if cf.get_platform() == "linux":
        #data = ior.read2SparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.csv")
        #ior.saveSparseMatrix("/home/hduser/workspace/MLS/data/data_694_446.dat",data)
        mat = ior.rawData2matrix("/home/hduser/workspace/MLS/data/data_694_446.dat",0, 446, 696)
        #print(mat[:, [0,2]]) # get column 0,2
        sA = sparse.csr_matrix(mat)
        print(sA)
        sD = sparse.csr_matrix.todense(sA)
        print(sD)
    #ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
