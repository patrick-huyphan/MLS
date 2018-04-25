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
import dataProcess.batchFP as bfg
import dataProcess.linearRegresion as linear
import SparkFP.pysparkFPTree as pspTree
import SparkFP.pysparkBatch as pspB

import dataProcess.Node as node
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from operator import add



from numpy import arange,array,ones,linalg
from pylab import plot,show
from scipy import sparse

def runADMM():
    mat = ior.rawData2matrix(path+"data_694_446.dat",0, 446, 696)
        #print(mat[:, [0,2]]) # get column 0,2
    ascc.ASCC(mat)

def runSpark():
    #def parallel():
    conf = SparkConf().setAppName('MyApp')
    sc = SparkContext(conf=conf)
    path = ""
    if cf.get_platform() == "linux":
        path = "/home/hduser/workspace/MLS/data/"
    else:
        path = "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
    
    dataName = ["mushroom.dat","mushroom_.dat"]
     
    pspB.runSparkBatch(sc,path + "mushroom_.dat", path + "mushroom_.dat")
    
    pspTree.runSparkFPTree(sc,path + "mushroom_.dat", path + "mushroom_.dat")
    
    #endTime = time.time() - startTime
    #print("total time: "+str(endTime))

    sc.stop()


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
    
    #linear.lnr()
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
    transactions1 = ior.read2RawData(path+"mushroom.dat",0, 200, 150)
    transactions2 = ior.read2RawData(path+"mushroom.dat",200, 570, 150)
    
    bfg.runBatchMerge(transactions2, transactions2)
    
    fpg.runFPtreeMerge(transactions1, transactions2)
    
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
    
    runSpark()
