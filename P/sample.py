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

def runADMM(data):
    #mat = ior.rawData2matrix(data,0, 446, 696)
        #print(mat[:, [0,2]]) # get column 0,2
    admm.run("ADMM")
    ascc.ASCC(data)
    

def runSpark(data1, data2):
    #def parallel():
    conf = SparkConf().setAppName('MyApp')
    sc = SparkContext(conf=conf)
    '''
    path = ""
    if cf.get_platform() == "linux":
        path = "/home/hduser/workspace/MLS/data/"
    else:
        path = "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
    
    dataName = ["mushroom.dat","mushroom_.dat"]
    '''
     
    pspB.runSparkBatch(sc, data1, data2)
    
    pspTree.runSparkFPTree(sc, data1, data2)
    
    #endTime = time.time() - startTime
    #print("total time: "+str(endTime))

    sc.stop()

'''
def linuxDataPath():
    return 
def winDataPath():
    return 
'''

if __name__ == "__main__":
    print("main start")

    #iow.write("test write")
    #ior.read("test read")

    path = ""

    if cf.get_platform() == "linux":
        path = "/home/hduser/workspace/MLS/data/"
    else:
        path = "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
    
    typeR = cf.getRunningConfig(path+"config.txt")
    
    #cf.pythonVer()
    
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
    if typeR ==0 or typeR ==1:
        transactions1 = ior.read2RawData(path+"mushroom.dat",0, 200, 150)
        transactions2 = ior.read2RawData(path+"mushroom.dat",200, 570, 150)
    
    if typeR == 1:   
        fpg.runFPtreeMerge(transactions1, transactions2, 2)
    
    elif typeR == 0:
        bfg.runBatchMerge(transactions1, transactions2, 2)
    
    elif typeR == 2:
        mat = ior.rawData2matrix(path+"data_694_446.dat",0, 446, 696)
        runADMM(mat)
    
    elif typeR == 3:
        runSpark(path+"mushroom.dat", path+"mushroom.dat")
    
    
