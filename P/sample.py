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

def FPG(transactions, threshold):
    
    startTime = time.time()
    rootTree1 = fpg.FPTree(transactions, threshold, None, None)
    patterns1 = rootTree1.mine_patterns(threshold)

    endTime = time.time() - startTime
    print("FPtree take total time: "+str(endTime)) 
    
    count = 0    
    for patte in patterns1:
        print(str(count) +"\t Pattern: "+ str(patte))
        count +=1
        
    return 0

def FPG2(transactions1, transactions2, threshold):
    
    startTime = time.time()

    rootTree1 = fpg.FPTree(transactions1, threshold, None, None)
    
    #rootTree1.printPattern()

    rootTree2 = fpg.FPTree(transactions2, threshold, None, None)

    #startTime = time.time()
        
    rootTree1.BIT_FPGrowth(rootTree2)

    #rootTree1.printPattern()

    patterns1 = rootTree1.mine_patterns(threshold)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime)) 
    
    count = 0    
    for patte in patterns1:
        print(str(count) +"\t Pattern: "+ str(patte))
        count +=1

    return 0
        
def Batch(transactions, threshold):
    
    startTime = time.time()
    batch1 = bfg.Batch(transactions, threshold)

    endTime = time.time() - startTime
    print("Bit pattern take total time: "+str(endTime)) 
    
    mining_order = sorted(batch1.batch, key=lambda x: x.count, reverse=True)
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch1: "+ str(pattern.value))
        count +=1
    return 0

def Batch2(transactions1, transactions2, threshold):
    
    startTime = time.time()
    batch1 = bfg.Batch(transactions1, threshold)


    batch2 = bfg.Batch(transactions2, threshold)

    endTime = time.time() - startTime
    print("Bit pattern take total time: "+str(endTime)) 
    
    mining_order = sorted(batch1.batch, key=lambda x: x.count, reverse=True)
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch1: "+ str(pattern.value))
        count +=1
    return 0
    
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
    #T10I4D100K
    data = []
    
    #if typeR ==0 or typeR ==1:
    
    if typeR == 1:
        transactions1 = ior.read2RawData(path+"mushroom.dat",0, 550, 160)
        data.append(transactions1[0:100])
        data.append(transactions1[200:320])
        
        transactions3 = ior.read2RawData(path+"T10I4D100K.dat",0, 10000, 1000)
        data.append(transactions3[0:1000])
        data.append(transactions3[1001:2000])
        data.append(transactions3[2001:3000])
        data.append(transactions3[3001:4000])
        data.append(transactions3[4001:5000])
        data.append(transactions3[5001:6000])
        data.append(transactions3[6001:7000])
        data.append(transactions3[7001:8000])
        data.append(transactions3[8001:9000])
        data.append(transactions3[9001:10000])
    
       
        #fpg.runFPtreeMerge(data, 2)
    
        bfg.runBatchMerge(data, 2)
        
    elif typeR == 0:
        transactions = ior.read2RawData(path+"T10I4D100K.dat",0, 10000,1000)
        
        #Batch(transactions, 2)
        
        #FPG(transactions, 2)
        
        transactions3 = transactions[0:5000]#ior.read2RawData(path+"T10I4D100K.dat",0, 5000, 1000)
        data.append(transactions3)
        transactions4 = transactions[5001:10000]#ior.read2RawData(path+"T10I4D100K.dat",5000, 10000, 1000)
        data.append(transactions4)
        
        fpg.runFPtreeMerge(data, 2)
    
        #bfg.runBatchMerge(data, 2)
            
    elif typeR == 2:
        mat = ior.rawData2matrix(path+"data_694_446.dat",0, 446, 696)
        runADMM(mat)
    
    elif typeR == 3:
        runSpark(path+"mushroom.dat", path+"mushroom.dat")
    
    
