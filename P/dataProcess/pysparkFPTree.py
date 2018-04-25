import os
import sys
sys.path.insert(0, '/home/hduser/workspace/MLS/P/')

import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.fpgrowth as fpg
import config.config as cf
import dataProcess.ASCC as ascc
import dataProcess.Node as node
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from operator import add
from numpy import arange,array,ones,linalg
from pylab import plot,show
from scipy import sparse
import time

import random


class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []
        #self.batch = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None
        
    def intercept(self, other):
        value = sorted(set(other.value) & set(self.value), key = self.value.index)
        return FPNode(value,self.count + other.count, self)

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child
    
    def isZero(self):
        return self.value.length>0
    
    def isContained(self, other):
        return other.value.issubset(self.value)
'''
TODO:
'''
class FPTree(object):
    def __init__(self, transactions, threshold):
        return 0

    def buildTree(self):
        return 0
        
    def insertPattern(self):
        return 0
        
    def mergeTree(self):
        return 0


"""
build tree in parallelize
"""
'''
def buildTree(sContext, trans):
    step1 = sContext.parallelize(trans)
    step2 = step1.map()
    return ret

def mergeFPTree(sContext, tree_1, tree_2):
    return ret
'''

"""
Merge tree in parallelize
- map all data to spark
- build tree parallel: each node
- reduce: merge all tree
- save tree.
- merge with other tree
"""    
'''
def fptree(sContext, trans_1, trans_2):
    tree1= buildTree(trans_1)
    tree2= buildTree(trans_2)
    ret = mergeFPTree(sContext, tree1, tree2);
    return ret
'''
"""
build batch in parallelize
"""    
'''
def buildBathc(sContext, trans):
    step1 = sContext.parallelize(trans)
    step2 = step1.map(lambda x:(x,1))
    step3 = step2.flatmat
    return ret

def mergeBatch(sContext, bathc_1, batch_2):
    return ret
'''
"""
mege batch in parallelize
- map all data to spark
- build batch parallel: each node
- reduce: merge all bathc
- save batch.
- merge with other batch
"""
def buildFreqSequences(trans):
    return 0
    
'''
def batch(sContext, trans_1, trans_2):  
    batch1= buildBathc(sContext, trans_1) 
    bathc2= buildBathc(sContext, trans_2)
    ret = mergeBatch(sContext, batch1, bathc2)
    return ret
'''
def fggrowth(sc, dataName):
    # $example on$
    data = sc.textFile(dataName)
    transactions = data.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
    result = model.freqItemsets().collect()
    for fi in result:
        print(fi)
           
    
def splitLine2(line):
    kv= []
    tmp = line.strip().split(' ')
    for v in tmp:
        kv.append(int(v))
    kv.append(1)
    return kv
    
'''
the last element is the count of pattern
'''

'''
def getLine2List(line):
    ret = []
    #print("inline "+str(line))
    for t1 in line:
        if type(t1) is list:
            #tmp1.append(t1)
            islist = 0
            for tmp in t1:
                if type(tmp) is list:
                    islist += 1
                    #tmp.append(islist)
                    ret.append(tmp)#.append(3))
            if islist == 0:
                #t1.append(1)
                ret.append(t1)
    return ret
'''

'''
from list data, build batch and merge 2 batch
'''
def mergebatch(p1, p2):
    ret=[]

    batch1 = Batch(p1,0)
    batch2 = Batch(p2,0)
    batch3 = mergeBatchLocal(batch1, batch2).batch

    batch4 = []
    batch5 = []

    #print("p1: "+str(p1))
    #print("p2: "+str(p2))

    l = len(batch3)
    count =0
    for itemA in batch3:
        itemA.value.append(itemA.count)
        if count > l/2:
            batch5.append(itemA.value)
            #print(" item5 add: "+ str(itemA.value))
        else:
            batch4.append(itemA.value)
            #print(" item4 add: "+ str(itemA.value))
        count += 1

    #if len(batch4) == 0:
    #    ret.append(p1)
    #    ret.append(p2)
    #else:
    #ret.append(batch4)
    #ret.append(batch5)
    ret.append(batch3)
    
    #print("b1: "+str(batch4))
    #print("b2: "+str(batch5))
    
    return ret
'''
def reducePattern(l1, l2):
    tmp1 = getLine2List(l1)
    tmp2 = getLine2List(l2)
    #print(" l1 "+str(l1))
    #print(" l2 "+str(l2))
    #ret.append(tmp1)
    #ret.append(tmp2)
    return mergebatch(tmp1, tmp2)
'''

def reducePattern2(l1, l2):
    tmp1 = []
    tmp2 = []
    
    startTime = time.time()
    #print(" l1 "+str(l1))
    #print(" l2 "+str(l2))
    for i in l1:
        if type(i) is list:
            tmp1.append(i)
            #ret.append(i)
    for i in l2:
        if type(i) is list:
            tmp2.append(i)
            #ret.append(i)

    #print(" t1 "+str(tmp1))
    #print(" t2 "+str(tmp2))
    endTime = time.time() - startTime
    print("(1)Take time running: "+ str(endTime))
    batch1 = Batch(tmp1,0)
    batch2 = Batch(tmp2,0)
    endTime = time.time() - startTime
    print("(2)Take time running: "+ str(endTime))
    batch3 = mergeBatchLocal(batch1, batch2).batch
    endTime = time.time() - startTime
    print("(3)Take time running: "+ str(endTime))
    
    #tmp3 = mergebatch(tmp1, tmp2)
    ret = []
    for tm in batch3:
        itm = tm.value
        itm. append(tm.count)
        ret.append(itm)
    #print(" t3 "+str(ret))
    endTime = time.time() - startTime
    print("(4)Take time running: "+ str(endTime))
    return ret 
'''
map trans to pair(trans,count)
reduce: mergeTrans, list of tran and count
'''
def useReduce(trans):
    trans2 = trans.reduce(reducePattern2)
    count= 0
    for kv in trans2:#.collect():
        print(str(count)+"---2: "+str(kv[-1])+"---" +str(kv))
        count +=1
            
def seqOp(acc, value):
    #acc[0] + value
    #acc[1] + 1
    print(" seqOp a "+str(acc))
    print(" seqOp  v"+str(value)) 
    return [acc, value]

def combOp(acc1, acc2):
    #acc1[0] + acc2[0]
    #acc1[1] + acc2[1]
    print(" combOp a1 "+str(acc1))
    print(" combOp a2 "+str(acc2))
    ret = reducePattern2(acc1, acc2) 
    return ret #[acc1, acc2]

'''
#TODO:
'''
def useAggregate(trans):        
    trans2 = trans.aggregate(([], []),seqOp, combOp)
    #((1, 0),
    #(lambda acc, value: (acc[0] + value, acc[1] + 1)),
    #(lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])))
    count= 0
    for kv in trans2:#.collect():
        print(str(count)+"---2: "+str(kv[-1])+"---" +str(kv))
        count +=1
        
'''
rebuild frequence and broadcash

Case 1: from original data
    read 2 pattern
    merge pattern
Case 2: from batch
    read batch
    merge pattern
'''

def spMergeFPTreeFun(sc, dataName1, dataName2):
    data = sc.textFile(dataName)
    
    f1 = data.map(buildFreqSequences())
    
    data2 = sc.textFile(dataName2)
    
    f2 = data2.map(buildFreqSequences())
    
    f = f1.union(f2)
    
    data3 = data.union(data2)
    
    #print(data.getNumPartitions())
    
    trans = data3.map(lambda line : (splitLine2(line),1))
    
    #for kv in trans.collect():
    #    print(str(kv)+ "---")
    
    #useAggregate(trans)
    
    useReduce(trans)

'''
def linuxDataPath():
    return "/home/hduser/workspace/MLS/data/"

def winDataPath():
    return "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
'''
'''
spark-submit --py-files 'SparkADMM.py,LogisticRegressionSolver.py,ADMMDataFrames.py,AbstractSolver.py' driver.py
'''
def runSparkFPTree(sc, data1, data2):
    #def parallel():
    #conf = SparkConf().setAppName('MyFirstStandaloneApp')
    #sc = SparkContext(conf=conf)
    #path = ""
    #if cf.get_platform() == "linux":
    #    path = "/home/hduser/workspace/MLS/data/"
    #else:
    #    path = "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
#   mat = ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
    
    #fggrowth(sc,"/home/hduser/workspace/MLS/data/mushroom.dat")
    #dataName = ["mushroom.dat","mushroom_.dat"]
    startTime = time.time()
    spMergeBatchFun(sc,data1, data2)  #path + "mushroom_.dat", path + "mushroom_.dat") #str(dataName[sys.argv[0]]))#
    endTime = time.time() - startTime
    print("total time: "+str(endTime))

    #spMergeFPTreeFun(sc,path + "mushroom_.dat", path + "mushroom_.dat")
    
    #sc.stop()
