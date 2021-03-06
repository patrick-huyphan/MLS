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

class PatternNode(object):
    def __init__(self, value, count):
        """
        Create the node.
        """
        self.value = value
        self.count = count

    def intercept(self, other):
        value = sorted(set(other.value) & set(self.value), key = self.value.index)
        return PatternNode(value,self.count + other.count)
        
    def isZero(self):
        return self.value.length>0
    
    def isContained(self, other):
        return other.value.issubset(self.value)

class Batch(object):
    def __init__(self, transactions, threshold):
        """
        Initialize the batch.
        """
        self.batch = []
        #print("batch start")
        #self.frequent = self.find_frequent_items(transactions, threshold)
        #self.batch = self.build_batchPa(transactions, self.frequent)
        for transaction in transactions:
            #_tmp = transaction[:-1]
            #count = transaction[-1]
            #sorted(sorted_items)
            #print(str(count) +" sorted_items "+str(_tmp))
            #newNode = FPNode(_tmp, count, None)
            self.batch.append(PatternNode(transaction[:-1], transaction[-1], None))
        #print("batch end")

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            _tmp = transaction[0:-1]
            for item in _tmp:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1

        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]
        #print(str(items))
        return items
        
    def build_batch(self, transactions, frequent):
        """
        Build patern.
        """
        for transaction in transactions:
            _tmp = transaction[0:-1]
            sorted_items = [x for x in _tmp if x in frequent]
            #sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            sorted(sorted_items)
            #print("sorted_items "+str(sorted_items))
            
            if len(sorted_items) > 0:
                self.insert_batch(sorted_items, 1)
            
            #newNode = FPNode(sorted_items, 1, None)
            #self.batch.append(newNode)
            
        #print(len(batch))        
        return self.batch
    '''
    check in current batch: if true -> continue
    check in update batch: if true -> continue, else add to update.
    merge update into current
    '''
    def insert_batch(self, item, count):
        #print("insert_batch")
        count = False
        flag1 = False
        flag2 = False
        idx=0
        #print("item "+str(item))
        mBatch = []
        for pattern in self.batch:
            #print("--"+str(sorted(pattern.value)) + " "+ str(pattern.count))
            sa = set(pattern.value)
            sb = set(item)
            c = sa.intersection(sb)
            #d = c
            newNode = PatternNode(sorted(c), pattern.count+1)
            if len(c)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sa.issubset(c)):
                    #print("c in pat " + str(idx))
                    #node =  self.batch[idx]
                    self.batch.remove(pattern)
                    #print(" remove "+str(sorted(pattern.value)) + " "+ str(pattern.count))
                if(sb.issubset(c)):
                    flag1 = True
                    #print("flag1 = True")
                    #print("c in item " + str(idx))                    
                for mx in mBatch:
                    ms = set(mx.value)
                    if(ms.issubset(c)):
                        mBatch.remove(mx)
                        #print(" remove mx "+str(sorted(mx.value)) + " "+ str(mx.count))
                        #print("mBatch.remove(mx)") 
                    if(c.issubset(ms)):
                        flag2 = True
                        #print(":") 
            else:
                flag2 = True
                
            if(flag2 == False):
                #print("add node "+str(newNode.value) +" "+str(newNode.count))
                mBatch.append(newNode)
            #else:
            #    print("not add node "+str(newNode.value) +" "+str(newNode.count))
            flag2 = False
            
            #if(pattern.value == item):
                #print(str(idx))
            #    count = True
            #    batch[idx].count = batch[idx].count+1
            #idx = idx +1
                        
        if count==False:
            #print("add new node "+str(item))
            self.batch.append(PatternNode(item, 1))
        #print(len(batch))
        for node in mBatch:
            self.batch.append(node)
        #return batch
        #print("------------------------------------------")

    def insert_batch2(self, item, count):
        #print("insert_batch")
        #count = False
        flag1 = False
        flag2 = False
        idx=0
        #print("item "+str(item))
        mBatch = []
        for pattern in self.batch:
            #print("--"+str(sorted(pattern.value)) + " "+ str(pattern.count))
            sa = set(pattern.value)
            sb = set(item)
            c = sa.intersection(sb)

            newNode = PatternNode(sorted(c), pattern.count+count)
            if len(c)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sa.issubset(c)):
                    #print("c in pat " + str(idx))
                    #node =  self.batch[idx]
                    self.batch.remove(pattern)
                    #print(" remove "+str(sorted(pattern.value)) + " "+ str(pattern.count))
                if(sb.issubset(c)):
                    flag1 = True
                    #print("flag1 = True, c in item " + str(idx))                    
                for mx in mBatch:
                    ms = set(mx.value)
                    if(ms.issubset(c)):
                        mBatch.remove(mx)
                        #print(" remove mx "+str(sorted(mx.value)) + " "+ str(mx.count))
                    if(c.issubset(ms)):
                        flag2 = True
                        #print(":") 
            else:
                flag2 = True
                
            if(flag2 == False):
                #print("add node "+str(newNode.value) +" "+str(newNode.count))
                mBatch.append(newNode)
            #else:
            #    print("not add node "+str(newNode.value) +" "+str(newNode.count))
            flag2 = False            
                        
        if flag1==False:
            #print("add new node "+str(item))
            self.batch.append(PatternNode(item, count))
        #print(len(batch))
        for node in mBatch:
            self.batch.append(node)
        #return batch
        #print("------------------------------------------")        

    """
    merge batch A and B:
    - for each itemA in A:
        for each itemB in B:
            q= itemA intersec itemB
            if(q != 0):
            if itema is child(Q): A\itemA
            if....
            
            A = A U C
            
    """
    def mergeBatch(self, other):
        for itemA in other.batch:
            self.insert(itemA, self.batch)
            #for itemB in self.batch:
        #return newbatch        


def mergeBatchLocal(transactions1, transactions2):
    for itemA in transactions2.batch:
        #print(" itemA: "+ str(itemA.value) +" "+ str(itemA.count))
        transactions1.insert_batch2(itemA.value, itemA.count)
#    for itemA in transactions1.batch:
#        print(" itemA: "+ str(itemA.value) +" "+ str(itemA.count))
    return transactions1


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

def reducePattern(l1, l2):
    tmp1 = getLine2List(l1)
    tmp2 = getLine2List(l2)
    #print(" l1 "+str(l1))
    #print(" l2 "+str(l2))
    #ret.append(tmp1)
    #ret.append(tmp2)
    return mergebatch(tmp1, tmp2)
'''

def reducePattern(l1, l2):
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
    trans2 = trans.reduce(reducePattern)
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
def spMergeBatchFun(sc, dataName1, dataName2):
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
def runSparkBatch(sc,data1, data2):
    startTime = time.time()
    spMergeBatchFun(sc,data1, data2)
    endTime = time.time() - startTime
    print("total time: "+str(endTime))
