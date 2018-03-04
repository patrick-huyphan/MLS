import os
import sys
sys.path.insert(0, '/home/hduser/workspace/MLS/P/')

import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.fpgrowth as fpg
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from operator import add

import random
num_samples = 100000000
def inside(p):     
    x, y = random.random(), random.random()
    return x*x + y*y < 1

"""
build tree in parallelize
"""
def buildTree(sContext, trans):
    step1 = sContext.parallelize(trans)
    step2 = step1.map()
    return ret

def mergeFPTree(sContext, tree_1, tree_2):
    return ret
    
"""
Merge tree in parallelize
- map all data to spark
- build tree parallel: each node
- reduce: merge all tree
- save tree.
- merge with other tree
"""    
def fptree(sContext, trans_1, trans_2):
    tree1= buildTree(trans_1)
    tree2= buildTree(trans_2)
    ret = mergeFPTree(sContext, tree1, tree2);
    return ret

"""
build batch in parallelize
"""    
def buildBathc(sContext, trans):
    step1 = sContext.parallelize(trans)
    step2 = step1.map(lambda x:(x,1))
    step3 = step2.flatmat
    return ret

def mergeBatch(sContext, bathc_1, batch_2):
    return ret
"""
mege batch in parallelize
- map all data to spark
- build batch parallel: each node
- reduce: merge all bathc
- save batch.
- merge with other batch
"""
def batch(sContext, trans_1, trans_2):  
    batch1= buildBathc(sContext, trans_1) 
    bathc2= buildBathc(sContext, trans_2)
    ret = mergeBatch(sContext, batch1, bathc2)
    return ret

def fggrowth(sc, dataName):
    # $example on$
    data = sc.textFile(dataName)
    transactions = data.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
    result = model.freqItemsets().collect()
    for fi in result:
        print(fi)
        
def f(x): print(x)

def buildKey(x):
    key=''
    for k in x:
        key +=str(k)+"/"
    return key
    
def splitLine(line):
    kv=[]
    for item in line.strip().split(' '): 
        kv.append([item, line.strip().split(' ')])
    return kv
    
def mergeLine(l1, l2):
    ret = []
    
    for x in l1:
        ret.append([x[0],x[1]])
        
    for y in l2:
        idx=0
        for k in ret:
            if k[0]==y[0]:
                ret[idx] = [k[0],x[1]+y[1]]
            else:
                ret.append([y[0],y[1]])
            
            idx=idx+1
#    print(ret)
    return ret

def addCount(line):
    ret=[]
    for x in line:
        ret.append([x,1])
    return ret
    
def addCount2(line):
    return [line,1]

def mergeLine2(l1, l2):
    ret = []
    count = False
    flag1 = False
    flag2 = False
    idx1=0
    idx2=0
    #print(l1)
    for t1 in l1:
        #print(t1)
        if type(t1) is list:
            ret.append(t1)
        idx1 = idx1+1

    for t1 in l2:
        #print(t1)
        #ret.append(t1)
        if type(t1) is list:
            ret.append([t1,idx2])
        idx2 = idx2+1 

        #for t2 in l2:
            #print(t1)
            #print(t2)
            #sb = set(t2[0])
            #sa = set(t1[0])
            #c = sa.intersection(sb)
            #if len(c)>0:
            #    ret.append([c,t1[1]+t2[1]])
            #else:
            #ret.append(t1)
            #    ret.append(t2)
    return ret

'''
map trans to pair(trans,count)
reduce: mergeTrans, list of tran and count
'''
def sampleFun2(sc, dataName):
    data = sc.textFile(dataName)
    
    trans = data.map(lambda line : line.strip().split(' ')).map(addCount2)#splitLine(line))
    
    #trans = data.flatMap(lambda line : splitLine(line)).map(lambda x: (x[0],x[1]))
    #print(trans.collect())
    for kv in trans.collect():
        print(str(kv)+ "---")
        
    trans2 = trans.reduce(mergeLine2)

#    trans2 = trans.groupBy(lambda word: word[0])#groupByKey() #reduceByKey(lambda a, b: [a,b])
    for kv in trans2:
        print("---2: " +str(kv))
    
if __name__ == "__main__":
    conf = SparkConf().setAppName('MyFirstStandaloneApp')
    sc = SparkContext(conf=conf)

#   mat = ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
    
    #fggrowth(sc,"/home/hduser/workspace/MLS/data/mushroom.dat")
    
    sampleFun2(sc,"/home/hduser/workspace/MLS/data/mushroom.dat")
    
    #transactions = ior.read2RawData("/home/hduser/workspace/MLS/data/mushroom.dat",50, 150, 30)
    
    #mat = ior.read2SparseMatrix("C:\Data\Master\data_mining\data\data_694_446.csv")
    
    #matrdd = sc.parallelize(transactions,2)

    #print(matrdd.collect())
    
    #first = matrdd.take(2)
    #print(first)
    #count = sc.parallelize(range(0, num_samples)).filter(inside).count()
    #pi = 4 * count / num_samples
    #print(pi)
    sc.stop()