import os
import sys
import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.fpgrowth as fpg
from pyspark import SparkContext, SparkConf

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
    step2 = step1.map()
    return ret
    
def mergeBatch(sContext, bathc_1, batch_2):
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
    return ret;
    
if __name__ == "__main__":
    conf = SparkConf().setAppName('MyFirstStandaloneApp')
    sc = SparkContext(conf=conf)

#   mat = ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")

    mat = ior.read2SparseMatrix("C:\Data\Master\data_mining\data\data_694_446.csv")

    matrdd = sc.parallelize(mat,2)

    print(matrdd.collect())
    
    first = matrdd.take(2)
    print(first)
    #count = sc.parallelize(range(0, num_samples)).filter(inside).count()
    #pi = 4 * count / num_samples
    #print(pi)
    sc.stop()