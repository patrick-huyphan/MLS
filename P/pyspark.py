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

def buildBatch(data):
	print("buildBatch")

def mergeBatch(P1, P2):
	print("mergeBatch")
	
def buildFPTree(data):
	print("buildBatch")

def mergeFPtree(P1, P2):
	print("mergeFPtree")	
	
if __name__ == "__main__":
	conf = SparkConf().setAppName('MyFirstStandaloneApp')
	sc = SparkContext(conf=conf)

#	mat = ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")

	mat = ior.read2SparseMatrix("C:\Data\Master\data_mining\data\data_694_446.csv")

	matrdd = sc.parallelize(mat,2)

	print(matrdd.collect())
	
	first = matrdd.take(2)
	print(first)
	#count = sc.parallelize(range(0, num_samples)).filter(inside).count()
	#pi = 4 * count / num_samples
	#print(pi)
	sc.stop()