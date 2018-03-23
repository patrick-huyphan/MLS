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
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from operator import add
from numpy import arange,array,ones,linalg
from pylab import plot,show
from scipy import sparse

import random
num_samples = 100000000


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
    

class Batch(object):
    def __init__(self, transactions, threshold):
        """
        Initialize the batch.
        """
        self.batch = []
        #print("batch start")
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.batch = self.build_batch(transactions, self.frequent)
        #print("batch end")

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            for item in transaction:
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
            sorted_items = [x for x in transaction if x in frequent]
            #sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            sorted(sorted_items)
            #print("sorted_items "+str(sorted_items))
            if len(sorted_items) > 0:
                self.insert_batch(sorted_items, 1)
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
            newNode = FPNode(sorted(c), pattern.count+1, None)
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
            self.batch.append(FPNode(item, 1, None))
        #print(len(batch))
        for node in mBatch:
            self.batch.append(node)
        #return batch
        #print("------------------------------------------")

    def insert_batch2(self, item, count):
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
            newNode = FPNode(sorted(c), pattern.count+1, None)
            if len(c)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sa.issubset(c)):
                    #print("c in pat " + str(idx))
                    #node =  self.batch[idx]
                    self.batch.remove(pattern)
                    print(" remove "+str(sorted(pattern.value)) + " "+ str(pattern.count))
                if(sb.issubset(c)):
                    flag1 = True
                    #print("flag1 = True")
                    #print("c in item " + str(idx))                    
                for mx in mBatch:
                    ms = set(mx.value)
                    if(ms.issubset(c)):
                        mBatch.remove(mx)
                        print(" remove mx "+str(sorted(mx.value)) + " "+ str(mx.count))
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
            self.batch.append(FPNode(item, 1, None))
        #print(len(batch))
        for node in mBatch:
            self.batch.append(node)
        #return batch
        #print("------------------------------------------")        
    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        '''
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))
        '''
    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        '''
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns
        '''
        
    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}

        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns

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
    return transactions1


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
def splitLine2(line):
    kv= line.strip().split(' ')
    kv.append(1)
    return kv
    
def addCount(line):
    ret=[]
    for x in line:
        ret.append([x,1])
    return ret
    
def addCount2(line):
    return [line,1]

# get list return list
def checkInput(inputList):
    return 0

'''
the last element is the count of pattern
'''
def getLine2List(line):
    ret = []
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
                ret.append(t1)#.append(1))
    return ret
def batch2List(batch):
    ret = []
    #for pa in batch.batch:
    return ret
'''
from list data, build batch and merge 2 batch
'''
def mergebatch(p1, p2):
    batch1 = Batch(p1,2)
    batch2 = Batch(p2,2)
    batch3 = mergeBatchLocal(batch1, batch2).batch
    #ret.append(p1)
    #ret.append(p2)
    return batch3

def reducePattern(l1, l2):
    tmp1 = getLine2List(l1)
    tmp2 = getLine2List(l2)
    
    #ret.append(tmp1)
    #ret.append(tmp2)
    return mergebatch(tmp1, tmp2)

'''
map trans to pair(trans,count)
reduce: mergeTrans, list of tran and count
'''
def sampleFun2(sc, dataName):
    data = sc.textFile(dataName,8)
    print(data.getNumPartitions())
    #for kv in data.collect():
    #    print(str(kv)+ "---")
    trans = data.map(lambda line : (splitLine2(line),1))
    
    #trans = data.flatMap(lambda line : splitLine(line)).map(lambda x: (x[0],x[1]))
    #print(trans.collect())
    for kv in trans.collect():
        print(str(kv)+ "---")
        
    trans2 = trans.reduce(reducePattern)
    count= 0
    for kv in trans2:#.collect():
        count +=1
        for kv2 in kv:
            print(str(count)+"---2: " +str(kv2))
            #for kv3 in kv2:
            #    print("---3: " +str(kv3))
    #trans3= data.mapPartitions(lambda line:  line.strip().split(' ')).collect()
    
    #trans3 = data.mapPartitions(lambda line : [line, line] , 8).collect()
    #trans3 = trans.
    
#    trans2 = trans.groupBy(lambda word: word[0])#groupByKey() #reduceByKey(lambda a, b: [a,b])
#    for kv in trans3:
#        print("\n---2: " +str(kv))
        
def linuxDataPath():
    return "/home/hduser/workspace/MLS/data/"
def winDataPath():
    return "C:\\cygwin64\\home\\patrick_huy\\workspace\\allinOne\\data\\"
    
if __name__ == "__main__":
    conf = SparkConf().setAppName('MyFirstStandaloneApp')
    sc = SparkContext(conf=conf)
    path = ""
    if cf.get_platform() == "linux":
        path = linuxDataPath()
    else:
        path = winDataPath()
#   mat = ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")
    
    #fggrowth(sc,"/home/hduser/workspace/MLS/data/mushroom.dat")
    
    sampleFun2(sc,path+"mushroom.dat")
    
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
