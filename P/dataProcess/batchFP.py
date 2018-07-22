'''
implement algorithm for frequence pattern mining by bit chain of transaction
implement algorithm to merge the batch of bitchain for incremental mining

Author: huyphan.aj@gmail.com
'''
import itertools
import dataProcess.Node as Node
import time
import copy
from datetime import datetime

class Batch(object):
    def __init__(self, transactions, threshold):
        """
        Initialize the batch.
        """
        self.batch = []
        #print("batch start")
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.batch = self.build_batch(transactions, self.frequent)
        #self.batchTmp = copy.copy(self.batch)
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
        #print("build_batch "+str(len(transactions)))
        #batchtmp = []
        for transaction in transactions:
            #print(transaction)
            sorted_items = [x for x in transaction if x in frequent]
            #print("sorted_items "+str(sorted_items))
            if len(sorted_items) > 0:
                self.insert_batch(sorted_items, 1)
                    
        #print(len(batch))
        '''
        mining_order = sorted(batchtmp, key=lambda x: x.count, reverse=True)
        for pattern in mining_order:
            print(" batch1: "+ str(pattern.value) +" "+ str(pattern.count))
        '''
        return self.batch
    '''
    check in current batch: if true -> continue
    check in update batch: if true -> continue, else add to update.
    merge update into current
    '''
    

    def insert_batch(self, item, count):
        #print("\nInsert_batch "+str(item))

        flag1 = False        
        sa = set(item)
        #print("item sa2 "+str(sa))
        
        #mBatch = []
        mbtm = {}
        
        for pattern in self.batch:
            flag2 = False
            #print("--"+str(sorted(pattern.value)) + " "+ str(pattern.count))
            sb = set(pattern.value)
            sc = sa.intersection(sb)
            totalCount = pattern.count + count
            
            if len(sc)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sb.issubset(sc)):
                    #lt = str(len(self.batch))
                    #self.batch.remove(pattern)
                    pattern.sign = True
                    #print(lt+"-"+str(len(self.batch))+" \tremove 1 \t"+ str(pattern.count) + " "+str(sorted(pattern.value)))
                
                if(sa.issubset(sc)):
                    flag1 = True
                    #print("flag1 = True, c in item " + str(idx))
                for mx in mbtm.keys():
                    ms = set(mx)
                    if(ms.issubset(sc) and (mbtm[mx]  <= totalCount)):
                        #lt= str(len(mBatch))
                        #mbtm[mx] = 0
                        mbtm.pop(mx, None)
                        #mBatch.remove(mx)
                        #print(lt+"-"+str(len(mBatch))+" \tremove 2 \t"+ str(mx.count)+ " "+str(sorted(ms, key = lambda x: int(x))))
                        break
                    if(sc.issubset(ms) and (totalCount <= mbtm[mx])):
                        flag2 = True
                        #print("2")
                        break
                '''
                for mx in mBatch[:]:
                    ms = set(mx.value)
                    if(ms.issubset(sc) and (mx.count  <= totalCount)):
                        #lt= str(len(mBatch))
                        mBatch.remove(mx)
                        #print(lt+"-"+str(len(mBatch))+" \tremove 2 \t"+ str(mx.count)+ " "+str(sorted(ms, key = lambda x: int(x))))
                        break
                    if(sc.issubset(ms) and (totalCount <= mx.count )):
                        flag2 = True
                        #print("2")
                        break
                '''
            else:
                flag2 = True
            #print(str(flag1)+" "+str(flag2))
            if(flag2 == False):
                ssc = sorted(sc, key = lambda x: int(x))
                #print("add node 2\t\t"+str(pattern.count)+" "+str(count)+" "+str(ssc))
                newNode = Node.PatternNode(list(ssc), totalCount)
                #mBatch.append(newNode)
                mbtm[tuple(ssc)] = totalCount
        for pattern in self.batch[:]:
            if(pattern.sign == True):
                self.batch.remove(pattern)

        if flag1==False:
            #print("add new node \t"+str(count) +" "+str(item))
            newNode = Node.PatternNode(item, count)
            self.batch.append(newNode)
            #print("add 1")
        #print(len(batch))
        #lb = str(len(self.batch))
        '''
        for batch in mBatch:
            self.batch.append(batch)
        '''
        for batch in mbtm:
            if(mbtm[batch] >0):
                self.batch.append(Node.PatternNode(list(batch), mbtm[batch]))
            
        #print(lb +" + "+str(len(mBatch))+" = "+str(len(self.batch)))

        #print("------------------------------------------")

    '''
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
            #d = c
            newNode = Node.PatternNode(sorted(c), pattern.count+count)
            if len(c)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sa.issubset(c)):
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
                        
        if flag1==False:
            #print("add new node "+str(item))
            self.batch.append(Node.PatternNode(item, count))
        #print(len(batch))
        for node in mBatch:
            self.batch.append(node)
        #return batch
        #print("------------------------------------------")        
    '''

    def mine_Bpatterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        '''
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))
        '''
    def zip_Bpatterns(self, patterns):
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
        
    def generate_Bpattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}

        return patterns

    def mine_subB_trees(self, threshold):
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
        #sortF1 = sorted(self.frequent, key =lambda x: self.frequent[x], reverse=True)
        #sortF2 = sorted(other.frequent, key =lambda x: other.frequent[x], reverse=True)
        #print("self.frequent: "+str(sortF1))
        #print("other.frequent: "+str(sortF2))
        
        mbatch = []
        i = 1
        for item1 in self.batch:
            sa = set(item1.value)
            j=1
            #flag0 = False
            for item2 in other.batch:
                flag0 = False
                sb = set(item2.value)
                q = sa.intersection(sb)
                counttotal = item1.count+ item2.count
                '''
                if(len(sa) == len(sb) == len(sc)):
                    print(sa)
                    print(sb)
                    print(sc)
                    print("-----")
                '''
                lb = str(len(mbatch))
                if len(q) >0:
                    if sa.issubset(q):
                         item1.sign = True
                    if sb.issubset(q):
                         item2.sign = True
                    #print(len(mbatch))
                    for item3 in mbatch[:]:
                        sd = set(item3.value)
                        if sd.issubset(q) and (item3.count <= counttotal):
                            mbatch.remove(item3)
                            break
                        if q.issubset(sd) and (counttotal <= item3.count):
                            flag0 = True
                            break
                    #print(".....................")
                else:
                    flag0 = True

                if flag0 == False:
                    #print("add "+str(counttotal)+" "+str(sc))
                    sq = sorted(q, key = lambda x: int(x))
                    mbatch.append(Node.PatternNode(sq,counttotal))
                #print(str(i)+"-"+str(j)+":\t"+lb+"-"+str(len(mbatch)))
                j +=1
            i +=1
        for item1 in self.batch[:]:
            if item1.sign == True:
                #print("remove item in batch "+ str(item1.count)+" \t"+str(item1.value))
                self.batch.remove(item1)
        
        for item1 in other.batch:
            if item1.sign != True:
                #print("add item in batch "+str(item1.count)+" \t"+str(item1.value))
                self.batch.append(item1)
                
        for item1 in mbatch:
            #print("add2 item in batch "+str(item1.count)+" \t"+str(item1.value))
            self.batch.append(item1)
            
    def HMergeBatch(self, other):        
        #mbatch = []
        mbtm = {}
        i = 1
        for item1 in self.batch:
            sa = set(item1.value)
            j=1
            for item2 in other.batch:
                sb = set(item2.value)
                q = sa.intersection(sb)
                counttotal = item1.count+ item2.count
                #lb = str(len(mbatch))
                if len(q) >0:
                    sq = sorted(q, key = lambda x: int(x))
                    #mbatch.append(Node.PatternNode(sq,counttotal))                    
                    key = tuple(sq)
                    try:
                        if(counttotal > mbtm[key]):
                            #print(str(key)+"\t"+str(mbtm[key])+"\t: new: "+str(counttotal))
                            mbtm[key] = counttotal
                    except:
                        mbtm[key] = counttotal
                        pass
                    
                    #print(str(i)+"-"+str(j)+":\t"+lb+"-"+str(len(mbatch)))
                j +=1
            i +=1
        '''
        k = 0
        for item1 in mbtm.keys():
            print(str(k)+"\txxxx\t"+str(list(item1)) +"\t"+str(mbtm[item1]))
            k +=1
        '''
        print(str(len(self.batch))+"\t"+str(len(other.batch))+"\t"+str(len(mbtm)))
        '''
        #R = copy.copy(mbatch)
        for item1 in range(len(mbatch)-1):
            #print(".")
            if mbatch[item1].sign == True:
                print("continue1\t"+str(item1))
                continue
            
            sa = sorted(mbatch[item1].value)
        
            for item2 in range(item1+1,len(mbatch)-1):
                if mbatch[item2].sign == True:
                    continue
                #print("000000 " + str(item1)+" "+str(item2))
                sb = sorted(mbatch[item2].value)
                if sa == sb:
                    if mbatch[item1].count <= mbatch[item2].count and mbatch[item1].sign == False:
                        mbatch[item1].sign = True
                        #print("111111 " + str(item1)+" "+str(item2))
                    elif mbatch[item2].count < mbatch[item1].count and mbatch[item2].sign == False:
                        mbatch[item2].sign = True
                        #print("222222 " + str(item1)+" "+str(item2))
        
        
        for item in mbatch[:]:
            if item.sign == True:
                mbatch.remove(item)
            #else:
            #    print(str(item.count)+" \t"+str(item.value))
        
        print(len(mbatch))
        '''
        mbatch = []
        for item2 in mbtm.keys():
            mbatch.append(Node.PatternNode(list(item2),mbtm[item2]))
        #print(len(mbatch))
        
        #R1 = copy.copy(self.batch)
        for item1 in self.batch[:]:
            sa = sorted(item1.value)
            for item2 in mbatch:
                sb = sorted(item2.value)
                if sa == sb and item1.count <= item2.count:
                    self.batch.remove(item1)
                    #item1.sign = True
        '''
        for item1 in self.batch[:]:
            if item1.sign == True:
                self.batch.remove(item1)
                #print(str(item1.count)+"..."+str(item1.value)+"..."+str(item2.count)+"..."+str(item2.value))
        '''
        print(len(self.batch))
        
        R2 = copy.copy(other.batch)
        for item1 in R2[:]:
            sa = sorted(item1.value)
            for item2 in mbatch:
                sb = sorted(item2.value)
                if sa == sb and item1.count <= item2.count:
                    R2.remove(item1)
                    #item1.sign = True
        '''
        for item1 in R2[:]:             
            if item1.sign == True:
                R2.remove(item1)
                #print(str(item1.count)+"..."+str(item1.value)+"..."+str(item2.count)+"..."+str(item2.value))
        '''
        print(len(R2))
        self.batch.extend(R2)
        '''
        for item2 in mbtm.keys():
            self.batch.append(Node.PatternNode(list(item2),mbtm[item2]))
        '''
        self.batch.extend(mbatch)
        
        print(len(self.batch))
        
    def VMergeBatch(self, other):
        Q=[]
        Q1=[]
        Q2=[]
        
        for item1 in self.batch:
            print(".")
        
        for item1 in other.batch:
            print(".")
        
        for item1 in self.batch:
            for item1 in other.batch:
                print(".")
        
        for item1 in Q:
            for item1 in Q:
                print(".")
        
        for item1 in Q:
            for item1 in Q1:
                print(".")
        
        for item1 in Q:
            for item1 in Q2:
                print(".")
        return 0
                
    def printBatch(self):
        for pattern in seft.batch:
            print(str(pattern.count)+ "\t"+str(pattern.value))

    def save(parttern):
        path = ""
        return path
    
    def load(path):
        pattern
        return pattern
    
def find_frequent_patterns_batch(transactions, support_threshold):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    return Batch(transactions, support_threshold)
    #return batch #.mine_patterns(support_threshold)

#def printBatch(batch)
    
def runBatchMerge(transactions, threshold):
    batch1 = Batch(transactions[0], threshold)
    
    mining_order = sorted(batch1.batch, key=lambda x: x.count, reverse=True)
    '''
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch1: "+ str(pattern.value))
        count +=1
    '''
    batch2 = Batch(transactions[1], threshold)
    
    mining_order = sorted(batch2.batch, key=lambda x: x.count, reverse=True)
    '''
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch2: "+ str(pattern.value))
        count +=1
    '''
    startTime = time.time()
    
    batch1.HMergeBatch(batch2)
    
    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    mining_order = sorted(batch1.batch, key=lambda x: x.count, reverse=True)
    count = 0
    for pattern in mining_order:
        print(str(count)+"\t"+ str(pattern.count)+"\tbatch3: "+ str(pattern.value) )
        count +=1

    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    return 0

def test(transactions, threshold):
    startTime = time.time()
    batch1 = Batch(transactions[0], threshold)
    endTime = time.time() - startTime
    print("Build Bath 0 take total time: "+str(endTime))
    
    startTime = time.time()
    batch2 = Batch(transactions[1], threshold)
    endTime = time.time() - startTime
    print("Build Bath 1 take total time: "+str(endTime))
    
    startTime = time.time()
    batch3 = Batch(transactions[2], threshold)
    endTime = time.time() - startTime
    print("Build Bath 2 take total time: "+str(endTime))
    
    startTime = time.time()
    batch4 = Batch(transactions[3], threshold)
    endTime = time.time() - startTime
    print("Build Bath 3 take total time: "+str(endTime))
    
    startTime = time.time()
    batch5 = Batch(transactions[4], threshold)
    endTime = time.time() - startTime
    print("Build Bath 4 take total time: "+str(endTime))
    
    startTime = time.time()
    batch6 = Batch(transactions[5], threshold)
    endTime = time.time() - startTime
    print("Build Bath 5 take total time: "+str(endTime))
    
    startTime = time.time()
    batch7 = Batch(transactions[6], threshold)    
    endTime = time.time() - startTime
    print("Build Bath 6 take total time: "+str(endTime))

    startTime = time.time()
    batch8 = Batch(transactions[7], threshold)
    endTime = time.time() - startTime
    print("Build Bath 7 take total time: "+str(endTime))
    
    startTime = time.time()
    batch9 = Batch(transactions[8], threshold)
    endTime = time.time() - startTime
    print("Build Bath 8 take total time: "+str(endTime))
    
    startTime = time.time()
    batch10 = Batch(transactions[9], threshold)
    endTime = time.time() - startTime
    print("Build Bath 9 take total time: "+str(endTime))
    
    startTime = time.time()
    batch11 = Batch(transactions[10], threshold)
    endTime = time.time() - startTime
    print("Build Bath 10 take total time: "+str(endTime))
    
    startTime = time.time()
    batch12 = Batch(transactions[11], threshold)
    endTime = time.time() - startTime
    print("Build Bath 11 take total time: "+str(endTime))
    
    startTime = time.time()
    batch13 = Batch(transactions[12], threshold)
    endTime = time.time() - startTime
    print("Build Bath 12 take total time: "+str(endTime))
    
    startTime = time.time()
    batch14 = Batch(transactions[13], threshold)
    endTime = time.time() - startTime
    print("Build Bath 13 take total time: "+str(endTime))
    
    startTime = time.time()
    batch15 = Batch(transactions[14], threshold)
    endTime = time.time() - startTime
    print("Build Bath 14 take total time: "+str(endTime))
    
    startTime = time.time()
    batch16 = Batch(transactions[15], threshold)
    endTime = time.time() - startTime
    print("Build Bath 15 take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch2)
    endTime = time.time() - startTime
    print("BathcMerge(1-2) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch4.HMergeBatch(batch3)
    endTime = time.time() - startTime
    print("BathcMerge(3-4) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch6.HMergeBatch(batch5)
    endTime = time.time() - startTime
    print("BathcMerge(5-6) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch8.HMergeBatch(batch7)
    endTime = time.time() - startTime
    print("BathcMerge(7-8) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch10.HMergeBatch(batch9)
    endTime = time.time() - startTime
    print("BathcMerge(9-10) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch11.HMergeBatch(batch12)
    endTime = time.time() - startTime
    print("BathcMerge(11-12) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch13.HMergeBatch(batch14)
    endTime = time.time() - startTime
    print("BathcMerge(13-14) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch15.HMergeBatch(batch16)
    endTime = time.time() - startTime
    print("BathcMerge(15-16) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch4)
    endTime = time.time() - startTime
    print("BathcMerge(12-34) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch6.HMergeBatch(batch8)
    endTime = time.time() - startTime
    print("BathcMerge(56-78) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch10.HMergeBatch(batch11)
    endTime = time.time() - startTime
    print("BathcMerge(910-1112) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch13.HMergeBatch(batch15)
    endTime = time.time() - startTime
    print("BathcMerge(1314-1516) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch6)
    endTime = time.time() - startTime
    print("BathcMerge(1-8) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch10.HMergeBatch(batch13)
    endTime = time.time() - startTime
    print("BathcMerge(9-16) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch10)
    endTime = time.time() - startTime
    print("BathcMerge(1-16) take total time: "+str(endTime))
    
    '''
    for pattern in batch1.batch:
        print(" batch3: "+ str(pattern.value) +" "+ str(pattern.count))
    '''

    #endTime = time.time() - startTime
    #print("BathcMerge take total time: "+str(endTime))
    
    return 0
    
def test_2(transactions, threshold):
    startTime = time.time()
    batch1 = Batch(transactions[0], threshold)
    endTime = time.time() - startTime
    print("Build Bath 0 take total time: "+str(endTime))
    
    startTime = time.time()
    batch2 = Batch(transactions[1], threshold)
    endTime = time.time() - startTime
    print("Build Bath 1 take total time: "+str(endTime))
    
    startTime = time.time()
    batch3 = Batch(transactions[2], threshold)
    endTime = time.time() - startTime
    print("Build Bath 2 take total time: "+str(endTime))
    
    startTime = time.time()
    batch4 = Batch(transactions[3], threshold)
    endTime = time.time() - startTime
    print("Build Bath 3 take total time: "+str(endTime))
    
    startTime = time.time()
    batch5 = Batch(transactions[4], threshold)
    endTime = time.time() - startTime
    print("Build Bath 4 take total time: "+str(endTime))
    
    startTime = time.time()
    batch6 = Batch(transactions[5], threshold)
    endTime = time.time() - startTime
    print("Build Bath 5 take total time: "+str(endTime))
    
    startTime = time.time()
    batch7 = Batch(transactions[6], threshold)    
    endTime = time.time() - startTime
    print("Build Bath 6 take total time: "+str(endTime))

    startTime = time.time()
    batch8 = Batch(transactions[7], threshold)
    endTime = time.time() - startTime
    print("Build Bath 7 take total time: "+str(endTime))
    
    startTime = time.time()
    batch9 = Batch(transactions[8], threshold)
    endTime = time.time() - startTime
    print("Build Bath 8 take total time: "+str(endTime))
    
    startTime = time.time()
    batch10 = Batch(transactions[9], threshold)
    endTime = time.time() - startTime
    print("Build Bath 9 take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch2)
    endTime = time.time() - startTime
    print("BathcMerge(1-2) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch3)
    endTime = time.time() - startTime
    print("BathcMerge(1-3) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch4)
    endTime = time.time() - startTime
    print("BathcMerge(1-4) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch5)
    endTime = time.time() - startTime
    print("BathcMerge(1-5) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch6)
    endTime = time.time() - startTime
    print("BathcMerge(1-6) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch7)
    endTime = time.time() - startTime
    print("BathcMerge(1-7) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch8)
    endTime = time.time() - startTime
    print("BathcMerge(1-8) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch9)
    endTime = time.time() - startTime
    print("BathcMerge(1-9) take total time: "+str(endTime))
    
    startTime = time.time()
    #batch3 = 
    batch1.HMergeBatch(batch10)
    endTime = time.time() - startTime
    print("BathcMerge(1-10) take total time: "+str(endTime))
        
    for pattern in batch1.batch:
        print(" batch3: "+ str(pattern.value) +" "+ str(pattern.count))


    #endTime = time.time() - startTime
    #print("BathcMerge take total time: "+str(endTime))
    
    return 0
    
def test_3(transactions, threshold):
    i = 0
    #print("=====================\t"+ str(len(transactions)))
    for transaction in transactions:
        startTime = time.time()
        Batch(transaction, threshold)
        endTime = time.time() - startTime
        print(str(i)+"\t"+str(endTime))
        i +=1
    return 0
