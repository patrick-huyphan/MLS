'''
implement algorithm for frequence pattern mining by bit chain of transaction
implement algorithm to merge the batch of bitchain for incremental mining

Author: huyphan.aj@gmail.com
'''
import itertools
import dataProcess.Node as Node
import time
import copy

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
        #batchtmp = []
        for transaction in transactions:
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
        mBatch = []
        tmpBtc = copy.copy(self.batch)
        
        for pattern in self.batch[:]:
            flag2 = False
            #print("--"+str(sorted(pattern.value)) + " "+ str(pattern.count))
            sb = set(pattern.value)
            sc = sa.intersection(sb)
            totalCount = pattern.count + count
            
            if len(sc)>0:
                #print("intersection "+str(c)+" "+ str(newNode.count))
                if(sb.issubset(sc) and pattern.count <= totalCount):
                    #lt = str(len(self.batch))
                    self.batch.remove(pattern)
                    #tmpBtc.remove(pattern)
                    #print(lt+"-"+str(len(self.batch))+" \tremove 1 \t"+ str(pattern.count) + " "+str(sorted(pattern.value)))
                
                if(sa.issubset(sc)):
                    flag1 = True
                    #print("flag1 = True, c in item " + str(idx))
                #mtmpBtc = copy.copy(mBatch)
                for mx in mBatch[:]:
                    ms = set(mx.value)
                    if(ms.issubset(sc) and (mx.count  <= totalCount)):
                        #lt= str(len(mBatch))
                        mBatch.remove(mx)
                        #mtmpBtc.remove(mx)
                        #print(lt+"-"+str(len(mBatch))+" \tremove 2 \t"+ str(mx.count)+ " "+str(sorted(ms, key = lambda x: int(x))))
                        break
                    if(sc.issubset(ms) and (totalCount <= mx.count )):
                        flag2 = True
                        #print("2")
                        break
                #mBatch = mtmpBtc
            else:
                flag2 = True
            #print(str(flag1)+" "+str(flag2))
            if(flag2 == False):
                ssc = sorted(sc, key = lambda x: int(x))
                #print("add node 2\t\t"+str(pattern.count)+" "+str(count)+" "+str(ssc))
                newNode = Node.PatternNode(list(ssc), totalCount)
                mBatch.append(newNode)
        #self.batch = tmpBtc
        if flag1==False:
            #print("add new node \t"+str(count) +" "+str(item))
            newNode = Node.PatternNode(item, count)
            self.batch.append(newNode)
            #print("add 1")
        #print(len(batch))
        #lb = str(len(self.batch))
        for batch in mBatch:
            self.batch.append(batch)
        #print(lb +" "+str(len(mBatch))+" "+str(len(self.batch)))
        #return mBatchTmp
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
        print("self.frequent: "+str(self.frequent))
        print("other.frequent: "+str(other.frequent))
        '''
        for item in other.batch:
            #print(str(other.batch.index(item)) + " "+ str(len(other.batch)))
            self.insert_batch(item.value, item.count)

        print("merge.frequent: "+str(self.frequent))
        '''
        
        mbatch = []
        for item1 in self.batch:
            sa = set(item1.value)
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
                    
                if len(q) >0:
                    if sa.issubset(q):
                         #print(str(item1.count)+" sa "+str(sa))
                         item1.sign = -1
                    if sb.issubset(q):
                         #print(str(item2.count)+" sb "+str(sb))
                         item2.sign = -1

                    for item3 in mbatch:
                        sd = set(item3.value)
                        
                        if sd.issubset(q) and (item3.count <= counttotal):
                            mbatch.remove(item3)
                            break
                        if q.issubset(sd) and (counttotal <= item3.count):
                            flag0 = True
                            break
                else:
                    flag0 = True
                    
                if flag0 == False:
                    #print("add "+str(counttotal)+" "+str(sc))
                    sq = sorted(q, key = lambda x: int(x))
                    mbatch.append(Node.PatternNode(sq,counttotal))

        for item1 in self.batch[:]:
            if item1.sign == -1:
                #print("remove item in batch "+ str(item1.count)+" \t"+str(item1.value))
                self.batch.remove(item1)
        
        for item1 in other.batch:
            if item1.sign != -1:
                #print("add item in batch "+str(item1.count)+" \t"+str(item1.value))
                self.batch.append(item1)
                
        for item1 in mbatch:
            #print("add2 item in batch "+str(item1.count)+" \t"+str(item1.value))
            self.batch.append(item1)
        
    def printBatch(self):
        for pattern in seft.batch:
            print(str(pattern.count)+ " "+str(pattern.value))


def find_frequent_patterns_batch(transactions, support_threshold):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    return Batch(transactions, support_threshold)
    #return batch #.mine_patterns(support_threshold)

#def printBatch(batch)
    
def runBatchMerge(transactions1,transactions2, threshold):
    startTime = time.time()
    
    batch1 = Batch(transactions1, threshold)
    
    mining_order = sorted(batch1.batch, key=lambda x: x.count, reverse=True)
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch1: "+ str(pattern.value))
        count +=1
    
    batch2 = Batch(transactions2, threshold)
    
    mining_order = sorted(batch2.batch, key=lambda x: x.count, reverse=True)
    count = 0
    for pattern in mining_order:
        print(str(count) +"\t"+ str(pattern.count)+"\tbatch2: "+ str(pattern.value))
        count +=1
    
    batch1.mergeBatch(batch2)
    
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

def test(transactions1,transactions2, threshold):
    startTime = time.time()
    
    batch1 = Batch(transactions1, threshold)
    
    batch2 = Batch(transactions2, threshold)
    
    batch3 = Batch(transactions1, threshold)
    
    batch4 = Batch(transactions2, threshold)
    
    batch5 = Batch(transactions1, threshold)
    
    batch6 = Batch(transactions2, threshold)
    
    #batch3 = 
    batch1.mergeBatch(batch2)
    
    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    
    for pattern in batch1.batch:
        print(" batch3: "+ str(pattern.value) +" "+ str(pattern.count))


    endTime = time.time() - startTime
    print("BathcMerge take total time: "+str(endTime))
    return 0
