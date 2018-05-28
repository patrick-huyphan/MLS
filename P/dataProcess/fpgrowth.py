'''
implement FPtree and FPGrowth algorithm
implement merge FPtree anlgorimth for incremental mining

Author: huyphan.aj@gmail.com
'''
import itertools
import dataProcess.Node as Node
import time
import copy
from itertools import chain
from collections import defaultdict, Counter

class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, threshold, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(
            transactions, root_value,
            root_count, self.frequent, self.headers)

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
        #print("find_frequent_items "+ str(items))
        return items

    @staticmethod
    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None
        #print("headers "+ str(headers))
        return headers

    def build_fptree(self, transactions, root_value,
                     root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        #print("build_fptree: \t root: "+str(root_value) +" \t count: "+ str(root_count))
        #print("frequent "+str(frequent))
        
        root = Node.FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            #print(sorted_items)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)
        
        #print("headers " +str(headers.keys()))
        return root
    '''
    def build_fptree(self, path, root_value,
                     root_count, frequent, headers): #FromPath
        """
        Build the FP tree and return the root node.
        """
        print("build_fptree from path : \t root: "+str(root_value) +" \t count: "+ str(root_count))
        #print("frequent "+str(frequent))
        
        root = Node.FPNode(root_value, root_count, None)

        for transaction in path:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            #print(sorted_items)
            if len(sorted_items) > 0:
                self.insert_path_tree(sorted_items, root, headers)
        
        #print("headers " +str(headers.keys()))
        return root
    '''
    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        """
        #print("insert_tree "+ str(items))
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

        
    def insert_path_tree(self, path, root, headers):
        """
        Recursively grow FP tree.
        """
        #print(headers)
        '''
        paths = []
        for p in path:
            paths.append(p.value)
        print(str(root.value)+" insert_tree "+ str(paths))
        '''
        first = path[0].value
        child = root.get_child(first)
        if child is not None:
            child.count += path[0].count
        else:
            # Add new child.
            child = root.add_child_node(path[0])
            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    #print("...")
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = path[1:]
        if len(remaining_items) > 0:
            self.insert_path_tree(remaining_items, child, headers)
            
    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        #print("zip_patterns")
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        #print("generate_pattern_list")
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] =  min([self.frequent[x] for x in subset])

        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x], reverse=True)
        
        #print("mine_sub_trees "+str(mining_order))

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
            #print(str(item)+", suffixes: "+str(suffixes))
            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent
                
                #if len(path)>0:
                #    print(str(suffix.parent.value) +" ("+ str(frequency) +" "+ str(len(suffix.children)) +") "+ str(suffix.value) +" \t path: "+ str(path))
                
                if len(path)>0:
                    for i in range(frequency):
                        conditional_tree_input.append(path)
                    

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            if len(conditional_tree_input)>0:
                subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
                subtree_patterns = subtree.mine_patterns(threshold)

                # Insert subtree patterns into main patterns dictionary.
                for pattern in subtree_patterns.keys():
                    if pattern in patterns:
                        patterns[pattern] += subtree_patterns[pattern]
                    else:
                        patterns[pattern] = subtree_patterns[pattern]
        
        #print(len(patterns))
        return patterns
    '''
    flow of paper:
    Note: should reset: parent, child, and link
    '''
    def sortWithOrder(self, path,gOrder):
        pathTmp = path[::-1]
        pathdata = []

        '''
        for item in pathTmp:
            #a = pathTmp.index(item)
            b = gOrder.index(item.value)
            pathdata.append([item.value, b])
        print("Path 0"+ str(pathdata))
        '''        
        aMax = -1
        for item in pathTmp:
            aMin = 10000
            for item2 in pathTmp:
                d = gOrder.index(item2.value)
                if  d >aMax and d < aMin:
                    aMin = d
            #print(str(d)+" "+str(aMin)+" "+str(aMax))
            for item2 in pathTmp:
                if item2.value == gOrder[aMin]:
                    tmpIt = Node.FPNode(item2.value, item2.count, None)
                    #tmpIt.link= None
                    #tmpIt.children = []
                    pathdata.append(tmpIt)
                    aMax = aMin
                    break
        ''' 
        pathdata2 = []
        for item in pathdata:
            b = gOrder.index(item.value)
            pathdata2.append([item.value, item.count,b])
        print("Path 1"+ str(pathdata2))
        '''
        return pathdata
        
    # read itemSet to vector and arange with gOrder
    def readItemSets(self, gOrder):
        print("\n ReadItemSets")
        vItemSet = []
        #self.printTree()
        mining_order = sorted(self.frequent.keys(), key=lambda x: self.frequent[x], reverse=False)
        #print("gOrder frequent \t"+str(gOrder))
        #print("mergeTree frequent \t"+str(mining_order))
        
        for item in mining_order:
            node = self.headers[item]
            suffixesV0 = []
            #header = []
            while node is not None:
                #header.append([node.value, node.count])
                suffixesV0.append(node)
                node = node.link
            
            #print(str(item)+" Header: "+ str(header))
            #count = 0
            for suffix in suffixesV0:                
                # check full path, children = 0
                if (suffix.count > 0) and (len(suffix.children)==0):
                    pathNode = [suffix]
                    #path = [[suffix.value, suffix.count]]
                    parent = suffix.parent
                    #print(str(suffix.value)+":"+str(suffix.count) +"-"+str(pathNode[-1].count))
                    #parent.count = parent.count - pathNode[-1].count
                    
                    # trace back the path
                    while parent.parent is not None:
                        #parent.count = parent.count - suffix.count
                        cTmp = parent.count
                        parent.count = suffix.count
                        pathNode.append(copy.copy(parent))
                        #path.append([parent.value,parent.count, tmp])
                        
                        # update count of checked node
                        parent.count = cTmp - suffix.count
                        parent = parent.parent
                    
                    #print("----------"+ str(count)+":\t path: "+ str(path))
                    #count +=1
                    vItemSet.append(pathNode)
                    
                    '''
                    pathdata = []
                    for item in pathNode:
                        pathdata.append([item.value,item.count])
                    pathdatatmp = self.sortWithOrder(pathdata, gOrder)
                    print("Path "+ str(pathdatatmp))
                    '''
        '''
        for path in vItemSet:
            pathdata = []
            for item in path:
        '''
        # re-sort path with global order
        rvItemSet = []
        for path in vItemSet:
            pathdatatmp = self.sortWithOrder(path, gOrder)
            rvItemSet.append(pathdatatmp)
        
        # add new header
        for item in gOrder:
            if item not in self.headers.keys():
                print("add key "+str(item))
                self.headers[item] = None
                
        return rvItemSet
    
    #rebuild tree with new vector of itemSet
    def v2Tree(self,itemSet, frequence):
        #print("\n V2Tree")
        for key in self.headers.keys():
            self.headers[key] = None

        root = Node.FPNode(None, None, None)
        for path in itemSet:
            #print("root of new path:\t"+ str(path[-1].value))
            self.insert_path_tree(path,root, self.headers)
        #update frequent
        self.frequent = frequence
        self.root = root

        return self
        
    #merge vector itemset to tree
    def mergeV2T(self, itemSets):
        #print("\n MergeV2T")
        for path in itemSets:
            #print("root of path:\t"+ str(path[-1].value))
            self.insert_path_tree(path,self.root, self.headers)
        return self

    def BIT_FPGrowth(self, other):
        # merge frequent
        print("\n BIT_FPGrowth")
        #print("self.frequent: "+str(self.frequent))
        #print("other.frequent: "+str(other.frequent))
        
        #Merge frequent
        newFrequent = dict(Counter(self.frequent) + Counter(other.frequent))
        #print("merge.frequent: "+str(newFrequent))
        
        #Sort and update frequent
        mining_order = sorted(newFrequent.keys(), key=lambda x: newFrequent[x], reverse=True)
        #print("mergeTree frequent "+str(mining_order))
        
        # read itemset and sort with new order
        vItemSet1 = self.readItemSets(mining_order)
        
        # rebuild FPtree with new order
        newTree = self.v2Tree(vItemSet1, newFrequent)
        #newTree.printPattern()
        
        # read other tree to itemset
        vItemSet2 = other.readItemSets(mining_order)
        
        # Add other itemset to new FPtree
        newTree.mergeV2T(vItemSet2)
        
        #newTree.printPattern()
        # return new FP tree
        return newTree

    """
    self.frequent: all node with frequency
    self.header: list of root
    self.root: list of node in header, include link to child
    - update frequency list
    - merge header: self.header + other.header
    - merge other tree into FP tree: self + other, from root of self compare with root of other, update count of node, loop in subtree
    
    rebuild frequency header and tree
    for rebuild tree:
        case1: same root
    
        case2: dif root, should find posible position to merge
    """
    '''
    def mergeTree(self, other):
        print("ROOT1: "+str(self.root.value)+"\t child of root: "+str(len(self.root.children)))
        listParrent1 = self.root.children
        print("ROOT2: "+str(other.root.value)+"\t child of root: "+str(len(other.root.children)))
        listParrent2 = other.root.children
        listtmp={}
        i=0
        # merge frequent
        print("self.frequent: "+str(self.frequent))
        print("other.frequent: "+str(other.frequent))
        
        d = dict(Counter(self.frequent) + Counter(other.frequent))
        print("merge.frequent: "+str(d))
                
        mining_order1 = sorted(self.frequent.keys(), key=lambda x: self.frequent[x], reverse=True)
        #print("mergeTree frequent "+str(self.frequent))
        print("mergeTree frequent "+str(mining_order1))
        mining_order2 = sorted(other.frequent.keys(), key=lambda x: other.frequent[x], reverse=True)
        #print("mergeTree frequent "+str(other.frequent))
        print("mergeTree frequent "+str(mining_order2))
        
        mining_order = sorted(d.keys(), key=lambda x: d[x], reverse=True)
        #print("mergeTree frequent "+str(self.frequent))
        print("mergeTree frequent "+str(mining_order))
        
        # merge header
        print("self.header: "+str(self.headers))
        print("other.header: "+str(other.headers))
        
        #merge tree
        #erch node in tree2, find in tree1 and update tree1:
        #get node in tree2
        #find it position in tree1 from parent to children
        
        for item in mining_order:
            if item in mining_order1:
                node = self.headers[item]
                suffixesV0 = []
                while node is not None:
                    suffixesV0.append(node)
                    node = node.link
                #print("*************************** 1:\t"+str(item) +"\t"+str(suffixesV0))
                for suffix in suffixesV0:
                    path = [suffix.value]
                    parent = suffix.parent
                
                    while parent.parent is not None:
                        path.append(parent.value)
                        parent = parent.parent
                
                    print("---------- 1:  ("+ str(len(suffix.children)) +") "+ " \t path: "+ str(path))
                    
            if item in mining_order2:
                node = other.headers[item]
                suffixesV0 = []
                while node is not None:
                    suffixesV0.append(node)
                    node = node.link
                #print("*************************** 2:\t"+str(item) +"\t"+str(suffixesV0))
                
                for suffix in suffixesV0:
                    path = [suffix.value]
                    parent = suffix.parent
                
                    while parent.parent is not None:
                        path.append(parent.value)
                        parent = parent.parent
                
                    print("--         2:  ("+ str(len(suffix.children)) +") "+ " \t path: "+ str(path))
                        
        for item2 in mining_order2:
            print("***************************:\t"+str(item2)+"   "+str(item2))
            if item2 in mining_order1: # (item == item2):
                node = self.headers[item2]
                suffixesV0 = []
                while node is not None:
                    suffixesV0.append(node.parent.value)
                    node = node.link
                    
                node = self.headers[item2]
                while node.link is not None:
                    node = node.link
                #extend link of node in self
                node.link = other.headers[item2]
                               
                node2 = other.headers[item2]
                suffixesV1 = []
                while node2 is not None:
                    #suffixes2.append(node2)
                    suffixesV1.append(node2.parent.value)
                    node2 = node2.link

                node = self.headers[item2]
                suffixesV = []
                while node is not None:
                    #suffixes.append(node)
                    suffixesV.append(node.parent.value)
                    node = node.link
                
                print(suffixesV0)
                print(suffixesV1)
                print(suffixesV)

                
                #for suffix in suffixes:
                #    frequency = suffix.count
                #    path = []
                #    parent = suffix.parent
                #
                #    while parent.parent is not None:
                #        path.append(parent.value)
                #        parent = parent.parent
                #
                #    print("1: "+str(suffix.parent.value) +" ("+ str(frequency) +" "+ str(len(suffix.children)) +") "+ str(suffix.value) +" \t path: "+ str(path))
             
                

                #print(str(item2)+", suffixes: "+ str(suffixesV2))
                
                #for suffix2 in suffixes2:
                #    frequency2 = suffix2.count
                #    path2 = []
                #    parent2 = suffix2.parent
                #    
                #    while parent2.parent is not None:
                #        path2.append(parent2.value)
                #        parent2 = parent2.parent
                #    
                #    print("2: "+str(suffix2.parent.value) +" ("+ str(suffix2.count) +" "+ str(len(suffix2.children)) +") "+ str(suffix2.value) +" \t path: "+ str(path2))
                #    
                #    #for i in range(frequency):
                #    #    conditional_tree_input.append(path)
                
                #print("conditional_tree_input: "+ str(conditional_tree_input))
            
            else:
                print("new node: "+str(item2))
                self.frequent[item2]  = other.frequent[item2]
                self.headers[item2] = other.headers[item2]
                
                node = other.headers[item2]
                suffixes = []
                #suffixesV = []
                while node is not None:
                    suffixes.append(node)
                    node = node.link

                #print(str(item)+", suffixes2: "+ str(suffixesV))
            
                for suffix in suffixes:
                    frequency = suffix.count
                    path = []
                    parent = suffix.parent
                
                    while parent.parent is not None:
                        path.append(parent.value)
                        parent = parent.parent
                
                    print("3: "+str(suffix.parent.value) +" ("+ str(frequency) +" "+ str(len(suffix.children)) +") "+ str(suffix.value) +" \t path: "+ str(path))
             
             
                suffixes1 = []
                #conditional_tree_input = []
                node2 = self.headers[item2]
                while node2 is not None:
                    suffixes1.append(node2)
                    node2 = node2.link

                #print(str(item2)+", suffixes: "+ str(suffixesV2))

                for suffix in suffixes1:
                    frequency2 = suffix.count
                    path = []
                    parent = suffix.parent
                    
                    while parent.parent is not None:
                        path.append(parent.value)
                        parent = parent.parent
                    
                    print("4: "+str(suffix.parent.value) +" ("+ str(suffix.count) +" "+ str(len(suffix.children)) +") "+ str(suffix.value) +" \t path: "+ str(path))
                    
        # merge root
        for node in listParrent1:
            listtmp[node.value] = i
            i+=1
        print(listtmp)
        
        for node in listParrent2:
            print(node.value)
            if node.value in listtmp.keys():
                print("meger node")
                self.root.children[listtmp[node.value]].mergeNode(node)
            else:
                print("add node")
                self.root.children.append(node)
        print("ROOT: "+str(self.root.value)+"\t child of root: "+str(len(self.root.children)))
        #return self
    '''
    '''
    from root, add root to stack, get all child and add to stack
    get last stack element, get chill and push to stack until child is 0, print path
    pop stack check chils of node if path of child is set, remove from stack
    '''
    def printTree(self):
        #print("root: "+ str(node.value))
        print("ROOT: "+str(self.root.value)+"\t SubTree of root: "+str(len(self.root.children)))
        listParrent = self.root.children
        trace = {}
        while len(listParrent)>0:
            node = listParrent.pop()
            #print("Num of pa: "+str(len(listParrent)))
            cnode = []
            for nod in node.children:
                cnode.append(nod.value)
            print("P:"+str(node.parent.value) +"\t N "+str(node.value)+" \t C:"+ str(node.count)+"\t nc:"+str(len(node.children))+"\t"+str(cnode))
            #print(self.headers[node].children)
            #trace[node.value] = node.count
            if(len(node.children)>0):
                listParrent.extend(node.children)
        
    def printPattern(self):
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x], reverse=True)
        print("ROOT: "+str(self.root.value)+"\t SubTree of root: "+str(len(self.root.children)))
        listParrent = self.root.children
        for subRoot in listParrent:
            print("Root "+ subRoot.value)
        for item in mining_order:
            suffixes = []
            suffixesNode = []
            #conditional_tree_input = []
            node = self.headers[item]
            while node is not None:
                suffixes.append(node)
                #suffixesNode.append([node.value,node.parent.value])
                node = node.link
            #print("item: "+str(item)+", suffixes: "+str(suffixesNode))
                            
            for suffix in suffixes:
                #frequency = suffix.count
                path = [suffix.value]
                parent = suffix.parent
                childs = []
                for v in suffix.children:
                    childs.append(v.value)
                    
                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent
                
                if len(path)>0:
                    if(len(suffix.children) > 0):
                        print("Count: "+str(suffix.count) +"\t"+str(len(suffix.children))+"\t path: "+ str(path[::-1])+"\t-> (childs:" +str(childs)+")")
                    else:
                        print("Count: "+str(suffix.count) +"\t"+str(len(suffix.children))+"\t path: "+ str(path[::-1]))
                    #for i in range(frequency):
                    #    conditional_tree_input.append(path)
                else:
                    print("item without path\t count: "+str(suffix.count) +"\t-> (childs: "+str(len(suffix.children))+":" +str(childs)+")" )
                    
            '''
            if len(conditional_tree_input)>0:
                subtree = FPTree(conditional_tree_input, 0, item, self.frequent[item])
                subtree.printPattern()
            '''
        
        '''
        while listNode is not None:
            currentNode = listNode.pop()
            for tmp in currentNode.children:
                if(tmp not in listNode):
                    listNode.append(tmp)
            listViss.append(currentNode)
        '''

def buildFPTree(transactions, support_threshold):
    rootTree = FPTree(transactions, support_threshold, None, None)
    print("buildFPTree "+str(rootTree.root.value))
    return rootTree

def find_frequent_patterns(tree, support_threshold):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    return tree.mine_patterns(support_threshold)
    
def generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dict
    of association rules in the form
    {(left): ((right), confidence)}
    """
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules


def runFPtreeMerge(transactions, threshold):

    transactions1 = transactions[0]
    transactions2 = transactions[1]

    #for tran in transactions:
    #    print(" transaction: "+ str(tran))
    
    rootTree1 = FPTree(transactions1, threshold, None, None)
    
    #rootTree1.printPattern()
    
    rootTree2 = FPTree(transactions2, threshold, None, None)
    #rootTree2.printTree()
    #rootTree2.printPattern()
    
    startTime = time.time()
        
    rootTree1.BIT_FPGrowth(rootTree2)
    
    #rootTree1.printPattern()
    
    patterns1 = rootTree1.mine_patterns(threshold)
    endTime = time.time() - startTime
    count  = 0
    for patte in patterns1:
        print(str(count)+"\t pattern: "+ str(patte))
        count +=1
    
    print("FPtreeMerge take total time: "+str(endTime)) 
    
    
    '''
    #patterns1 = find_frequent_patterns(rootTree1, 2)
    for patte in patterns:
        print(" pattern: "+ str(patte))
        
    rules = generate_association_rules(patterns1, 0.7)
    for rule in rules:
        print(" rule: " + str(rule))
    '''
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))

    return 0
    
    
def test(transactions, threshold):

    transactions1 = transactions[0]
    transactions2 = transactions[1]
    startTime = time.time()

    #for tran in transactions:
    #    print(" transaction: "+ str(tran))
    
    rootTree1 = FPTree(transactions[0], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    #rootTree1.printPattern()
    
    startTime = time.time()
    rootTree2 = FPTree(transactions[1], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    #rootTree2.printTree()
    #rootTree2.printPattern()
    
    startTime = time.time()
    rootTree3 = FPTree(transactions[2], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree4 = FPTree(transactions[3], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree5 = FPTree(transactions[4], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree6 = FPTree(transactions[5], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree7 = FPTree(transactions[6], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree8 = FPTree(transactions[7], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree9 = FPTree(transactions[8], threshold, None, None)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    '''
    rootTree1.mergeTree(rootTree2)
    
    rootTree1.mergeTree(rootTree3)
    
    rootTree1.mergeTree(rootTree4)
    
    rootTree1.mergeTree(rootTree5)
    
    rootTree1.mergeTree(rootTree6)
    
    rootTree1.mergeTree(rootTree7)
    
    rootTree1.mergeTree(rootTree8)
    
    rootTree1.mergeTree(rootTree9)
    '''
    
    #rootTree1.printTree()
    #rootTree1.printPattern()
    
    
    startTime = time.time()
    rootTree1.BIT_FPGrowth(rootTree2)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))    
    
    startTime = time.time()
    rootTree4.BIT_FPGrowth(rootTree3)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
        
    startTime = time.time()
    rootTree6.BIT_FPGrowth(rootTree5)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree8.BIT_FPGrowth(rootTree7)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree10.BIT_FPGrowth(rootTree9)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree1.BIT_FPGrowth(rootTree4)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
    startTime = time.time()
    rootTree8.BIT_FPGrowth(rootTree6)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))

    startTime = time.time()
    rootTree1.BIT_FPGrowth(rootTree8)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))

    startTime = time.time()
    rootTree1.BIT_FPGrowth(rootTree10)
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    
        
    startTime = time.time()
    patterns1 = rootTree1.mine_patterns(threshold)
    endTime = time.time() - startTime
    print("mine_patterns take total time: "+str(endTime))
    
    '''
    patterns1 = find_frequent_patterns(tree3, 2)
    for patte in patterns:
        print(" pattern: "+ str(patte))
        
    rules = fpg.generate_association_rules(patterns1, 0.7)
    for rule in rules:
        print(" rule: " + str(rule))
    
    endTime = time.time() - startTime
    print("FPtreeMerge take total time: "+str(endTime))
    '''
    return 0
    
