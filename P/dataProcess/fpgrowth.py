import itertools
import dataProcess.Node as node

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
        print("batch start")
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.batch = self.build_batch(transactions, self.frequent)
        print("batch end")

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
        #count = 0
        #idx=0
        flag1 = False
        flag2 = False

        #print("item "+str(item))
        mBatch = []
        for pattern in self.batch:
            #print("--"+str(sorted(pattern.value)) + " "+ str(pattern.count))
            sa = set(pattern.value)
            sb = set(item)
            c = sa.intersection(sb)
            #d = c
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
                        
        if flag1==False:
            #print("add new node "+str(item))
            self.batch.append(PatternNode(item, count))
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
            #d = c
            newNode = PatternNode(sorted(c), pattern.count+count)
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
                        
        if flag1==False:
            #print("add new node "+str(item))
            self.batch.append(PatternNode(item, count))
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

    def printBatch(self):
        for pattern in seft.batch:
            print(str(pattern.count)+ " "+str(pattern.value))


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
        
        root = FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)
        print("build_fptree "+str(root.value) +" "+ str(root_value))
        print("frequent "+str(frequent.keys()))
        print("headers " +str(headers.keys()))
        return root

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
        print("zip_patterns")
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
        print("generate_pattern_list")
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
                              key=lambda x: self.frequent[x])
        
        print("mine_sub_trees "+str(mining_order))

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
            print(str(item)+", suffixes: "+str(suffixes))
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
    self.frequent: all node with frequency
    self.header: list of root
    self.root: list of node in header, include link to child
    - update frequency list
    - merge header: self.header + other.header
    - merge other tree into FP tree: self + other, from root of self compare with root of other, update count of node, loop in subtree
    """
    def mergeTree(self, other):
        #newTree ={}
        #update frequency
        tmp1 = []
        for i in self.frequent.keys():
            notmp = self.root.get_child(i)
            if notmp is not None:
                tmp1.append(notmp)
        tmp2 = []
        for i in other.frequent.keys():
            notmp = other.root.get_child(i)
            if notmp is not None:
                tmp2.append(notmp)

        print("Child srft: " + str(tmp1))
        print("Child other: " + str(tmp2))
        print("mergeTree self: "+str(self.root.value))
        print("mergeTree other: "+str(other.root.value))
        print("mergeTree self frequent: "+str(self.frequent))
        print("mergeTree other frequent : "+str(other.frequent))
        
        items1 = list(self.frequent.keys())
        items2 = list(other.frequent.keys())
        '''
        for item1 in items1:
            if item1 in items2:
                self.frequent[item1] = self.frequent[item1] + other.frequent[item1]
        for item2 in items2:
            if item2 not in items1:
                self.frequent[item2]  = other.frequent[item2]
        '''
        
        mining_order1 = sorted(self.frequent.keys(), key=lambda x: self.frequent[x])
        print("mergeTree frequent "+str(self.frequent))

        mining_order2 = sorted(other.frequent.keys(), key=lambda x: other.frequent[x])
        print("mergeTree frequent "+str(other.frequent))
        
        #print("mine_sub_trees "+str(mining_order))
        #update header
        hitems1 = list(self.headers.keys())
        hitems2 = list(other.headers.keys())
        tmp3 = {}
        for h in hitems1:
            tmp3[h] = self.headers[h].value
        tmp4 = {}
        for h in hitems2:
            tmp4[h] = other.headers[h].value

        print("mergeTree self headers:  "+str(tmp3))
        print("mergeTree other headers: "+str(tmp4))
        '''
        for item1 in hitems1:
            if item1 in items2:
                self.header[item1] = self.headers[item1] + other.headers[item1]
        for item2 in hitems2:
            if item2 not in items1:
                self.header[item2]  = other.headers[item2]
        
        print("mergeTree "+str(self.headers.keys()))
        '''
        '''
        merge tree
        erch node in tree2, find in tree1 and update tree1:
        get node in tree2
        find it position in tree1 from parent to children
        '''
        for item in mining_order2:
            node = other.headers[item]
            suffixes = []
            suffixesV = []
            while node is not None:
                suffixes.append(node)
                #print(node.value)
                suffixesV.append(node.value)
                node = node.link

            print(str(item)+", suffixes2: "+ str(suffixesV))
            
            for item2 in mining_order1:
                suffixes2 = []
                conditional_tree_input = []
                node2 = self.headers[item2]
                suffixesV2 = []
                while node2 is not None:
                    suffixes2.append(node2)
                    #print(node.value)
                    #suffixesV2.append(node2.value)
                    node2 = node2.link

                #print(str(item2)+", suffixes: "+ str(suffixesV2))

                for suffix2 in suffixes2:
                    frequency = suffix2.count
                    path = []
                    parent = suffix2.parent
                    
                    while parent.parent is not None:
                        path.append(parent.value)
                        parent = parent.parent
                    
                    for i in range(frequency):
                        conditional_tree_input.append(path)
                    
                #print("conditional_tree_input: "+ str(conditional_tree_input))
            

        #first = items[0]
        #child = node.get_child(first)
        
        #for(patt in self.value):
        #    newTree.append(patt)
        return self
    def printTree(self):
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])
        listNode = []
        node = self.root
        listNode.append(node)
        listViss = []
        print("root: "+ str(node.value))
        print(list(mining_order))
        #print(list(self.headers.keys()))
        
        for item in mining_order:
            suffixes = []
            suffixesNode = []
            conditional_tree_input = []
            node = self.headers[item]
            while node is not None:
                suffixes.append(node)
                suffixesNode.append(node.value)
                node = node.link
                
                
            #print(str(item)+", suffixes: "+str(suffixesNode))

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
    #tree = FPTree(transactions, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)


def find_frequent_patterns_batch(transactions, support_threshold):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    return Batch(transactions, support_threshold)
    #return batch #.mine_patterns(support_threshold)

def mergeBatch(transactions1, transactions2):
    for itemA in transactions2.batch:
        #print(" itemA: "+ str(itemA.value) +" "+ str(itemA.count))
        transactions1.insert_batch2(itemA.value, itemA.count)
    return transactions1
    

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
