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

    def mergeNode(self, node2):
        print("mergeNode "+str(self.value))
        self.count += node2.count
        #subNode1 = self.children
        #subNode2 = node2.children
        listTmp = {}
        i = 0
        for node in self.children:
            listTmp[node.value] = [i,node.count]
            i +=1
        print(listTmp)
            
        listTmp2 = {}
        i = 0
        for node in node2.children:
            listTmp2[node.value] = [i,node.count]
            i +=1
        print(listTmp2)

        for node in node2.children:
            if node.value in listTmp.keys():
                print("merge "+ str(node.value))
                self.children[listTmp[node.value][0]].mergeNode(node)
            else:
                print("add "+ str(node.value))
                self.children.append(node)    
        
        listTmp3 = {}
        i = 0
        for node in self.children:
            listTmp3[node.value] = [i,node.count]
            i +=1
        print(str(self.value)+ "----"+str(listTmp3))
        
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
