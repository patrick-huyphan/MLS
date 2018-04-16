#!/usr/bin/python
# MLS
import numpy as np

def read(data):
    print("hello" + data)
    a = np.arange(15).reshape(3, 5)
    print (a)

def readFile(fileName):
    f = open(fileName, 'r')
    for line in f:
        print(line)
    f.close()

def read2RawData(fileName, fromx, tox, maxcol):
    f = open(fileName, 'r')
    mat = []
    count = 0
    #print("read2RawData")
    for line in f:
        
        if((fromx>0 and count< fromx) or (tox>0 and count>tox)):
            #print(str(count))
            count += 1
            continue
        #print("MAX in line: "+str(max))
        llist = []
        for s in line.split():
            llist.append(s)
            if(maxcol>0) and (int(s) > maxcol):
                break
            #print(s)
        #print(llist)
        mat.append(llist)
        count += 1
        #if(maxrow>0):
        #    if(count>maxrow):
        #        break
    f.close()
    return mat
    
def read2Matrix(fileName):
    f = open(fileName, 'r')
    max = 0
    # check max inline
    for line in f:
        mi = int(line.split()[-1])
        if(mi>max):max = mi
        #print("MAX in line: "+str(mi) +":"+ str(max))
    print("MAX: "+str(max))
    f.seek(0)
    # create matrix
    mat = []
    for line in f:
        #print("MAX in line: "+str(max))
        llist = [0]*max
        for s in line.split():
            llist[int(s)-1] = 1
            #print(s)
        #print(llist)
        mat.append(llist)
    f.close()
    return mat
    
def read2SparseMatrix(fileName):
    f = open(fileName, 'r')    
    sizehw= f.readline()
    w= int(sizehw.split(":")[1])
    h= int(sizehw.split(":")[2])
    mat = [[0 for x in range(w)] for y in range(h)] 
    for line in f:
        #print(line)
        tmp = line.split(":")
        mat[int(tmp[1])][int(tmp[2])]= float(tmp[3].strip('\n'))

    f.close()
    #print(mat)
    return mat    

def saveSparseMatrix(fileName, data):
    f = open(fileName, 'w')
    for row in data:
        wrow = ""
        count = 0
        for x in row:
            if x!=0:
                wrow = wrow+ str(count)+":"+str(x) +" "
            count= count+1
        f.write(wrow+"\n")
    f.close()
     
def read2RawData2(fileName, fromx, tox, maxcol):
    f = open(fileName, 'r')
    mat = []
    count = 0
    max = 0
    # check max inline
    for line in f:
        mi = int(line.split()[-1].split(":")[0])
        if(mi>max):max = mi
        #print("MAX in line: "+str(mi) +":"+ str(max))
    print("MAX: "+str(max))
    f.seek(0)
    
    for line in f:        
        if((fromx>0 and count< fromx) or (tox>0 and count>tox)):
            #print(str(count))
            count= count + 1
            continue
        #print("MAX in line: "+str(max))
        llist = []
        for s in line.split():
            #print(s)
            data = s.split(":")
            llist.append(data[0])
            if(int(data[0]) > maxcol):
                break
        print(llist)
        mat.append(llist)
        count= count + 1
        #if(maxrow>0):
        #    if(count>maxrow):
        #        break
    f.close()
    return mat

def rawData2matrix(fileName, fromx, tox, maxcol):
    f = open(fileName, 'r')
    mat = []
    count = 0
    max = 0
    # check max inline
    for line in f:
        mi = int(line.split()[-1].split(":")[0])
        if(mi>max):max = mi
        count= count +1
        #print("MAX in line: "+str(mi) +":"+ str(max))
    print("MAX: "+str(max) + " "+str(count))
    f.seek(0)
    
        # create matrix
    mat = np.zeros((count,max+1), dtype=float)
    count =0
    count2 = 0
    for line in f:
        #print("MAX in line: "+str(max))
        if((fromx>0 and count< fromx) or (tox>0 and count>tox)):
            #print(str(count))
            count= count + 1
            continue
        
        #llist = [0]*(max+1)
        for s in line.split():
            data = s.split(":")
            try:
                mat[count2][int(data[0])] = data[1]#llist[int(data[0])] = data[1]
            except:
                print( "except "+str(count2)+"-"+str(data[0]) +" value "+str(data[1]))
            if(int(data[0]) > maxcol):
                break
        count2 = count2+1

    f.close()
    #np.set_printoptions(threshold=np.inf)
    ret = np.array(mat)
    print(ret) 
    return ret
