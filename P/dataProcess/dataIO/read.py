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