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
	for line in f:
		#print(line)
		mi =int(line.split()[-1])
		print("MAX in line: "+str(mi))
		llist = [0]*mi
		for s in line.split():
			llist[int(s)-1] = 1
			#print(s)
		print(llist)
	f.close()