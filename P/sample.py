#!/usr/bin/python
# MLS

import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.fpgrowth as fpg
from numpy import arange,array,ones,linalg
from pylab import plot,show

def lnr():
	xi = arange(0,9)
	A = array([ xi, ones(9)])
	# linearly generated sequence
	y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
	w = linalg.lstsq(A.T,y)[0] # obtaining the parameters

	# plotting the line
	line = w[0]*xi+w[1] # regression line
	plot(xi,line,'r-',xi,y,'o')
	show()
	
if __name__ == "__main__":
	print("main start")

	iow.write("test write")
	ior.read("test read")
	admm.run("ADMM")
	
	transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
	transactions = ior.read2RawData("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat",20)
	print(" transactions: "+ str(transactions))
	patterns = fpg.find_frequent_patterns(transactions, 2)
	print(" patterns: "+ str(patterns))
	rules = fpg.generate_association_rules(patterns, 0.7)
	print(" rules: " + str(rules))
	
	
	
	ior.read2SparseMatrix("C:\Data\Master\data_mining\data\data_694_446.csv")
	
	lnr()
	#ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")