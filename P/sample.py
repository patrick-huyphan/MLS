#!/usr/bin/python
# MLS

import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm
import dataProcess.fpgrowth as fpg

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
	print(" transactions: "+ str(transactions))
	patterns = fpg.find_frequent_patterns(transactions, 2)
	print(" patterns: "+ str(patterns))
	rules = fpg.generate_association_rules(patterns, 0.7)
	print(" rules: " + str(rules))
	
	ior.read2Matrix("C:\Users\patrick_huy\OneDrive\Documents\long prj\FPC\_DataSets\mushroom.dat")