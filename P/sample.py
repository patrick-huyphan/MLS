#!/usr/bin/python
# MLS

import numpy as np
import dataProcess.dataIO.read as ior
import dataProcess.dataIO.write as iow
import dataProcess.ADMM as admm

print("hello")

iow.write("test write")

ior.read("test read")

admm.run("ADMM")