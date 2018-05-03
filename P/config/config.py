#!/usr/bin/python
# MLS

import sys

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]

def getRunningConfig(configFile):
    f = open(configFile, 'r')
    typeR = 0
    for line in f:
        print("config \n"+line)
        kv= line.split(":")
        if kv[0] == "type":
            typeR=kv[1]
    f.close()
    return typeR
    
def pythonVer():
    print (sys.version)
    return 0
