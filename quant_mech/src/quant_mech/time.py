'''
Created on 16 Apr 2015

Module containing functions to get current time to use for analysing performance of algorithms

@author: rstones
'''
from datetime import datetime

def getTime():
    return datetime.now()

def duration(t1, t2):
    return t1 - t2