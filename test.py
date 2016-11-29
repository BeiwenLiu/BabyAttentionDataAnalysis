#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:18:06 2016

@author: MacbookRetina
"""
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import threading
import random


data = []

def main():
    x = loadmat('10912.mat')
    xValues = []
    yValues = []
    timeValues = []

    #print x['EyeData'][4][0]

    for y in range(0,40611):
        xValues.append(x['EyeData'][y][4])
        yValues.append(x['EyeData'][y][5])
        timeValues.append(x['EyeData'][y][0])
        
    #graphXY(xValues,yValues)
       
    #Drop all -1 values within X Values    
    pre = np.array(xValues)
    preT = np.array(timeValues)
    sa = np.ma.masked_where(pre == -1, pre)
    
    nXValues = np.ma.masked_array(xValues, sa.mask)
    nTimeValues = np.ma.masked_array(timeValues, sa.mask)
    
    xValues = np.ma.compressed(nXValues).tolist()[1:]
    timeValues = np.ma.compressed(nTimeValues).tolist()[1:]
    
    averageX = sum(xValues)/len(xValues)
    
    threshold = raw_input("State your offset from average\n")
    
    totalTime = timeValues[-1] - timeValues[0]
    distractedTime = 0
    for x in range(1,len(xValues)):
        if (xValues[x] > averageX + float(threshold)) or (xValues[x] < averageX - float(threshold)):
            distractedTime = distractedTime + (timeValues[x]-timeValues[x-1])
   
    print averageX
    print distractedTime
    print totalTime
    print "distraction: " + str(round(distractedTime/totalTime*100,2)) + "%"
    graphX(timeValues,xValues)
    
def graphXY(x,y):
    axes = plt.gca()
    plt.scatter(x, y, s=1)
    axes.set_xlim([0,1])
    axes.set_ylim([-2,3])
    plt.show()

def graphX(time,x):
    axes = plt.gca()
    plt.scatter(time,x)
    #axes.set_xlim([0,1])
    #axes.set_ylim([-2,3])
    plt.show()
    
def ase():
    plt.axis([0, 10, 0, 1])
    plt.ion()

    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        plt.pause(0.05)

    while True:
        plt.pause(0.05)

#ase()
main()
#data_listener()