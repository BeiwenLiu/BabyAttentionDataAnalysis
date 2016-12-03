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
import numpy
from scipy import interpolate


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
    initialValue = timeValues[0]

    timeValues[:] = [x - initialValue for x in timeValues]
    
    averageX = sum(xValues)/len(xValues)
    
    threshold = raw_input("State your offset from average\n")
    
    totalTime = timeValues[-1] - timeValues[0]
    distractedTime = calculateTotalDistractionTime(averageX,threshold,xValues,timeValues)
    
    print averageX
    print median(xValues)
    print totalTime
    print "distraction: " + str(round(distractedTime/totalTime*100,2)) + "%"
    
   
    newTimeValues = []
    newxValues = []
    for i in range(0,len(timeValues)):
        if timeValues[i] not in newTimeValues:
            newTimeValues.append(timeValues[i])
            newxValues.append(xValues[i])
    
    xStart,yStart,xEnd,yEnd = calculations(averageX,threshold,newxValues,newTimeValues)
    graphX(xStart,yStart,xEnd,yEnd,timeValues,xValues)
    
    total = findDuration(xStart,xEnd)
    print round(total/totalTime*100,2)
    print "distraction: " + str(round(distractedTime/totalTime*100,2)) + "%"
    
#This method will find the start and end points
def calculations(averageX,threshold,xValues,timeValues):
    threshold = float(threshold)
    
    found = False
    
    currentY = 0
    previousY = xValues[1]
    currentX = 0
    previousX = timeValues[1]
    
    startY = 0 
    startX = 0
    endY = 0
    endX = 0
    
    startXY = []
    endXY = []
    
    for x in range(2,len(xValues)):
        currentY = xValues[x] # X Value
        currentX = timeValues[x]
        if ~found and (previousY < threshold + averageX and currentY > threshold + averageX):
            startX = slopeX(currentY,previousY,currentX,previousX,threshold + averageX)
            startY = averageX + threshold
            found = True
        
        if found and (previousY > threshold + averageX and currentY < threshold + averageX):
            
            endX = slopeX(currentY,previousY,currentX,previousX,threshold + averageX)
            endY = averageX + threshold
            found = False
            startXY.append([startX,startY]) #column 0 is the time and column 1 is the "x" distance
            endXY.append([endX,endY])
            
        previousY = currentY
        previousX = currentX

    return [row[0] for row in startXY], [row[1] for row in startXY], [row[0] for row in endXY], [row[1] for row in endXY]
        

def calculateTotalDistractionTime(averageX,threshold,xValues,timeValues):
    distractedTime = 0
    for x in range(1,len(xValues)):
        if (xValues[x] > averageX + float(threshold)) or (xValues[x] < averageX - float(threshold)):
            distractedTime = distractedTime + (timeValues[x]-timeValues[x-1])
            
    return distractedTime
    
def findDuration(start,end):
    duration = 0
    for x in range(0,len(start)):
        duration = duration + (end[0] - start[0])
    return duration
     
#This method will find the associated time with the "x" offset threshold as selected value between two known points       
def slopeX(y2,y1,x2,x1,threshold):
    print y2,y1,x2,x1,threshold
    slope = (y2 - y1)/(x2 - x1)
    b = y1-slope*x1
    xVal = (threshold - b) / slope
    
    print "original: " + str(x2)
    print "new : " + str(xVal)
    return xVal
    
def graphXY(x,y):
    axes = plt.gca()
    plt.scatter(x, y, s=1)
    axes.set_xlim([0,1])
    axes.set_ylim([-2,3])
    plt.show()

def graphX(startX,startY,endX,endY,time,x):
    axes = plt.gca()
   
    plt.plot(time,x)
    plt.scatter(startX,startY)
    plt.scatter(endX,endY)
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
        

def median(lst):
    return numpy.median(numpy.array(lst))

def practice():
    list1 = []
    list1.append([1,2])
    list1.append([2,3])
    print list1[0][0]


    
#ase()
main()
#data_listener()
#practice()