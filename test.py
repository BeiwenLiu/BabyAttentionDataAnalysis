#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:18:06 2016

@author: Beiwen Liu
"""
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy
from scipy import interpolate
import pandas as pd
from openpyxl import Workbook
import openpyxl
import os.path
from django.utils.encoding import smart_str
from openpyxl.styles import colors
from openpyxl.styles import Font, Color

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
    medianX = median(xValues)
    
    threshold = raw_input("State your offset from average\n")
    
    totalTime = timeValues[-1] - timeValues[0]
    distractedTime = calculateTotalDistractionTime(medianX,threshold,xValues,timeValues)
    
    print "Average X", averageX
    print "Median", medianX
    print totalTime
    print "distraction: " + str(round(distractedTime/totalTime*100,2)) + "%"
    
   
    newTimeValues = []
    newxValues = []
    for i in range(0,len(timeValues)):
        if timeValues[i] not in newTimeValues:
            newTimeValues.append(timeValues[i])
            newxValues.append(xValues[i])
    
    xStart,yStart,xEnd,yEnd,averageDistance = calculations(medianX,threshold,newxValues,newTimeValues)
    graphX(xStart,yStart,xEnd,yEnd,timeValues,xValues, medianX) #plots everything
    
    #histogram(xValues) # To enable histogram, uncomment here
    
    total = findDuration(xStart,xEnd)
    print round(total/totalTime*100,2)
    print "distraction: " + str(round(distractedTime/totalTime*100,2)) + "%"
    
    saveToExcel(median(xValues),averageX, xStart,xEnd,averageDistance,round(total/totalTime*100,2),round(distractedTime/totalTime*100,2),threshold)
    
def saveToExcel(median,mean,startTime,endTime,averageXDistance,dataDistractionPercent,calculatedDistractionPercent,threshold):
    new_file = 'results.xlsx'
    wb = openpyxl.Workbook()
    wb.create_sheet(index=0, title='Results') #Creating sheets based on original excel
    ws = wb.worksheets[0]
    ws['A1'] = 'Start Time'
    ws['B1'] = 'End time'
    ws['C1'] = 'Average X Distance'
    ws['D1'] = 'Median Center'
    ws['E1'] = 'Mean Center'
    ws['F1'] = 'Data Distraction Percentage'
    ws['G1'] = 'Caulcated Distraction Percentage'
    ws['H1'] = 'Offset Threshold'
    ws['D2'] = str(median)
    ws['E2'] = str(mean)
    for x in range(0,len(startTime)):
        ws['A{}'.format(x+2)] = str(startTime[x])
    for x in range(0,len(endTime)):
        ws['B{}'.format(x+2)] = str(endTime[x])
    for x in range(0,len(averageXDistance)):
        ws['C{}'.format(x+2)] = str(averageXDistance[x])
    ws['F2'] = str(dataDistractionPercent)
    ws['G2'] = str(calculatedDistractionPercent)
    ws['H2'] = str(threshold)
    wb.save(new_file)

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
    averageDistance = []

    tempDistance = []
    
    
    for x in range(2,len(xValues)):
        currentY = xValues[x] # X Distance Value
        currentX = timeValues[x]
        if ~found and (previousY < threshold + averageX and currentY > threshold + averageX):
            startX = slopeX(currentY,previousY,currentX,previousX,threshold + averageX)
            startY = averageX + threshold
            found = True
            
        if found:
            tempDistance.append(currentY)
        
        if found and (previousY > threshold + averageX and currentY < threshold + averageX):
            
            endX = slopeX(currentY,previousY,currentX,previousX,threshold + averageX)
            endY = averageX + threshold
            found = False
            startXY.append([startX,startY]) #column 0 is the time and column 1 is the "x" distance
            endXY.append([endX,endY])
            averageDistance.append(sum(tempDistance) / len(tempDistance))
            tempDistance = []
            
        previousY = currentY
        previousX = currentX

    return [row[0] for row in startXY], [row[1] for row in startXY], [row[0] for row in endXY], [row[1] for row in endXY], averageDistance
        

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
    #print y2,y1,x2,x1,threshold
    slope = (y2 - y1)/(x2 - x1)
    b = y1-slope*x1
    xVal = (threshold - b) / slope
    
    #print "original: " + str(x2)
    #print "new : " + str(xVal)
    return xVal
    
def graphXY(x,y):
    axes = plt.gca()
    plt.scatter(x, y, s=1)
    axes.set_xlim([0,1])
    axes.set_ylim([-2,3])
    plt.show()

def graphX(startX,startY,endX,endY,time,x,middleX):
    middleXList = []
    for num in range(0,len(time)):
        middleXList.append(middleX)
   
    plt.plot(time,x)
    plt.plot(time,middleXList, 'r') #Graphs center line according to middleX which can be average or median
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

#Histogram for distribution of distances 
def histogram(sa):
    
    # Get histogram of random data
    bins = []
    x = .1
    while x < 1:
        bins.append(x)
        x += .01
        
    y, x = np.histogram(sa, bins=bins)
    
    # Correct bin placement
    x = x[1:]
    
    # Turn into pandas Series
    hist = pd.Series(y, x)
    
    # Plot
    ax = hist.plot(kind='bar', width=1, alpha=0.5, align='center')
    ax.set_title('Uniform Bin Distribution of Distance Look Away')
    ax.set_xlabel('Distance Look Away')


    
#ase()
main()
#data_listener()
#practice()