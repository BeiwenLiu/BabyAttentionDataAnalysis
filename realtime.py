#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:38:09 2016

@author: MacbookRetina
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

xValues = []
yValues = []

def graph():
    
    main()
    counter = 0
    plt.ion()
    
    
    
    fig, ax = plt.subplots()
    
    plot = ax.scatter([], [])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    while counter < 40611:
        # get two gaussian random numbers, mean=0, std=1, 2 numbers
        point = np.array([xValues[counter],yValues[counter]])
        # get the current points as numpy array with shape  (N, 2)
        array = plot.get_offsets()
    
        # add the points to the plot
        array = np.append(array, point)
        plot.set_offsets(array)
    
        # update x and ylim to show all points:
        ax.set_xlim(array[:, 0].min() - 0.5, array[:,0].max() + 0.5)
        ax.set_ylim(array[:, 1].min() - 0.5, array[:, 1].max() + 0.5)
        # update the figure
        fig.canvas.draw()
        counter = counter + 1
    
def main():
    x = loadmat('10912.mat')
    for y in range(0,40611):
        xValues.append(x['EyeData'][y][4])
        yValues.append(x['EyeData'][y][5])
        
graph()