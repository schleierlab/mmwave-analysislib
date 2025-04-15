# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:48:53 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

def traceAll(covMatrix, n=None):
    """
    Given an nxn covariance matrix covMatrix, this function computes the trace
    for each of the 2n diagonals. n specifies the number of diagonals to consider
    to either side of the main diagonal
    """
    
    if not n:
        n = np.shape(covMatrix)[0]
    
    covTrace = np.zeros(2*n+1)
    
    for index in range(2*n+1):
        k = index - n
        myDiagonal = np.diag(covMatrix, k)
        covTrace[index] = np.mean(myDiagonal)
        
    return covTrace


def linearShift(data, angle, center=None, previewShift=False):
    '''
    Given an n-dimensional data matrix, this function iterates over the first 
    dimension (rowIdx) and shifts each in the second dimension by an amount given
    by tan(angle)*(rowIdx)
    '''
    
    nrows = np.shape(data)[0]
    
    if center==None:
        center = np.floor(nrows/2).astype('int')
    
    dataShifted = np.zeros(np.shape(data))
    
    dshift = np.tan(angle)
    
    for rowIdx, row in enumerate(data):
        shift = np.round((rowIdx-center)*dshift).astype('int')
        dataShifted[rowIdx] = np.roll(data[rowIdx], shift, axis=0)
        
    if previewShift:
        fig, ax = plt.subplots(figsize=(15,6))
        if (dataShifted.ndim)==2:
            img = dataShifted
        else:
            img = dataShifted[:,:,0]
        
        plt.imshow(img)
        plt.axis('off')
    
    return dataShifted