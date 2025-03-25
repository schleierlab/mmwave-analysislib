# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:16:05 2022

@author: Quantum Engineer
"""


import numpy as np
from numpy.fft import fftshift, fftfreq, fft
import os
import matplotlib.pyplot as plt
import h5py
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.signal import filtfilt, butter, iircomb, iirnotch

from analysis.graphic.export import dateString
from analysis.data import autolyze as az
from analysis.data import h5lyze as hz
from analysis.graphic.plotting import setAxParams, transparent_edge_plot


def padFolderNames(target, folder=None, d = None , m = None, y = None):
    
    directory = az.getDirectory(target, d = d , m = m, y = y)
    
    if folder:
        directory = os.path.join(directory,folder)
        
    
    subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
    
    maxPadLength = 0
    for subfolder in subfolders:
        splitSubfolder = str.split(subfolder, '_')
        try:
            checkFloat = float(splitSubfolder[-1])
        except:
            raise Exception('The last part of one of the folder names is not numeric!')
        
        numLength = len(splitSubfolder[-1])
        if numLength > maxPadLength:
            maxPadLength = numLength
    
    newSubfolders = []
    for subfolder in subfolders:
        subdir = os.path.join(directory, subfolder)

        splitSubfolder = str.split(subfolder, '_')
        if '.' not in splitSubfolder[-1]:
            splitSubfolder[-1] = str(splitSubfolder[-1]).zfill(maxPadLength)
            newSubfolder = '_'.join(splitSubfolder)
        elif '.' in splitSubfolder[-1] and 'e' not in splitSubfolder[-1]:
            splitSubfolder[-1] = splitSubfolder[-1].ljust(maxPadLength, '0')
            newSubfolder = '_'.join(splitSubfolder)
        else:
            newSubfolder = subfolder
        newSubfolders.append(newSubfolder)
        newSubdir = os.path.join(directory, newSubfolder)
        
        os.rename(subdir, newSubdir)
        
    return newSubfolders
    
subfolders = padFolderNames('Squeezing_dds_mw', folder = '000000__a4thTestFolderForPadding')    
    
# print(newSubfolders)