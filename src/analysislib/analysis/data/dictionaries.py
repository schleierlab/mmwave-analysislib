# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:02:55 2020

@author: Jacob
"""

import numpy as np
from analysis.math.statistics import runningMean

def atomDataFromDict(data, roi_key_1, roi_key_2, hstart, hend, RefTWROI, 
                     hwidth=10, step=10, F3shift = 0, 
                     scaleSignal = False, signalCalibrationDict=None):
    '''
    A function to extract atom signals and fractions from sequence data dictionary.
    It asummes data stored in 2D arrays with shape = (n_shots, ROIwidth)
    This function additionally applies the running mean to the data based
    on hstart, hend, hwidth (averaging window) and step.
    F3shift is the shift of the F=3 image w.r.t. F=4 image.
    '''
    

    
    A4 = np.array(data[roi_key_1])
    A3 = np.array(data[roi_key_2])
    
    A4 = np.apply_along_axis(runningMean, 1, A4[:,int(hstart):int(hend)], hwidth,step)
    A3 = np.apply_along_axis(runningMean, 1, A3[:,int(hstart)+F3shift:int(hend)+F3shift], hwidth,step)

    if scaleSignal:
        a = signalCalibrationDict['a'][RefTWROI]
        b = signalCalibrationDict['b'][RefTWROI]
        A3 = a*A3 - b*A4

    Af = A4/(A4+A3)
    
    return Af, A4, A3

# def calculateAtomFraction(data, roi_key_1, roi_key_2)