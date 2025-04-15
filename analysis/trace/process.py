# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:15:02 2020

@author: Jacob
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import datetime
import h5py
import os

from .extract import getTraceFromShot
from analysis.fit.functions import lorentzian_offset
from analysis.data import autolyze as az




#analyze_traces takes parameters and trace_values and then fits a fitting_function to the 
#trace values. Trace values are transformed according to trace_transform,
#e.g. if trace_transform = subtract_trace_vals then first and second trace_values
#are subtracted for each parameter value and then this is used to fit the fitting function.
def analyze_traces(folder, parameters, trace_values, trace_transform,paramName = 'Run', do_fit = False, fitting_function = lorentzian_offset, trace_title=""):
    transformed_trace_values = trace_transform(trace_values)
    fig,ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel(paramName)
    ax.set_ylabel("Voltage [V]")
    ax.set_ylim(ymin=-0.11,ymax = 0.1)#For MOT trace
#    ax.set_ylim(ymin=3.4, ymax=3.6)#For DT trace
    ax.grid()
    ax.plot(parameters,transformed_trace_values,'o')
    if do_fit:
        coeffs0, covar0 = curve_fit(fitting_function,parameters,transformed_trace_values, p0=[0.1, .1, .7, -0.2])
        fit_x_values = np.linspace(parameters[0],parameters[-1],100)
        ax.plot(fit_x_values,lorentzian_offset(fit_x_values,*coeffs0),'-', label = r'Lorentzian FW {:.1f}, center {:.2f}'.format(coeffs0[1],coeffs0[2]))

        #print(coeffs0)
    
    plt.legend(fontsize='small',loc=4)
#    plt.title('PD signal')
    plt.title(trace_title)
    
    dateForPrinting = datetime.datetime.today().strftime('%Y-%m-%d')

    
    plt.savefig(folder+'\\'+dateForPrinting+'_MOTSignalvs'+paramName+'.png')

##Functions to transform a list of trace_values into a single number.
#E.g. if you want to 
def subtract_trace_vals(trace_values):
    return trace_values[:,0]-trace_values[:, 1]    
    
def ratio_trace_vals(trace_values):
    return trace_values[:, 1]/trace_values[:,0]

def second_trace_vals(trace_values):
    return trace_values[:, 1]

#checks whether there was a mot on a given run:
def checkLock(filepath, threshold = 1.3):
    times_list=[1e-2-1.8e-3,1e-2+0.3e-3]
    trace = getTraceFromShot(filepath,'MOT_Fluorescence_PD', times_list, graph_trace = True)
    
    print(trace[0])
    
    if trace[0] > threshold:
        isLocked = True
    else:
        isLocked = False
        
    with h5py.File(filepath) as f:
        group = f['data']
        group.attrs['isLocked'] = isLocked
        
    return isLocked

def checkUVsignal(target,folder='',traceName='UV_PD_1',threshold=0.1,d=None,m=None,y=None):
    
    directory=az.getDirectory(target,d,m,y)
    
    directory=os.path.join(directory,folder)
    
    subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    if len(subfolders)==0:
        subfolders=[directory]
        
    for subfolder in subfolders:
        print(f'######  Analyzing folder {subfolder}  ######')
        print()
        shots = az.getShots(os.path.join(directory,subfolder))


        for shot in shots:
            try:
                trace, time = getTraceFromShot(shot,traceName)
                signalOffset = np.mean(trace[-20:-1])
                trace = trace-signalOffset
                
                if np.all(trace<threshold):
                    print(f"Trace in shot {os.path.basename(shot)} doesn't have any values larger than {threshold}")
            except:
                print(f'Trace {traceName} not found in shot {os.path.basename(shot)}.')

def getDelta(tracePts, nPoint):
    delta = (np.sum(tracePts[0:nPoint]) - np.sum(tracePts[nPoint:len(tracePts)]))/np.sum(tracePts)
    return(delta)

def getIndicesAboveThreshold(trace,threshold):
    '''
    Get indices of trace above threshold*max(trace).
    '''
    traceMax = np.max(trace)
    condlist = trace > threshold#*traceMax
    
    return np.argwhere(condlist)[:,0]

def partitionIndices(indices, minSeparation = 1):
    '''
    Partitiones a list of monotonically increasing indices into
    sequences of adjecent indices, separated by minSeparation between
    the last point in one set and the first point in the next set.
    Example: partitionIndices([1,2,7,8]) returns ([5], ([1,2],[4,5]))
    '''
    steps = indices[1:] - indices[:-1]
    partIndices = np.argwhere(steps>minSeparation)[:,0]
    partitionedIndices = np.split(indices,(partIndices+1))
    separations = steps[partIndices]
    
    return separations, partitionedIndices

def getPeaksInTrace(trace, threshold, minSeparation = 1):
    '''
    Returns indices of peaks in trace and their value. Peaks are defined
    as roughly values of the signal >threshold*max(trace) separated by more 
    than minSeparation. This function asummes ther are no anomalous values in 
    the trace like a really high point not part of a peak or a drop of signal below
    threhsold within a peak.
    '''
    indices = getIndicesAboveThreshold(trace,threshold)
    separations, partitionedIndices=partitionIndices(indices,minSeparation)
     
    peakIndices = []
    peaks = []
     
    for peakIndicesSet in partitionedIndices:

        if len(trace[peakIndicesSet])!=0:
            maxInd = np.argmax(trace[peakIndicesSet])
            peakIndices.append(peakIndicesSet[maxInd])
            peaks.append(trace[peakIndicesSet[maxInd]])
         
    return peakIndices, peaks
         
def getTracePeakArray(shots, threshold, traceName, preprocessFun = None, minSeparation=1):
    ''' Return peak values from trace traceName in a list 
    of shots.
    preprocessFun is a function acting on a 1D array returning an array of the
    same size (ideally). This function should perserve peak structure of 
    the trace.
    '''
    
    peakArray = []
    
    for shot in shots:
        trace, time = getTraceFromShot(shot,traceName)
        if preprocessFun:
            trace = preprocessFun(trace)
        peakIndices, peaks = getPeaksInTrace(trace, threshold,minSeparation)
        peakArray.append(np.array(peaks))
        
    return np.array(peakArray)

def powerSpectralDensity(y, dt, normalize=False, return_unit=False):
    """
    Caclualtes power spectral density in a signal. Returns V^2/Hz or 1/Hz
    depending on whether it is told to normalize such that the rms power is 1.

    Parameters
    ----------
    y : numpy array
        Signal to be analyzed.
    dt : float
        Signal sampling period.
    normalize : bool, optional
        If true, the signal to be analyzed is normalized such that the mean
        signal power is 1 and the units of the PSD are 1/Hz. If true, the signal
        is not normalized and the PSD is in units of V^2/Hz. The default is False.

    Returns
    -------
    f : numpy array
        Array of frequency values corresponding to the PSD.
    PSD : numpy array
        Power spectral density in V^2/Hz if not normalized, or 1/Hz if normalized.
    unit : string
        Units of the quantity being returned

    """
    
    if normalize:
        # normalize such that the average power is 1
        y_rms = np.sqrt(np.mean(y**2))
        y = y / y_rms
        unit = '1/Hz'
    else:
        unit = 'V$^2$/Hz'
    
    n = len(y)
    
    # generate the frequencies
    f = np.fft.fftfreq(n, dt)
    df = f[1] - f[0]
    
    # take the fourier transform
    Y = np.fft.fft(y)
    
    # scale by number of points
    Y = Y / n
    
    # restrict ourselves to the positive frequency components
    f = f[:n//2]
    Y = Y[:n//2]
    
    # convert from voltage to power spectrum by taking mod squared
    PS = np.abs(Y)**2
    
    # multiply everything except the DC component by 2 to account for negative frequency components
    PS[1:] *= 2
    
    # divide by differential frequency to get a power spectral density
    PSD = PS / df
    
    if return_unit:
        return f, PSD, unit
    else:
        return f, PSD
