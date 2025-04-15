# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:15:11 2020

@author: Jacob
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from analysis.data import autolyze as az
from analysis.data import read as rf


def getTraceFromShot(filepath,trace_name):  #not using times_list or graph_trace
    #Get the trace data:
    dat=rf.getdata(filepath, trace_name)
    data_length = len(dat)
    #Time and acquire hold time points and trace data respectively
    time = np.zeros(data_length)
    trace = np.zeros(data_length)
    for i, datum in enumerate(dat):
        time[i] = datum[0]
        trace[i] = datum[1]
    
    return trace, time

def plotTrace(shot,traceName='UV_PD_1',xlim=[None,None],processFun=None):
    '''
    

    Parameters
    ----------
    shot : string
        filepath of the shot.
    traceName : string, optional
        DESCRIPTION. The default is 'UV_PD_1'.
    xlim : list, optional
        Plot xlim. The default is [None,None].
    processFun : func, optional
        A function which takes an ndarray and returns
        an ndarray of the same length. The default is None.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''
    
    
    trace, time = getTraceFromShot(shot,traceName)

    if processFun:
        trace=processFun(trace)
        
    time=time-time[0]
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(time*1e3,trace,'r.', label = 'Full trace')


    ax.legend()

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.grid()
    ax.set_xlim(xlim)
    
    return fig, ax
    

#Get an analog trace from a single file:
def get_trace_points(filepath,trace_name, times_list, graph_trace=False, convolve_points=20):
    #Get the trace data:
    dat=rf.getdata(filepath, trace_name)
    data_length = len(dat)
    #Time and acquire hold time points and trace data respectively
    time = np.zeros(data_length)
    acquire = np.zeros(data_length)
    for i, datum in enumerate(dat):
        time[i] = datum[0]
        acquire[i]=datum[1]
    time_step = time[1] - time[0]    
    #Do a running average over the trace with convolve_points points
    acquire_conv = scipy.signal.convolve(acquire,np.ones(convolve_points)/convolve_points,mode='same')
    values_return = np.zeros(len(times_list)) 
    
    for i, time_in_trace in enumerate(times_list):
        if time_in_trace < 0:
            raise RuntimeError("Time in trace needs to be >= 0")
        elif time_in_trace >= time[-1]-time[0]:
            raise RuntimeError("Time in trace needs to be <{:.2f} ms".format((time[-1]-time[0])*1000))
        values_return[i] = acquire_conv[np.int((time_in_trace)/time_step)]
    colors = plt.cm.jet(np.linspace(0,1,len(times_list)))

    if graph_trace:
        fig,ax = plt.subplots()
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Voltage [V]")
        ax.plot((time-time[0])*1e3, acquire_conv)
        ax.set_xlim(xmin=0)#,xmax=(np.max(time_in_trace))*1e3+0.01)
        for i, time_in_trace in enumerate(times_list):
            ax.axvline(x=(time_in_trace)*1e3,c=colors[i],ls='--')
        plt.title('PD signal')
        #plt.savefig(folder+'\\PD_curve_conv_'+str(MW_detuning)+'.png',bbox_inches='tight')
        
    return values_return  

#analyze analog traces in folder "folder" with changing parameter1:
#times_list is list of times in seconds where we pick up the trace value
def get_traces_vsparameter(folder, parameter1, times_list=[] ,\
                           trace_name = 'MOT_Fluorescence_PD', do_graph=False,\
                           convolve_points=20):
    filepaths, filenames = rf.open_all_HDF5_in_dir(folder)
    parameters1 = []
    #parameters2 = []
    trace_values_list = []
    for item in filepaths:
        trace_values = get_trace_points(item,trace_name, times_list,do_graph ,\
                                        convolve_points)
        parameters1.append(az.getParam(item,parameter1))
        trace_values_list.append(trace_values)
    trace_values_list = np.array(trace_values_list) 
    parameters1 = np.array(parameters1)

    if do_graph:
        fig,ax = plt.subplots()
        ax.set_xlabel(parameter1)
        ax.set_ylabel("Voltage [V]")
        ax.plot(parameters1, trace_values_list[:,0],'o')
        plt.title(trace_name)
        ax.grid()
        plt.savefig('PlottingTracevs'+parameter1+'.png')
        
    return parameters1, trace_values_list