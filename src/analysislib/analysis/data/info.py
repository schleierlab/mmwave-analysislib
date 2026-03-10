# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:58:42 2020

@author: Quantum Engineer
"""

import numpy as np
from . import autolyze as az
import h5py
from . import h5lyze as hz



def getTweezerRunInfo(shot):
    '''
    Get tweezer frequencies from a shot. Returns numpy arrays.

    Parameters
    ----------
    shot : string
        Shot path.

    Returns
    -------
    x_freqs : np.array
    y_freqs : np.array
    x_increment : np.array
    y_increment : np.array

    '''
    x_freqs =  az.getParam(shot,'TW_x_freqs')
    y_freqs = az.getParam(shot,'TW_y_freqs')
    
    if x_freqs.shape:
        x_increment = np.diff(x_freqs)
    else:
        x_increment = 0
        y_freqs = np.array([y_freqs])

        
    if y_freqs.shape:
        y_increment = np.diff(y_freqs)
    else:
        y_freqs = np.array([y_freqs])
        y_increment = 0
        
    return x_freqs, y_freqs, x_increment, y_increment

def getTweezerAmplitudes(shot,device,channel):
    with h5py.File(shot,'r') as f:
        devices = f['devices']

        spectrumDevice=devices[device]
        for c, info in spectrumDevice['waveform_groups']['group 0'].items():
            if f'ch = {channel}' in c:
                dictionary=hz.datasetsToDictionary(info)
                
                amps= {e[0]*1e-6: e[3] for e in dictionary['pulse_data']}
    return amps
        