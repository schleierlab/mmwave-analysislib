# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:13:42 2021

@author: Quantum Engineer
"""

import numpy as np
import csv

def load_scope_csv(csv_file, scope_type, pop_scope):
    if scope_type == 'keysight':
        return load_keysight_csv(csv_file)
    elif scope_type == 'rigol':
        return load_rigolMSO_csv(csv_file, pop_scope)
    else:
        raise Exception(f"Scope '{scope_type}' not implemented")

def load_keysight_csv(csv_file):
    """

    Parameters
    ----------
    csv_file : TYPE
        Path to csv file.

    Returns
    -------
    t : numpy arry
        time trace.
    data : dictionary of numpy arrays
        data traces.

    """
    
    with open(csv_file, 'r', newline='') as f:
        # Prepare the csv reader
        reader = csv.reader(f, delimiter=',')
        
        # Pop off the first two lines
        header = next(reader)
        units = next(reader)
        
        # Prepare data containers
        t = []
        
        channels = header[1:]
        data = {ch: [] for ch in channels}
        
        n_skipped = 0
        
        # Read out the data
        for idx, row in enumerate(reader):
            bad_line = False
            
            try:
                row = [float(val) for val in row]
            except ValueError:
                # Something about this row didn't save correctly. 
                if len(t) == 0:
                    # We're at the beginning of the file. Just move on.
                    continue
                else:
                    # We're at the end of the file. Time to stop.
                    break
                
            t.append(row[0])
            
            for ch, val in zip(channels, row[1:]):
                data[ch].append(val)
                
        for ch in channels:
            data[ch] = np.array(data[ch])
            
        t = np.array(t)
                
        return t, data

def load_rigolMSO_csv(csv_file, pop_scope):
    """

    Parameters
    ----------
    csv_file : TYPE
        Path to csv file.

    Returns
    -------
    t : numpy arry
        time trace.
    data : dictionary of numpy arrays
        data traces.

    """
    
    with open(csv_file, 'r', newline='') as f:
        # Prepare the csv reader
        reader = csv.reader(f, delimiter=',')
        
        # Pop off the first two lines
        header = next(reader)
        units = next(reader)
        
        if pop_scope:
            header.pop()
            units.pop()
            
        t_start= float(units[-2])
        t_incr = float(units[-1])
        
        header = header[:-2] # NOTE: This was changed on 09/28/22 you may have to change back to -3
        units = units[:-2]
        
        # Prepare data containers
        t = []
        
        channels = header[1:]
        data = {ch: [] for ch in channels}
        
        n_skipped = 0
        
        # Read out the data
        for idx, row in enumerate(reader):
            bad_line = False
            
            row = [val for val in row if val != '']
            
            try:
                row = [float(val) for val in row]
            except ValueError:
                # Something about this row didn't save correctly. 
                if len(t) == 0:
                    # We're at the beginning of the file. Just move on.
                    continue
                else:
                    # We're at the end of the file. Time to stop.
                    break
                
            t.append(row[0]*t_incr + t_start)
            
            for ch, val in zip(channels, row[1:]):
                data[ch].append(val)
                
        for ch in channels:
            data[ch] = np.array(data[ch])
                
        t = np.array(t)
                
        return t, data
                    
                
if __name__ == "__main__":
    csv_file = r'Z:\Experiments\rydberglab\Squeezing_dds_mw\2021\12\07\000002_uv_noise_traces\0_before_relock\keysight\short.csv'
    t, data = load_keysight_csv(csv_file)