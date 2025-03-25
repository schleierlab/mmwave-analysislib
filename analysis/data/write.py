# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:38:50 2020

@author: Jacob
"""

import csv
import os

def writeToCSV(filepath, writemode, rowstring):
    # writemodes: 'wb' rewrites, 'a' appends
    # rowstring: list of strings, e.g., ['col1text,'col2text', 'col3text']
    with open(filepath, writemode) as paramfile:
        csv_writer = csv.writer(paramfile, delimiter=',')
        csv_writer.writerow(rowstring)
        
def array_to_csv(data, csv_file, header=None, units=None, delimiter=',', newline=''):
    assert data.ndim == 2, 'Can only write a two dimensional array to .csv'
    assert csv_file.endswith('.csv'), 'File does not end with .csv'
    
    n_columns = len(data[0])

    with open(csv_file, 'w', newline=newline) as f:
        writer = csv.writer(f, delimiter=delimiter)
        
        if header is not None:
            assert len(header) == n_columns, 'Number of header entries must match number of columns in array'
            writer.writerow(header)
            
        if units is not None:
            assert len(units) == n_columns, 'Number of units entries must match number of columns in array'
            writer.writerow(units)
        
        for row in data:
            writer.writerow(row)            