# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:48:37 2023

@author: sslab
"""
import csv

import sys
root_path = r"X:\userlib\analysislib"
#root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"


if root_path not in sys.path:
    sys.path.append(root_path)

try:
    lyse
except:
    import lyse

from analysis.data import h5lyze as hz
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt

if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'

counts = []
with open(count_file_path, newline='') as csvfile:
    counts = [list(map(float, row))[0] for row in csv.reader(csvfile)]

fig, ax = plt.subplots(constrained_layout=True)

ax.plot(counts)
ax.set_xlabel('Shot number')
ax.set_ylabel('MOT atom count')

ax.grid(color='0.7', which='major')
ax.grid(color='0.9', which='minor')