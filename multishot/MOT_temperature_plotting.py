# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:48:37 2023

@author: sslab
"""
import csv


import sys
root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"


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

temp_file_path = folder_path+'\\temp.csv'

counts = []
with open(count_file_path, newline='') as csvfile:
    counts = [list(map(float, row))[0] for row in csv.reader(csvfile)]

temperature_x = []
temperature_y = []
tof = []

with open(temp_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in csv.reader(csvfile , delimiter=','):
        tof.append(list(map(float, row))[0])
        temperature_x.append(list(map(float, row))[1])
        temperature_y.append(list(map(float, row))[2])



fig, axs = plt.subplots(nrows=2, ncols=1)
(ax_count),(ax_temp) = axs
fig, ax = plt.subplots(constrained_layout=True)

ax_count.plot(tof , counts)
ax_count.set_xlabel('Time of flight (s)')
ax_count.set_ylabel('MOT atom count')

ax_count.grid(color='0.7', which='major')
ax_count.grid(color='0.9', which='minor')

ax_temp.plot(tof , np.array(temperature_x)*1e6, label='X temperature')
ax_temp.plot(tof , np.array(temperature_y)*1e6, label='Y temperature')
ax_temp.set_xlabel('Time of flight (s)')
ax_temp.set_ylabel('Temperature (uK)')
ax.legend()

ax_temp.grid(color='0.7', which='major')
ax_temp.grid(color='0.9', which='minor')
