# -*- coding: utf-8 -*-
"""
@author: Lin Xin
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
count_file_path = folder_path+'\\tweezer_waist.csv'

waist_x = []
waist_y = []
with open(count_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in csv.reader(csvfile, delimiter=','):
        waist_x.append(list(map(float, row))[0])
        waist_y.append(list(map(float, row))[1])


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\tweezer_amplitude.csv'
roi_number_lst_path = folder_path+'\\roi_number_lst.csv'

amplitude = np.loadtxt(count_file_path)
roi_number_lst = np.loadtxt(roi_number_lst_path)


count_file_path = folder_path+'\\site_roi_x.csv'

site_roi_x = np.loadtxt(count_file_path)






fig, ax = plt.subplots(constrained_layout=True)

ax.plot(waist_x, label='x waist')
ax.plot(waist_y, label='y waist')
ax.set_xlabel('Shot')
ax.set_ylabel('waist (um)')
ax.legend()
ax.grid(color='0.7', which='major')
ax.grid(color='0.9', which='minor')


# fig1, ax1 = plt.subplots(constrained_layout=True)
# print(np.shape(site_roi_x))
# print(site_roi_x[:,0])
# ax1.plot(site_roi_x[:,0], amplitude, 'o', label='amplitude')
# ax1.set_xlabel('site roi (px)')
# ax1.set_ylabel('amplitude (counts)')
# ax1.legend()
# ax1.grid(color='0.7', which='major')
# ax1.grid(color='0.9', which='minor')


# fig2, ax2 = plt.subplots(constrained_layout=True)
# ax2.plot(site_roi_x[:,0], roi_number_lst, 'o', label='sum')
# ax2.set_xlabel('site roi (px)')
# ax2.set_ylabel('sum in roi (counts)')
# ax2.legend()
# ax2.grid(color='0.7', which='major')
# ax2.grid(color='0.9', which='minor')