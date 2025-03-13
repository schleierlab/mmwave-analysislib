# -*- coding: utf-8 -*-
"""
@author: Lin Xin
"""

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
import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter.filedialog import askdirectory

from tweezer_waist_analysis import auto_roi_detection, plot_shots_avg, gauss2D, fit_gauss2D, avg_shots_gauss_fit

while True:
    try:
        folder = askdirectory(title='Select multishot folder') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break

file_path = folder + "\\average_first_shot.npy"

data = np.load(file_path)
# data = data[55:75]

neighborhood_size = 10
threshold = 50 #np.max(avg_shot_bkg_sub[0])/2 #40 #48 #83 #60
mag = 300/40.4  # 250/40.4
pix = 5.5  # um
dx = pix/mag

site_roi_x, site_roi_y = auto_roi_detection(data, neighborhood_size, threshold)

plot_shots_avg(data, site_roi_x, site_roi_y, n_shots=1, dx=dx, show_roi=True)

roi_number_lst = []
for i in np.arange(site_roi_x.shape[0]):
    site_roi_signal = data[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
    roi_number_lst.append(np.sum(site_roi_signal))

params_lst = avg_shots_gauss_fit(data, site_roi_x, site_roi_y, plot = False)
amplitude, mux, muy, sigmax, sigmay, rotation, offset = params_lst[:, 0], params_lst[:, 1], params_lst[:, 2], params_lst[:, 3], params_lst[:, 4], params_lst[:, 5], params_lst[:, 6]

print(f" mux = {repr(mux)}, muy = {repr(muy)}, sigma_x = {repr(sigmax)}, sigmay = {repr(sigmay)}, rotation = {repr(rotation)},offset = {repr(offset)}")
waist_x = 2*np.mean(sigmax)*dx
waist_y = 2*np.mean(sigmay)*dx
amplitude = np.abs(amplitude)

waist_x = 2*sigmax*dx
waist_y = 2*sigmay*dx

print(f"waist_x = {np.round(np.average(np.abs(waist_x)),2)}, waist_y = {np.round(np.average(np.abs(waist_y)),2)}")

fig, ax = plt.subplots(constrained_layout=True)

ax.plot(waist_x, label='x waist')
ax.plot(waist_y, label='y waist')
ax.set_xlabel('Trap number')
ax.set_ylabel('waist (um)')
ax.legend()
ax.grid(color='0.7', which='major')
ax.grid(color='0.9', which='minor')

plt.figure()
plt.plot(amplitude, label='amplitude')
plt.xlabel('Trap number')
plt.ylabel('amplitude (counts)')
plt.legend()
plt.grid(color='0.7', which='major')
plt.grid(color='0.9', which='minor')




