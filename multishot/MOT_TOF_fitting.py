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
# from uncertainties import ufloat

if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

m = 2.20694650e-25 # kg mass of cesium
kB = 1.3806503e-23 # J/K Boltzman constant

folder_path = '\\'.join(h5_path.split('\\')[0:-1])

counts = []
tof = []
sigma_a = []
sigma_b = []

tof_fit_file_path = folder_path+'\\TOFdata.csv'

with open(tof_fit_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in csv.reader(csvfile , delimiter=','):
        counts.append(list(map(float, row))[0])
        tof.append(list(map(float, row))[1])
        sigma_b.append(list(map(float, row))[2])
        sigma_a.append(list(map(float, row))[3])

t = np.array(tof)

# TODO make uncertainties bit work

# coefficient_a, cov_a = np.polyfit(t**2, np.array(sigma_a)**2, 1, cov = True)
# uncertain_a = np.sqrt(np.diag(cov_a))
# [slope_a, intercept_a] = coefficient_a

# slope_a_uncertain = ufloat(slope_a, uncertain_a[0])
# coefficient_b, cov_b = np.polyfit(t**2, np.array(sigma_b)**2, 1, cov = True)
# [slope_b, intercept_b] = coefficient_b
# uncertain_b = np.sqrt(np.diag(cov_b))
# slope_b_uncertain = ufloat(slope_b, uncertain_b[0])
# slope_uncertain = np.array([slope_a_uncertain, slope_b_uncertain])
# T = slope_uncertain*m/kB*1e6

slope_a, intercept_a = np.polyfit(t**2, np.array(sigma_a)**2, 1)
slope_b, intercept_b = np.polyfit(t**2, np.array(sigma_b)**2, 1)
slope = np.array([slope_a, slope_b])
T = slope*m/kB*1e6

fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
(ax_count), (ax_tof_fit) = axs

ax_count.plot(t*1e3 , counts,'-o')
ax_count.set_xlabel('Time of flight (ms)')
ax_count.set_ylabel('MOT atom count')
#ax_count.set_ylim(ymin=0, ymax=max(counts)+1e5)

ax_count.grid(color='0.7', which='major')
ax_count.grid(color='0.9', which='minor')

t_plot = np.arange(0,max(t)+1e-3,1e-3)


ax_tof_fit.plot(t_plot**2*1e6 , (slope_a*t_plot**2 + intercept_a)*1e12, label=rf'major axis fit, $T = {T[0]:.3f}$ $\mu$K')
ax_tof_fit.plot(t_plot**2*1e6 , (slope_b*t_plot**2 + intercept_b)*1e12, label=rf'minor axis fit, $T = {T[1]:.3f}$ $\mu$K')
# ax_tof_fit.plot(t_plot**2*1e6 , (slope_a*t_plot**2 + intercept_a)*1e12, label=rf'major axis fit, $T = {T[0]:SL}$ $\mu$K')
# ax_tof_fit.plot(t_plot**2*1e6 , (slope_b*t_plot**2 + intercept_b)*1e12, label=rf'minor axis fit, $T = {T[1]:SL}$ $\mu$K')

ax_tof_fit.plot(t**2*1e6, np.array(sigma_a)**2*1e12, 'oC0')
ax_tof_fit.plot(t**2*1e6, np.array(sigma_b)**2*1e12, 'oC1')
ax_tof_fit.set_xlabel(r'Time$^2$ (ms$^2$)')
ax_tof_fit.set_ylabel(r'$\sigma^2$ ($\mu$m$^2$)')
ax_tof_fit.legend()
ax_tof_fit.set_ylim(ymin=0)
ax_tof_fit.tick_params(axis = 'x')
ax_tof_fit.tick_params(axis = 'y')

ax_tof_fit.grid(color='0.7', which='major')
ax_tof_fit.grid(color='0.9', which='minor')
