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

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    for group in g:
        for glob in g[group]:
            if g[group][glob][0:2] == "np":
                loop_glob = glob
                # print(repr(glob))
                # print(g[group][glob][:])
                loop_var = np.size(eval(g[group][glob][:]))
                print(f"group = {str(group)}, glob = {str(glob)}")
                unit = hz.attributesToDictionary(f['globals'][str(group)])['units'][str(glob)]
            if glob == 'n_shots':
                loop_var_bk = np.size(eval('np.'+g[group][glob][:]))
                unit_bk = 'shots'

    try:
        print(f"number of detunings = {loop_var}")
    except:
        loop_var = loop_var_bk
        loop_glob = 'n_shot'
        unit = unit_bk
        print(f"number of detunings = {loop_var}")

    counts = []
    var = []


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
        var.append(list(map(float, row))[3])



collected_counts = [[] for i in range(loop_var)]
collected_counts_2 = [[] for i in range(loop_var)]
collected_counts_3 = [[] for i in range(loop_var)]
collected_vars = [[] for i in range(loop_var)]


i=0
for c, c2, c3,  v in zip(temperature_x, temperature_y, counts, var):
    collected_counts[i%loop_var].append(c)
    collected_counts_2[i%loop_var].append(c2)
    collected_counts_3[i%loop_var].append(c3)
    collected_vars[i%loop_var].append(v)
    i+=1

avg_counts = [np.mean(collected_counts[i]) for i in range(loop_var)]
avg_counts_2 = [np.mean(collected_counts_2[i]) for i in range(loop_var)]
avg_counts_3 = [np.mean(collected_counts_3[i]) for i in range(loop_var)]
avg_vars = [np.mean(collected_vars[i]) for i in range(loop_var)]

std_counts = [np.std(collected_counts[i]) for i in range(loop_var)]
std_counts_2 = [np.std(collected_counts_2[i]) for i in range(loop_var)]
std_counts_3 = [np.std(collected_counts_3[i]) for i in range(loop_var)]

fig, axs = plt.subplots(nrows=2, ncols=1, layout="constrained")
(ax_count),(ax_temp) = axs

# ax_count.plot(tof , counts)
ax_count.errorbar(avg_vars,np.array(avg_counts_3), yerr = np.array(std_counts_3))
# ax_count.set_xlabel('Time of flight (s)')
ax_count.set_ylabel('MOT atom count')

ax_count.grid(color='0.7', which='major')
ax_count.grid(color='0.9', which='minor')

# ax_temp.plot(tof , np.array(temperature_x)*1e6, label='X temperature')
# ax_temp.plot(tof , np.array(temperature_y)*1e6, label='Y temperature')
ax_temp.errorbar(avg_vars,np.array(avg_counts)*1e6, yerr = np.array(std_counts)*1e6, label='X temperature')
ax_temp.errorbar(avg_vars,np.array(avg_counts_2)*1e6, yerr = np.array(std_counts_2)*1e6, label='Y temperature')
# ax_temp.set_xlabel('Time of flight (s)')
ax_temp.set_ylabel('Temperature (uK)')
ax_temp.set_xlabel(f'{loop_glob} ({unit})')
ax_count.legend()
ax_temp.legend()

ax_temp.grid(color='0.7', which='major')
ax_temp.grid(color='0.9', which='minor')
