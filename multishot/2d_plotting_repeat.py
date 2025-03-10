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

df = lyse.data()
h5_path = df.filepath.iloc[-1]

# print( lyse.data().filepath.iloc[-1])
folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'



with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    loop_glob = []
    i = 0
    for group in g:
        for glob in g[group]:
            if g[group][glob][0:2] == "np":
                loop_glob.append(glob)
                if i == 0:
                    set_var_1 = eval(g[group][glob][:])
                    set_var_1 = np.unique(set_var_1)
                    i+=1
                elif i == 1:
                    set_var_2 = eval(g[group][glob][:])
                    set_var_2 = np.unique(set_var_2)
                # print(g[group][glob][:])

counts = []
var_1 = []
var_2 = []


try:
    with open(count_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in csv.reader(csvfile , delimiter=','):
            # print(row)
            counts.append(list(map(float, row))[0])
            var_1.append(list(map(float, row))[1])
            var_2.append(list(map(float, row))[2])

    fig, axs= plt.subplots(ncols=2, constrained_layout=True)
    (ax1, ax2) = axs

    # number_of_detunings = 9
    avg_counts = np.zeros((set_var_1.shape[0],set_var_2.shape[0]))
    std_counts = np.zeros((set_var_1.shape[0],set_var_2.shape[0]))

    # print(f"set_var_1 = {set_var_1}")
    # print(f"set_var_2 = {set_var_2}")
    # print(f"var_1 = {var_1}")
    # print(f"var_2 = {var_2}")
    # print(f"counts = {counts}")

    var_1 = np.array(var_1)
    var_2 = np.array(var_2)

    for i in np.arange(set_var_1.shape[0]):
        for j in np.arange(set_var_2.shape[0]):
            loc = np.where((var_1 == set_var_1[i]) & (var_2 == set_var_2[j]))[0]
            # print("loc = ", loc)
            if loc.shape[0] != 0:
                # print(f"loc = ", loc)
                # print(np.asarray(counts).shape)
                new_counts = np.asarray(counts)[loc]
                avg_counts[i,j] = np.mean(new_counts)
                std_counts[i,j] = np.std(new_counts)

    # print(avg_counts, std_counts)



    # ax.plot(avg_vars, avg_counts)
    # ax.errorbar(avg_vars, avg_counts, yerr=std_counts, fmt='-o')

    avg_counts = np.array(avg_counts)
    std_counts = np.array(std_counts)

    # print(var_1)
    # print(var_2)
    # plot_extent = np.array([np.min(var_2)-0.5, np.max(var_2)+0.5, np.min(var_1)-0.5, np.max(var_1)+0.5])
    plot_extent = np.array([np.min(var_2), np.max(var_2), np.min(var_1), np.max(var_1)])
    neg = ax1.imshow(avg_counts, extent=plot_extent, interpolation='none', aspect='auto', origin='lower')
    fig.colorbar(neg, ax = ax1)
    ax1.set_title("mean")

    neg = ax2.imshow(std_counts, extent=plot_extent, interpolation='none', aspect='auto', origin='lower')
    fig.colorbar(neg, ax = ax2)
    ax2.set_title("std")

    # ax.set_xlabel('time (s)')
    for  ax in axs:
        ax.set_xlabel(loop_glob[1])
        # ax.set_ylabel('Gaussian Peak (counts)')
        ax.set_ylabel(loop_glob[0])

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
except OSError:
    print("Could not open/read file:")









