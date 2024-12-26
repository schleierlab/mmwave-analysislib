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

import h5py
import matplotlib.pyplot as plt

# from analysis.data import autolyze as az
import numpy as np
from analysis.data import h5lyze as hz

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
print(count_file_path, h5_path)


try:
    with h5py.File(h5_path, mode='r+') as f:
        g = hz.attributesToDictionary(f['globals'])
        for group in g:
            for glob in g[group]:
                if g[group][glob][0:2] == "np":
                    loop_glob = glob
                    # print(repr(glob))
                    # print(g[group][glob][:])
                    number_of_detunings = np.size(eval(g[group][glob][:]))
                    print(f"group = {str(group)}, glob = {str(glob)}")
                    unit = hz.attributesToDictionary(f['globals'][str(group)])['units'][str(glob)]
                if glob == 'n_shot':
                    number_of_detunings_bk = np.size(eval('np.'+g[group][glob][:]))
                    unit_bk = 'shots'

    try:
        print(f"number of detunings = {number_of_detunings}")
    except:
        number_of_detunings = number_of_detunings_bk
        loop_glob = 'n_shot'
        unit = unit_bk
        print(f"number of detunings = {number_of_detunings}")

    counts = []
    var = []



    with open(count_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in csv.reader(csvfile , delimiter=','):
            # print(row)
            counts.append(list(map(float, row))[0])
            var.append(list(map(float, row))[1])

    fig, ax = plt.subplots(constrained_layout=True)

    # number_of_detunings = 9
    collected_counts = [[] for i in range(number_of_detunings)]
    collected_vars = [[] for i in range(number_of_detunings)]
    i=0
    for c,v in zip(counts,var):
        collected_counts[i%number_of_detunings].append(c)
        collected_vars[i%number_of_detunings].append(v)
        i+=1

    avg_counts = [np.mean(collected_counts[i]) for i in range(number_of_detunings)]
    avg_vars = [np.mean(collected_vars[i]) for i in range(number_of_detunings)]

    std_counts = [np.std(collected_counts[i]) for i in range(number_of_detunings)]

    # print(avg_counts, avg_vars)



    # ax.plot(avg_vars, avg_counts)
    ax.errorbar(avg_vars, avg_counts, yerr=std_counts, fmt='-o')
    # ax.set_xlabel('time (s)')
    ax.set_xlabel(f'{loop_glob} ({unit})')
    # ax.set_ylabel('Gaussian Peak (counts)')
    ax.set_ylabel('Survival rate')
    # ax.set_ylabel('survival rate')

    ax.grid(color='0.7', which='major')
    ax.grid(color='0.9', which='minor')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    #fig.savefig(folder_path + '\data.png')

except:
    print("no data.csv found")
