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
    number_of_detunings = np.size(eval(g['Optical pumping, Microwaves']['uwave_time']))

counts = []
counts_2 = []
var = []



with open(count_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in csv.reader(csvfile , delimiter=','):
        print(row)
        counts.append(list(map(float, row))[0])
        counts_2.append(list(map(float, row))[1])
        var.append(list(map(float, row))[2])

fig, ax = plt.subplots(constrained_layout=True)

counts = np.array(counts)
counts_2 = np.array(counts_2)


total_counts = counts + counts_2
normalized_counts = counts / total_counts
normalized_counts_2 = counts_2 / total_counts

# number_of_detunings = 9
collected_counts = [[] for i in range(number_of_detunings)]
collected_counts_2 = [[] for i in range(number_of_detunings)]
collected_vars = [[] for i in range(number_of_detunings)]
i=0
for c,c2,v in zip(normalized_counts, normalized_counts_2, var):
    collected_counts[i%number_of_detunings].append(c)
    collected_counts_2[i%number_of_detunings].append(c2)
    collected_vars[i%number_of_detunings].append(v)
    i+=1

avg_counts = [np.mean(collected_counts[i]) for i in range(number_of_detunings)]
avg_counts_2 = [np.mean(collected_counts_2[i]) for i in range(number_of_detunings)]
avg_vars = [np.mean(collected_vars[i]) for i in range(number_of_detunings)]

std_counts = [np.std(collected_counts[i]) for i in range(number_of_detunings)]
std_counts_2 = [np.std(collected_counts_2[i]) for i in range(number_of_detunings)]

print(avg_counts, avg_counts_2, avg_vars)



# ax.plot(avg_vars, avg_counts)
ax.errorbar(avg_vars, avg_counts, yerr=std_counts, fmt='-o', label = 'F=4 atom')
ax.errorbar(avg_vars, avg_counts_2, yerr=std_counts, fmt='-s', label = 'F=3 atom')
ax.set_xlabel('time (s)')
ax.set_ylabel('Gaussian Peak (counts)')
ax.legend()

ax.grid(color='0.7', which='major')
ax.grid(color='0.9', which='minor')
