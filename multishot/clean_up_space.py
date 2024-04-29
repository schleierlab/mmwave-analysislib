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

# if lyse.spinning_top:
#     # If so, use the filepath of the current h5_path
#     h5_path = lyse.path
#     print('hi')
# else:
#     # If not, get the filepath of the last h5_path of the lyse DataFrame
#     df = lyse.data()
#     print(df)
#     h5_path = df.filepath.iloc[-1]

df = lyse.data()
h5_path = df.filepath.iloc[-1]

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    info_dict = hz.getAttributeDict(f)

print(h5_path)
folder_path = '\\'.join(h5_path.split('\\')[0:-1])
copy_path = h5_path.split(".")[0]+"_no_image."+h5_path.split(".")[1]

original = open(h5_path, 'rb')

print(copy_path)

print(g['Imaging']['save_image'])
if (g['Imaging']['save_image']=='False'):
    print('delete')
    del f['kinetix_images']
    copy = open(copy_path, "wb")
    copy.write(original.read())
    os.remove(h5_path)