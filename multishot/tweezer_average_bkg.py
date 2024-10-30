# -*- coding: utf-8 -*-
"""
@author: Lin Xin
"""
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
# from analysis.image.process import extractROIDataSingleSequence, getParamArray
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import os
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import glob
from pathlib import Path

from tkinter import Tk
from tkinter.filedialog import askdirectory


def avg_all_shots(folder, shots = 'defult', loop = True):
    n_shots = np.size([i for i in os.listdir(folder) if i.endswith('.h5')])

    for cnt in (np.arange(n_shots)):
        string = glob.glob(folder + f'\*{cnt}.h5')
        h5_path = string[0]
        #h5_path = folder + rf'{cnt:01}.h5'
        # print(h5_path)
        with h5py.File(h5_path, mode='r+') as f:
            # g = hz.attributesToDictionary(f['globals'])
            # info_dict = hz.getAttributeDict(f)
            images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)

            # run_number = info_dict.get('run number')
        image_types = list(images.keys())
        if cnt == 0:
            print(f'size of the image is {np.shape(images[image_types[0]])}')
            data = np.zeros((2, ) + np.shape(images[image_types[0]]))
        try:
            data[0] = data[0]+images[image_types[0]]
            data[1] = data[1]+images[image_types[1]]
        except:
            sys.exit('The data is not created, start from the first shot')

    N = n_shots

    return (data/N, N)


while True:
    try:
        folder = askdirectory(title='Select Folder for averaging the tweezer images') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break



avg_shot_bkg, N = avg_all_shots(folder)


folder = Path(folder)
folder_path = folder.parent #'X:\\userlib\\analysislib\\scripts\\multishot\\'

avg_shot_bkg_file_path =  Path(folder_path, 'avg_shot_bkg.npy') #folder_path + "\\avg_shot_bkg.npy"

print(f'avg_shot_bkg_file_path={repr(avg_shot_bkg_file_path)}')



np.save(avg_shot_bkg_file_path, avg_shot_bkg)


root = Tk()
root.destroy()




