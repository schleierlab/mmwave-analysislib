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

roi_x = [900,1300]
roi_y = [1100,1200]
roi_x_bkg = [2000, 2400]
roi_y_bkg= [2000, 2400]

avg_shot, N = avg_all_shots(folder)

MOT_image = avg_shot[0]
background_image = avg_shot[1]
sub_image = MOT_image - background_image # subtraction of the background
roi_MOT = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
roi_MOT[roi_MOT<0] = 0
roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]
roi_bkg[roi_bkg<0] = 0



folder = Path(folder)
folder_path = folder #'X:\\userlib\\analysislib\\scripts\\multishot\\'



fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs

fig.suptitle(f'Avg of {N} shots')

for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

image_scale = 4096 #2000 #4096 # 12 bit depth
raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

ax_mot_raw.set_title('Raw')
pos = ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_mot_raw)

ax_bkg_raw.set_title('Raw, no MOT')
pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_bkg_raw)

roi_image_scale = 100
roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

ax_mot_roi.set_title('MOT ROI')
pos = ax_mot_roi.imshow(
    roi_MOT,
    extent=np.array([roi_x[0], roi_x[1], roi_y[1], roi_y[0]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)

ax_bkg_roi.set_title('Background ROI')
pos = ax_bkg_roi.imshow(
    roi_bkg,
    vmin=-100,
    vmax=100,
    extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y[0]+roi_y[1], roi_y[0]]),
)
fig.colorbar(pos, ax=ax_bkg_roi)
fig.savefig(folder_path)
fig.savefig('average_shots.png')


root = Tk()
root.destroy()



