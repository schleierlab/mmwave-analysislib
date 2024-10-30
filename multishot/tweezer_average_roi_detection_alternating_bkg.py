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

from tkinter import Tk
from tkinter.filedialog import askdirectory


def avg_all_shots(folder, shots = 'defult', loop = True):
    n_shots = np.size([i for i in os.listdir(folder) if i.endswith('.h5')])

    if shots == 'defult':
        shots = n_shots
    # roi_number_lst = np.zeros([site_roi_x.shape[0], shots])
    # print(np.shape(roi_number_lst))

    for cnt in (np.arange(shots)):
        if loop == True:
            if cnt % 2 == 0:
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
                    bkg = np.zeros((2, ) + np.shape(images[image_types[0]]))
                # print(image_types)
                # image = np.array((images[image_types[0]], images[image_types[1]]))
                # print(np.shape(image))
                try:
                    data[0] = data[0]+images[image_types[0]]
                    data[1] = data[1]+images[image_types[1]]
                except:
                    sys.exit('The data is not created, start from the first shot')

            else:
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
                # print(image_types)
                # image = np.array((images[image_types[0]], images[image_types[1]]))
                # print(np.shape(image))
                try:
                    bkg[0] = bkg[0]+images[image_types[0]]
                    bkg[1] = bkg[1]+images[image_types[1]]
                except:
                    sys.exit('The bkg is not created, start from the first shot')
        # print(cnt)

    if loop == True:
        N = shots/2
    else:
        N = shots

    return ((data-bkg)/N, bkg/N, N)


def auto_roi_detection(data, neighborhood_size, threshold):
    #choose even number to make the roi centered
    import scipy.ndimage.filters as filters
    import scipy.ndimage as ndimage
    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    # print(slices)
    x, y = [], []
    site_roi_x, site_roi_y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)
        site_roi_x.append([int(x_center-neighborhood_size/2), int(x_center+neighborhood_size/2)])
        site_roi_y.append([int(y_center-neighborhood_size/2), int(y_center+neighborhood_size/2)])

    site_roi_x = np.array(site_roi_x)
    site_roi_y = np.array(site_roi_y)
    roi_x = np.array([np.min(site_roi_x)-50, np.max(site_roi_x)+50])
    return site_roi_x, site_roi_y, roi_x

def plot_shots_avg(data, site_roi_x,site_roi_y, n_shots =2, show_roi = True):
    # roi_x = np.array([np.min(site_roi_x)-10, np.max(site_roi_x)+10])
    roi_x = np.array([np.min(site_roi_x)-50, np.max(site_roi_x)+50])

    print(f'roi_x = {repr(roi_x)}')
    # roi_x = [1275,1475] #[1250, 1450]
    site_roi_x = site_roi_x - roi_x[0]

    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))
    fig.suptitle(f'{n_shots} shots average, Mag = 7.424, Pixel = 0.87 um')
    for ax in axs:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    (ax_first_image, ax_second_image) = axs

    image_scale = np.amax(data[0, :, roi_x[0]:roi_x[1]]) # 12 bit dept
    raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

    ax_first_image.set_title('first shot')
    pos = ax_first_image.imshow(data[0,:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
    #fig.colorbar(pos, ax=ax_first_image)

    ax_second_image.set_title('second shot')
    pos = ax_second_image.imshow(data[1,:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
    fig.colorbar(pos, ax=ax_second_image)


    if show_roi == True:
        ax_first_image.add_collection(PatchCollection(rect, match_original=True))
        ax_second_image.add_collection(PatchCollection(rect, match_original=True))



while True:
    try:
        folder = askdirectory(title='Select Folder for averaging the tweezer images') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break



avg_shot_bkg_sub, avg_shot_bkg, N = avg_all_shots(folder)
neighborhood_size = 6
threshold = 30 #np.max(avg_shot_bkg_sub[0])/2 #40 #48 #83 #60
site_roi_x, site_roi_y, roi_x = auto_roi_detection(avg_shot_bkg_sub[0], neighborhood_size, threshold)

print(f'site_roi_x={repr(site_roi_x)}, site_roi_y={repr(site_roi_y)}')

print(f'size of site roi = {site_roi_x.shape[0]} ')
plot_shots_avg(avg_shot_bkg_sub, site_roi_x,site_roi_y, N, show_roi=True)

folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
site_roi_x_file_path = folder_path + "\\site_roi_x.npy"
site_roi_y_file_path =  folder_path + "\\site_roi_y.npy"
roi_x_file_path = folder_path + "\\roi_x.npy"
avg_shot_bkg_file_path =  folder_path + "\\avg_shot_bkg.npy"


np.save(site_roi_x_file_path, site_roi_x)
np.save(site_roi_y_file_path, site_roi_y)
np.save(roi_x_file_path, roi_x)
np.save(avg_shot_bkg_file_path, avg_shot_bkg)


root = Tk()
root.destroy()




