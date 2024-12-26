# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:11:12 2023

@author: sslab
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


from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# from analysis.image.process import extractROIDataSingleSequence, getParamArray
# from analysis.data import autolyze as az
import numpy as np
from analysis.data import h5lyze as hz
from matplotlib.collections import PatchCollection

show_site_roi = True #False
load_roi = True
load_threshold = True

folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
if load_threshold == True:
    threshold = np.load(folder_path + "\\th.npy")[0]
    print(f'threshold = {threshold} count')
else:
    threshold = 746.8 #1090
#20240426

if load_roi == True:
    folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
    site_roi_x_file_path = folder_path + "\\site_roi_x.npy"
    site_roi_y_file_path =  folder_path + "\\site_roi_y.npy"
    roi_x_file_path =  folder_path + "\\roi_x.npy"
    roi_x = np.load(roi_x_file_path)
    print('roi_x = ', roi_x)
    # roi_x = [1383, 1699]
    site_roi_x = np.load(site_roi_x_file_path)
    site_roi_y = np.load(site_roi_y_file_path)
    site_roi_x = np.concatenate([[np.min(site_roi_x, axis = 0)], site_roi_x])
    site_roi_y = np.concatenate([[np.min(site_roi_y, axis = 0) + 10], site_roi_y])
else:
    # site_roi_x = np.array([
    #     [1419, 1425],
    #     [1429, 1435],
    #     [1438, 1444],
    #     [1446, 1452],
    #     [1455, 1461],
    #     [1464, 1470],
    #     [1365, 1371],
    #     [1375, 1381],
    #     [1383, 1389],
    #     [1392, 1398],
    #     [1401, 1407],
    #     [1410, 1416],
    #     [1293, 1299],
    #     [1302, 1308],
    #     [1311, 1317],
    #     [1320, 1326],
    #     [1329, 1335],
    #     [1338, 1344],
    #     [1347, 1353],
    #     [1356, 1362]])

    # site_roi_y = np.array([
    #     [34, 40],
    #     [34, 40],
    #     [34, 40],
    #     [34, 40],
    #     [34, 40],
    #     [34, 40],
    #     [35, 41],
    #     [35, 41],
    #     [35, 41],
    #     [35, 41],
    #     [35, 41],
    #     [35, 41],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42],
    #     [36, 42]])

    roi_x = [1173, 1523]

    site_roi_x = np.array([
        [1223, 1466]
    ])

    site_roi_y = np.array([
        [30, 50]
    ])


# Constants
px = 1 #6.5 # Pixels size
mag = 1 #18.56 #7.424 #12.37 # Magnefication
counts_per_atom = 1 # Counts per atom 16.6 counts per atom per ms
#roi_x = [1000, 1800] #roi_x = [750, 1250]#roi_x = [850, 1250] # Region of interest of X direction, MOT beam imaging
#roi_y = [800, 1500] #roi_y = [800, 1000] #[750, 1150] # Region of interest of Y direction, MOT beam imaging

#for dipole trap (20240109):
# roi_x = [800,1750]#[750, 1200]#roi_x = [850, 1250] # Region of interest of X direction, Img beam imaging
# roi_y = [1150,1300]#[1500, 2000] #[750, 1150] # Region of interest of Y direction, Img beam imaging


# roi_x = np.array([1383, 1699])
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction
para_name = 'manta_exposure'



# Is this script being run from within an interactive lyse session?
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

    try:
        loop_var = float(f['globals'].attrs.get(loop_glob))
        print(f'The global that is looping through is {loop_glob} and currently = {loop_var}')
    except:
        info_dict = hz.getAttributeDict(f)
        loop_var = info_dict.get('run number')
        print('Nothing is under loop')

    info_dict = hz.getAttributeDict(f)
    # images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
    # para = float(hz.attributesToDictionary(f).get('globals').get(para_name))
    try:
        kinetix_roi_row= np.array(hz.attributesToDictionary(f).get('globals').get('kinetix_roi_row'))
    except:
        kinetix_roi_row = np.array([1000,110])

    print(kinetix_roi_row)
    run_number = info_dict.get('run number')
    target_array = list(eval(f['/globals/Tweezers'].attrs['TW_target_array']))
    #print(f'run_number = {run_number} ')





avg_shot_bkg_file_path =  Path(Path(h5_path).parent.parent, 'avg_shot_bkg.npy')#folder_path + "\\avg_shot_bkg.npy"
avg_shot_bkg = np.load(avg_shot_bkg_file_path)
first_image_bkg = avg_shot_bkg[0,:,:]
second_image_bkg = avg_shot_bkg[1,:,:]

# site_roi_y = site_roi_y - kinetix_roi_row[0] - 1
site_roi_x = site_roi_x - roi_x[0]


rect = []
for i in np.arange(site_roi_x.shape[0]):
    rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))



# print(h5_path)
# print(h5_path.split(".")[0]+"_no_image."+h5_path.split(".")[1])

# original = open(h5_path, 'rb')
# copy = open(h5_path.split(".")[0]+"_no_image."+h5_path.split(".")[1], "wb")
# copy.write(original.read())

image_types = list(images.keys())
# print(images[image_types[0]])

# Defining the pixel size (um) and imaging from the magnification
# Also the total fov (in um) given the chip size in pixels
pixels = images[image_types[0]].shape[0]
fov = pixels*px*mag


# The MOT image and the backgrounds image are saved into the h5 according to the run file
# Image with MOT from the even shots and image of background from odd shots

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
# count_file_path = folder_path+'\\data.csv'
first_image_file_path = folder_path+'\\first'
second_image_file_path = folder_path+'\\seconds'

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
# count_file_path = folder_path+'\\data.csv'
count_file_path = folder_path+'\\data.csv' #'\\fault_rate_data.csv'



if run_number % 1 == 0: # all run number case
    first_image = images[image_types[0]]
    second_image = images[image_types[1]]

    np.save(first_image_file_path, first_image)
    np.save(second_image_file_path, second_image)

    if run_number == 0:
        with open(count_file_path, 'w') as f_object:
            f_object.write('')

    else:
        with open(count_file_path, 'a') as f_object:
            f_object.write('')

    try:
        sub_image1 = first_image - first_image_bkg # subtraction of the background
        sub_image2 = second_image - second_image_bkg
    except:
        print('Make sure you already have the averaged background!')
        quit()

    # We subtract the background from the MOT only in the ROI
    roi_MOT_1 = sub_image1[:, roi_x[0]:roi_x[1]]
    roi_bkg_1 = sub_image1[:, roi_x_bkg[0]:roi_x_bkg[1]]

    roi_MOT_2 = sub_image2[:, roi_x[0]:roi_x[1]]
    roi_bkg_2 = sub_image2[:, roi_x_bkg[0]:roi_x_bkg[1]]


    atom_number = 0 #roi_MOT_1.sum()-roi_bkg_1.sum()/roi_bkg_1.size * roi_MOT_1.size


    rect_sig_1 = []
    atom_exist_lst_1 = []
    roi_number_lst_1 = []
    for i in np.arange(site_roi_x.shape[0]):
        site_roi_signal = roi_MOT_1[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
        roi_number_lst_1.append(np.sum(site_roi_signal))
        atom_number += site_roi_signal.sum()
        if site_roi_signal.sum() > threshold:
            rect_sig_1.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='gold', facecolor='none'))
            atom_exist_lst_1.append(1)
        else:
            atom_exist_lst_1.append(0)

    #print(atom_exist_lst_1)

    rect_sig_2 = []
    atom_exist_lst_2 = []
    roi_number_lst_2 = []
    for i in np.arange(site_roi_x.shape[0]):
        site_roi_signal = roi_MOT_2[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
        roi_number_lst_2.append(np.sum(site_roi_signal))
        if site_roi_signal.sum() > threshold:
            rect_sig_2.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='gold', facecolor='none'))
            atom_exist_lst_2.append(1)
        else:
            atom_exist_lst_2.append(0)

    #print(atom_exist_lst_2)


    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    (ax_mot_roi_1, ax_bkg_roi_1), (ax_mot_roi_2, ax_bkg_roi_2) = axs

    fig.suptitle('Mag = 7.424, 12.37, 18.56, Pixel = 6.5 um')

    for ax in axs[0]:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
    for ax in axs[1]:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    roi_image_scale = 150#180 #180 #100 #500 #1000 #180 #150 #2000 #4096 #150
    roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

    ax_mot_roi_1.set_title('1st roi')
    pos = ax_mot_roi_1.imshow(roi_MOT_1, **roi_img_color_kw)
    if show_site_roi:
        ax_mot_roi_1.add_collection(PatchCollection(rect, match_original=True))
        ax_mot_roi_1.add_collection(PatchCollection(rect_sig_1, match_original=True))
    fig.colorbar(pos, ax=ax_mot_roi_1)

    ax_bkg_roi_1.set_title('1st bkg roi')
    pos = ax_bkg_roi_1.imshow(roi_bkg_1, **roi_img_color_kw)
    fig.colorbar(pos, ax=ax_bkg_roi_1)

    ax_mot_roi_2.set_title('2nd roi')
    pos = ax_mot_roi_2.imshow(roi_MOT_2, **roi_img_color_kw)
    if show_site_roi:
        ax_mot_roi_2.add_collection(PatchCollection(rect, match_original=True))
        ax_mot_roi_2.add_collection(PatchCollection(rect_sig_2, match_original=True))
    fig.colorbar(pos, ax=ax_mot_roi_2)

    ax_bkg_roi_2.set_title('2nd bkg roi')
    pos = ax_bkg_roi_2.imshow(roi_bkg_2, **roi_img_color_kw)
    fig.colorbar(pos, ax=ax_bkg_roi_2)

    atom_exist_lst_1 = np.array(atom_exist_lst_1)
    atom_exist_lst_2 = np.array(atom_exist_lst_2)


    # fault_rate = sum(np.bitwise_xor(atom_exist_lst_1,atom_exist_lst_2)) / np.sum(atom_exist_lst_1) #site_roi_x.shape[0]
    survival_rate = sum(1 for x,y in zip(atom_exist_lst_1,atom_exist_lst_2) if x == 1 and y == 1) / np.sum(atom_exist_lst_1)

    roi_number_lst_file_path = folder_path+'\\roi_number_lst.npy' #'\\fault_rate_data.csv'

    roi_number_lst = np.row_stack([np.array(roi_number_lst_1), np.array(roi_number_lst_2)])

    roi_number_lst = roi_number_lst.reshape((roi_number_lst.shape[0], roi_number_lst.shape[1],1))

    # print(f"roi_number_lst shape = {roi_number_lst.shape}")
    # create for temproray purpose
    # survival_rate = atom_number

    print("From original code", site_roi_x, site_roi_y)

    if run_number == 0:
        with open(count_file_path, 'w') as f_object:
            f_object.write(f'{survival_rate},{loop_var}\n')
        np.save(roi_number_lst_file_path, roi_number_lst)

    else:
        with open(count_file_path, 'a') as f_object:
            f_object.write(f'{survival_rate},{loop_var}\n')
        roi_number_lst_old = np.load(roi_number_lst_file_path)
        roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
        # print(f"roi_number_lst_new shape = {roi_number_lst_new.shape}")
        np.save(roi_number_lst_file_path, roi_number_lst_new)

    # os.remove(h5_path)







