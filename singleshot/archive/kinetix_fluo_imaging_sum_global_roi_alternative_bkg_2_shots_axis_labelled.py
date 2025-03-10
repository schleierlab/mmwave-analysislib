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

# Constants
px = 1 #6.5 # Pixels size
mag = 1 #18.56 #7.424 #12.37 # Magnefication
counts_per_atom = 1 # Counts per atom 16.6 counts per atom per ms
#roi_x = [1000, 1800] #roi_x = [750, 1250]#roi_x = [850, 1250] # Region of interest of X direction, MOT beam imaging
#roi_y = [800, 1500] #roi_y = [800, 1000] #[750, 1150] # Region of interest of Y direction, MOT beam imaging

#for dipole trap (20240109):
#roi_x = [800,1750]#[750, 1200]#roi_x = [850, 1250] # Region of interest of X direction, Img beam imaging
roi_x = [0, 2400]
roi_y = [1150,1300]#[1500, 2000] #[750, 1150] # Region of interest of Y direction, Img beam imaging

# #for tweezer:
# roi_y = [1150, 1220]
# roi_x = [1250, 1450]

#for tweezer, 2D (20240109):
# roi_y = [1130, 1240]
# roi_x = [1250, 1450]

# tweezer, 2D (20230124)
# roi_y = [1110, 1220]
# roi_x = [1250, 1450]

# roi_y = [1150, 1350]
# roi_x = [750, 2050]

# roi_y = [0, 2400]
# roi_x = [0, 2400]
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
    info_dict = hz.getAttributeDict(f)
    # images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
    para = float(hz.attributesToDictionary(f).get('globals').get(para_name))
    kinetix_roi_row= np.array(hz.attributesToDictionary(f).get('globals').get('kinetix_roi_row'))
    run_number = info_dict.get('run number')
    #print(f'run_number = {run_number} ')


site_roi_x = np.array([
       [1389, 1395],
       [1372, 1378],
       [1336, 1342],
       [1353, 1359],
       [1317, 1323],
       [1391, 1397],
       [1373, 1379],
       [1337, 1343],
       [1355, 1361],
       [1319, 1325],
       [1392, 1398],
       [1374, 1380],
       [1338, 1344],
       [1356, 1362],
       [1320, 1326],
       [1393, 1399],
       [1375, 1381],
       [1339, 1345],
       [1357, 1363],
       [1321, 1327],
       [1394, 1400],
       [1358, 1364],
       [1376, 1382],
       [1340, 1346],
       [1322, 1328]])
site_roi_y = np.array([
       [1118, 1124],
       [1119, 1125],
       [1120, 1126],
       [1120, 1126],
       [1121, 1127],
       [1136, 1142],
       [1137, 1143],
       [1138, 1144],
       [1138, 1144],
       [1139, 1145],
       [1154, 1160],
       [1155, 1161],
       [1156, 1162],
       [1156, 1162],
       [1157, 1163],
       [1172, 1178],
       [1173, 1179],
       [1174, 1180],
       [1174, 1180],
       [1175, 1181],
       [1190, 1196],
       [1191, 1197],
       [1191, 1197],
       [1192, 1198],
       [1193, 1199]])

threshold = 2.65731089e+03 * 0.6
site_roi_y = site_roi_y - kinetix_roi_row[0] - 1
site_roi_x = site_roi_x  - 1 #- roi_x[0]


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


# Set a ROI (in pxels) around the MOT
ROI = [roi_x, roi_y]

# The MOT image and the backgrounds image are saved into the h5 according to the run file
# Image with MOT from the even shots and image of background from odd shots

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
# count_file_path = folder_path+'\\data.csv'
first_image_file_path = folder_path+'\\first'
second_image_file_path = folder_path+'\\seconds'

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
# count_file_path = folder_path+'\\data.csv'
count_file_path = folder_path+'\\fault_rate_data.csv'



if run_number % 2 == 0: # even run number case
    first_image = images[image_types[0]]
    second_image = images[image_types[1]]

    np.save(first_image_file_path, first_image)
    np.save(second_image_file_path, second_image)

    if run_number == 0:
        with open(count_file_path, 'w') as f_object:
            f_object.write(f'')

    else:
        with open(count_file_path, 'a') as f_object:
            f_object.write(f'')



else:
    first_image_bkg = images[image_types[0]]
    second_image_bkg = images[image_types[1]]
    frist_image_load_file_path = first_image_file_path + '.npy'
    second_image_load_file_path = second_image_file_path + '.npy'

    first_image = np.load(frist_image_load_file_path)
    second_image = np.load(second_image_load_file_path)

    try:
        sub_image1 = first_image - first_image_bkg # subtraction of the background
        sub_image2 = second_image - second_image_bkg
    except:
        print('Start from even run number shots!')
        quit()

    # We subtract the background from the MOT only in the ROI
    roi_MOT_1 = sub_image1[:, roi_x[0]:roi_x[1]]
    roi_bkg_1 = sub_image1[:, roi_x_bkg[0]:roi_x_bkg[1]]

    roi_MOT_2 = sub_image2[:, roi_x[0]:roi_x[1]]
    roi_bkg_2 = sub_image2[:, roi_x_bkg[0]:roi_x_bkg[1]]


    rect_sig_1 = []
    atom_exist_lst_1 = []
    for i in np.arange(site_roi_x.shape[0]):
        site_roi_signal = sub_image1[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
        if site_roi_signal.sum() > threshold:
            rect_sig_1.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='gold', facecolor='none'))
            atom_exist_lst_1.append(1)
        else:
            atom_exist_lst_1.append(0)

    # print(atom_exist_lst_1)

    rect_sig_2 = []
    atom_exist_lst_2 = []
    for i in np.arange(site_roi_x.shape[0]):
        site_roi_signal = sub_image2[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
        if site_roi_signal.sum() > threshold:
            rect_sig_2.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='gold', facecolor='none'))
            atom_exist_lst_2.append(1)
        else:
            atom_exist_lst_2.append(0)

    # print(atom_exist_lst_2)


    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    (ax_mot_roi_1, ax_bkg_roi_1), (ax_mot_roi_2, ax_bkg_roi_2) = axs

    fig.suptitle('Mag = 7.424, 12.37, 18.56, Pixel = 6.5 um')

    for ax in axs[0]:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
    for ax in axs[1]:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    roi_image_scale = 50 #180 #180 #100 #500 #1000 #180 #150 #2000 #4096 #150

    roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale, origin = 'lower', extent = np.array([roi_x[0], roi_x[1], 0, kinetix_roi_row[1]]))

    roi_bkg_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale, origin = 'lower', extent = np.array([roi_x_bkg[0], roi_x_bkg[1], 0, kinetix_roi_row[1]]))

    ax_mot_roi_1.set_title('1st roi')
    pos = ax_mot_roi_1.imshow(roi_MOT_1, **roi_img_color_kw)
    ax_mot_roi_1.add_collection(PatchCollection(rect, match_original=True))
    ax_mot_roi_1.add_collection(PatchCollection(rect_sig_1, match_original=True))
    fig.colorbar(pos, ax=ax_mot_roi_1)

    ax_bkg_roi_1.set_title('1st bkg roi')
    pos = ax_bkg_roi_1.imshow(roi_bkg_1, **roi_bkg_color_kw)
    fig.colorbar(pos, ax=ax_bkg_roi_1)

    ax_mot_roi_2.set_title('2nd roi')
    pos = ax_mot_roi_2.imshow(roi_MOT_2, **roi_img_color_kw)
    ax_mot_roi_2.add_collection(PatchCollection(rect, match_original=True))
    ax_mot_roi_2.add_collection(PatchCollection(rect_sig_2, match_original=True))
    fig.colorbar(pos, ax=ax_mot_roi_2)

    ax_bkg_roi_2.set_title('2nd bkg roi')
    pos = ax_bkg_roi_2.imshow(roi_bkg_2, **roi_bkg_color_kw)
    fig.colorbar(pos, ax=ax_bkg_roi_2)

    atom_exist_lst_1 = np.array(atom_exist_lst_1)
    atom_exist_lst_2 = np.array(atom_exist_lst_2)


    fault_rate = sum( np.bitwise_xor(atom_exist_lst_1,atom_exist_lst_2)) /  site_roi_x.shape[0]

    # print(np.bitwise_xor(atom_exist_lst_1,atom_exist_lst_2))



    if run_number == 0:
        with open(count_file_path, 'w') as f_object:
            f_object.write(f'{fault_rate}\n')

    else:
        with open(count_file_path, 'a') as f_object:
            f_object.write(f'{fault_rate}\n')

    # os.remove(h5_path)







