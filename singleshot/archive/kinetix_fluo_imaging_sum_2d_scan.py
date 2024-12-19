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

# Constants
px = 1 #6.5 # Pixels size
mag = 1 #18.56 #7.424 #12.37 # Magnefication
counts_per_atom = 1 # Counts per atom 16.6 counts per atom per ms
#roi_x = [1000, 1800] #roi_x = [750, 1250]#roi_x = [850, 1250] # Region of interest of X direction, MOT beam imaging
#roi_y = [800, 1500] #roi_y = [800, 1000] #[750, 1150] # Region of interest of Y direction, MOT beam imaging

# for dipole trap (20240507):
roi_y = [1000,1200]
roi_x = [700,1900]

#for dipole trap (20240109):
# roi_x = [800,1750]#[750, 1200]#roi_x = [850, 1250] # Region of interest of X direction, Img beam imaging
# roi_y = [1150,1300]#[1500, 2000] #[750, 1150] # Region of interest of Y direction, Img beam imaging

# #for tweezer:
# roi_y = [1150, 1220]
# roi_x = [1250, 1450]

#for tweezer, 2D (20240109):
# roi_y = [1130, 1240]
# roi_x = [1250, 1450]

# tweezer, 2D (20230124)
# roi_y = [1110, 1220]
# roi_x = [1250, 1450]


# roi_y = [0, 2400]
# roi_x = [0, 2400]
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction
# para_name = 'n_shot'
# para_name = 'blue_456nm_duration'
# para_name = 'blue_456_detuning'



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
    loop_glob = []
    for group in g:
        for glob in g[group]:
            if g[group][glob][0:2] == "np":
                loop_glob.append(glob)
                # print(f"glob = {glob}")
    info_dict = hz.getAttributeDict(f)
    # images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
    # print(f"loop_glob = {loop_glob}")
    para_1 = float(hz.attributesToDictionary(f).get('globals').get(loop_glob[0]))
    try:
        para_2 = float(hz.attributesToDictionary(f).get('globals').get(loop_glob[1]))
    except:
        print('no second loop variable')
        sys.exit(1)
    try:
        blue_on = float(hz.attributesToDictionary(f).get('globals').get('do_456nm_laser'))
    except:
        pass
    run_number = info_dict.get('run number')
    print(f'run_number = {run_number}')



print(h5_path)
print(h5_path.split(".")[0]+"_no_image."+h5_path.split(".")[1])

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

# The MOT image and the backgrounds image are saved into the h5 according to the run file (first_MOT_images.py)
# as elements in the images dictionary. The MOT image is the first one, and the background is the second.
MOT_image = images[image_types[0]]
background_image = images[image_types[1]]
sub_image = MOT_image - background_image # subtraction of the background

# We subtract the background from the MOT only in the ROI
roi_MOT = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]

# Sum the counts from the (background subtracted) MOT in the ROI, and scale by the counts per atom (see Wiki)
# to find the atom number. Save all of these to the counts dictionary which will then be saved into the run's h5 file
electron_counts_mot = roi_MOT.sum()
atom_number_withbkg = electron_counts_mot / counts_per_atom
electron_counts_bkg = roi_bkg.sum()
bkg_number = electron_counts_bkg / counts_per_atom / roi_bkg.size * roi_MOT.size # average bkg floor in the size of roi_MOT
# atom_number = int(atom_number_withbkg)

# print('atom number with bkg',atom_number_withbkg)
# print('bkg number',bkg_number)
# print(roi_bkg)
atom_number = int(atom_number_withbkg- bkg_number)



# param_idx = 0
# fp = os.path.join(directory, folder)
# shots = az.getCompleteShots(az.getShots(fp))
# plot_parameter, param_array = getParamArray(shots, param_idx)


counts = {
    'counts per atom' : counts_per_atom,
    'counts': electron_counts_mot,
    'atom number': atom_number,
}


fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs

fig.suptitle('Mag = 7.424, 12.37, 18.56, Pixel = 6.5 um')

for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

image_scale = 3000 #4096 #2000 #4096 # 12 bit depth
raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

ax_mot_raw.set_title('Raw')
pos = ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_mot_raw)

ax_bkg_raw.set_title('Raw, no MOT')
pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_bkg_raw)

roi_image_scale = 900#900 #100 #500 #1000 #180 #150 #2000 #4096 #150
roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

try:
    if blue_on == 1.0:
        ax_mot_roi.set_title('MOT ROI, blue on')
    else:
        ax_mot_roi.set_title('MOT ROI')
except:
    ax_mot_roi.set_title('MOT ROI')


pos = ax_mot_roi.imshow(
    roi_MOT,
    extent=px*mag*np.array([roi_x[0], roi_x[1], roi_y[1], roi_y[0]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)


import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

#show 5 trap tweezers
# site_roi_x = np.array([[1260, 1266],

#                        [1317, 1323],

#                        [1335, 1341],

#                        [1353, 1359],

#                        [1371, 1377],

#                        [1389, 1395],])+1
# site_roi_y = np.array([[1181, 1188],

#                         [1180, 1187],

#                         [1179, 1186],

#                         [1179, 1186],

#                         [1178, 1185],

#                         [1177, 1184],])

#show 2D 25 traps old

site_roi_x = np.array([[1260, 1266],
                    [1387, 1393],
                    [1369, 1375],
                    [1334, 1340],
                    [1351, 1357],
                    [1316, 1322],
                    [1389, 1395],
                    [1371, 1377],
                    [1335, 1341],
                    [1353, 1359],
                    [1317, 1323],
                    [1390, 1396],
                    [1354, 1360],
                    [1372, 1378],
                    [1336, 1342],
                    [1318, 1324],
                    [1373, 1379],
                    [1391, 1397],
                    [1355, 1361],
                    [1337, 1343],
                    [1319, 1325],
                    [1374, 1380],
                    [1392, 1398],
                    [1356, 1362],
                    [1320, 1326],
                    [1338, 1344]])

site_roi_y = np.array([[1181, 1187],
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
                    [1173, 1179],
                    [1174, 1180],
                    [1175, 1181],
                    [1190, 1196],
                    [1190, 1196],
                    [1191, 1197],
                    [1192, 1198],
                    [1193, 1199],
                    [1208, 1214],
                    [1208, 1214],
                    [1209, 1215],
                    [1210, 1216],
                    [1210, 1216]])

rect = []
for i in np.arange(site_roi_x.shape[0]):
    rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))
# ax_mot_roi.add_collection(PatchCollection(rect, linewidth=1, edgecolor='r', facecolor='none'))



ax_bkg_roi.set_title('Background ROI')
pos = ax_bkg_roi.imshow(
    roi_bkg,
    vmin=-100,
    vmax=100,
    extent=px*mag*np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]]),
)
fig.colorbar(pos, ax=ax_bkg_roi)


print('atom_number =', atom_number, 'bkg_number=', bkg_number)

# Saving the counts dictionary into the run h5 file a group called analysis
with h5py.File(h5_path, mode='a') as f:

    analysis_group = f.require_group('analysis')
    region_group = analysis_group.require_group('counts')

    # print(g['Imaging']['save_image'])
    # if (g['Imaging']['save_image']=='False'):
    #     print('delete')
    #     del f['kinetix_images']

    hz.dictionaryToDatasets(region_group, counts, recursive=True)


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
# count_file_path = folder_path+'\\data.csv'
count_file_path = folder_path+'\\data.csv'


print(f"atom_number = {atom_number}, para_1 = {para_1}, para_2 = {para_2}")
if run_number == 0:
    with open(count_file_path, 'w') as f_object:
        f_object.write(f'{atom_number},{para_1},{para_2}\n')

else:
    with open(count_file_path, 'a') as f_object:
        f_object.write(f'{atom_number},{para_1},{para_2}\n')

# os.remove(h5_path)