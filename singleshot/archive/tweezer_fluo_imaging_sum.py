# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:11:12 2023

@author: sslab
"""
import sys
root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"

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
import csv

check_tweezer_position = False

# Constants
px = 1#5.5 # Pixels size
mag = 1#0.4 # Magnefication
counts_per_atom = 16.6 # Counts per atom 16.6 counts per atom per ms
roi_x = [0, 2560]
roi_y = [0, 2560]
# roi_x = [500, 1600] # Region of interest of X direction
# roi_y = [1000, 1700] # Region of interest of Y direction


#2024/04/16 (tweezers)
# roi_x = np.array([450,1000])+200
# roi_y = np.array([1100,1500])-100

roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction




# Is this script being run from within an interactive lyse session?
if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
    print(h5_path)
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['manta419b_tweezer_images'], recursive=True)


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
if check_tweezer_position == True:
    sub_image = MOT_image
else:
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
atom_number = int(atom_number_withbkg- bkg_number)


counts = {
    'counts per atom' : counts_per_atom,
    'counts': electron_counts_mot,
    'atom number': atom_number,
}


fig, axs = plt.subplots(nrows=2, ncols=2)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs

fig.suptitle('Mag = 7.424, 12.37, 18.56, Pixel = 6.5 um')

for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

image_scale = 100 #4096 # 12 bit depth
raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

ax_mot_raw.set_title('Raw')
pos = ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_mot_raw)

ax_bkg_raw.set_title('Raw, no MOT')
pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_bkg_raw)

# if check_tweezer_position == True:
#     roi_image_scale = 4000 #100
# else:
#     roi_image_scale = 100

roi_image_scale = 50 #100
roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

ax_mot_roi.set_title('MOT ROI')
pos = ax_mot_roi.imshow(
    roi_MOT,
    extent=px*mag*np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)

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

    hz.dictionaryToDatasets(region_group, counts, recursive=True)


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'

with open(count_file_path, 'a') as f_object:
    f_object.write(f'{atom_number}\n')
