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
import scipy.optimize as opt

# Constants
# molasses (20230318)
# roi_x = [950, 1200] #roi_x = [850, 1250] # Region of interest of X direction
# roi_y = [900, 1150] #[750, 1150] # Region of interest of Y direction

# dipole trap (20230318)
roi_x = [700, 1700]#roi_x = [850, 1250] # Region of interest of X direction
# roi_y = [1250, 1550]
roi_y = [1500, 1700]

roi_x = [500, 2000]
roi_y = [500, 2000]

roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction


# Is this script being run from within an interactive lyse session?
if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]


rep = h5_path[-5:-3]
# with h5py.File(h5_path, mode='r+') as f:
#     g = hz.attributesToDictionary(f['globals'])
#     info_dict = hz.getAttributeDict(f)
#     images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
#     uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get('mw_detuning_end'))
#     run_number = info_dict.get('run number')


with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    for group in g:
        for glob in g[group]:
            if g[group][glob][0:2] == "np":
                loop_glob = glob
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
    try:
        uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get(loop_glob))
    except:
        uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get('n_shot'))

    run_number = info_dict.get('run number')



image_types = list(images.keys())

# The MOT image and the backgrounds image are saved into the h5 according to the run file (first_MOT_images.py)
# as elements in the images dictionary. The MOT image is the first one, and the background is the second.c
MOT_image = images[image_types[0]]
background_image = images[image_types[1]]
sub_image = MOT_image - background_image # subtraction of the background

# We subtract the background from the MOT only in the ROI
roi_MOT = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]


# popt, pcov = fit_gauss2D(roi_MOT)
# print(repr(popt))
# #Integrating the Gaussian fit based on the amplitude and standard deviations
# gaussian_peak = popt[0]



atom_number = np.sum(np.sum(roi_MOT, axis = 0), axis = 0)



fig, axs = plt.subplots(nrows=2, ncols=2)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs
for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

image_scale = 1000 #3000 #300
# raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)
raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

ax_mot_raw.set_title('Raw')
ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)

ax_bkg_raw.set_title('Raw, no MOT')
pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_bkg_raw)

image_scale = 10
# raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)
roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

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
    vmin=-10,
    vmax=10,
    extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[1], roi_y_bkg[0]]),
)
fig.colorbar(pos, ax=ax_bkg_roi)



folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'




if  rep == '_0':
    with open(count_file_path, 'w') as f_object:
        f_object.write(f'{atom_number},{uwave_detuning}\n')

else:
    with open(count_file_path, 'a') as f_object:
        f_object.write(f'{atom_number},{uwave_detuning}\n')


