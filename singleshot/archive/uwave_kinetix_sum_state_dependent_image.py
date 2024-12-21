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
with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
    uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get('uwave_time'))
    run_number = info_dict.get('run number')
    kinetix_roi_row= np.array(hz.attributesToDictionary(f).get('globals').get('kinetix_roi_row'))



image_types = list(images.keys())

# The MOT image and the backgrounds image are saved into the h5 according to the run file (first_MOT_images.py)
# as elements in the images dictionary. The MOT image is the first one, and the background is the second.c
MOT_image = images[image_types[0]]
MOT_image_2 = images[image_types[1]]
background_image = images[image_types[2]]
background_image_2 = images[image_types[3]]
sub_image = MOT_image - background_image # subtraction of the background
sub_image_2 = MOT_image_2 - background_image_2 # subtraction of the background


roi_MOT = sub_image[:, roi_x[0]:roi_x[1]]
roi_bkg = sub_image[:, roi_x_bkg[0]:roi_x_bkg[1]]
roi_MOT_2 = sub_image_2[:, roi_x[0]:roi_x[1]]
roi_bkg_2 = sub_image_2[:, roi_x_bkg[0]:roi_x_bkg[1]]



# popt, pcov = fit_gauss2D(roi_MOT)
# print(repr(popt))
# #Integrating the Gaussian fit based on the amplitude and standard deviations
# gaussian_peak = popt[0]



atom_number = np.sum(np.sum(roi_MOT, axis = 0), axis = 0)
atom_number_2 = np.sum(np.sum(roi_MOT_2, axis = 0), axis = 0)



fig, axs = plt.subplots(nrows=2, ncols=2)

(ax_mot_1, ax_mot_2), (ax_bkg_1, ax_bkg_2) = axs
for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

image_scale = 10
roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)


ax_mot_1.set_title('1st image roi')
pos = ax_mot_1.imshow(
    roi_MOT,
    #extent=np.array([roi_x[0], roi_x[1], roi_y[1], roi_y[0]]),
    **roi_img_color_kw,
)

ax_mot_2.set_title('2nd image roi')
pos = ax_mot_2.imshow(
    roi_MOT_2,
    #extent=np.array([roi_x[0], roi_x[1], roi_y[1], roi_y[0]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_2)

ax_bkg_1.set_title('1st bkg roi')
pos = ax_bkg_1.imshow(
    roi_bkg,
    vmin=-10,
    vmax=10,
    #extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[1], roi_y_bkg[0]]),
)

ax_bkg_2.set_title('2nd bkg roi')
pos = ax_bkg_2.imshow(
    roi_bkg_2,
    vmin=-10,
    vmax=10,
    #extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[1], roi_y_bkg[0]]),
)

fig.colorbar(pos, ax=ax_bkg_2)


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'




if  rep == '_0':
    with open(count_file_path, 'w') as f_object:
        f_object.write(f'{atom_number},{atom_number_2},{uwave_detuning}\n')

else:
    with open(count_file_path, 'a') as f_object:
        f_object.write(f'{atom_number},{atom_number_2},{uwave_detuning}\n')


