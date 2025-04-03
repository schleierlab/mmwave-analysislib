# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:11:12 2023

@author: sslab
"""
import sys

# root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"
root_path = r"X:\userlib\analysislib"

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
# import csv
import scipy.optimize as opt
from labscriptlib.shot_globals import shot_globals

# Constants
px = 5.5  # Pixels size
mag = 0.4  # Magnefication
counts_per_atom = 16.6  # Counts per atom 16.6 counts per atom per ms
# roi_x = [900, 1200]
# roi_y = [900, 1200]
roi_x = [800, 1100]
roi_y = [900, 1200]

# roi_x = [850, 1250] # Region of interest of X direction
# roi_y = [750, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400]  # Region of interest of X direction
roi_y_bkg = [1900, 2400]  # Region of interest of Y direction
m = 2.20694650e-25  # kg mass of cesium
kB = 1.3806503e-23  # J/K Boltzman constant

#2D Gaussian function with the capability to rotate the axes by angle theta. Returns 1D array,instead of 2D, using ravel, which is necessary for fitting.
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def gauss2D(x, amplitude, mux, muy, sigmax, sigmay, rotation, offset):
    """
    2D Gaussian, see: https://en.wikipedia.org/wiki/Gaussian_function
    Parameters:
        amplitude
        mux
        muy
        sigmax
        sigmay
        rotation
        slopex
        slopey
        offset
    """
    assert len(x) == 2
    X = x[0]
    Y = x[1]
    A = (np.cos(rotation)**2)/(2*sigmax**2) + (np.sin(rotation)**2)/(2*sigmay**2)
    B = (np.sin(rotation*2))/(4*sigmay**2) - (np.sin(2*rotation))/(4*sigmax**2)
    C = (np.sin(rotation)**2)/(2*sigmax**2) + (np.cos(rotation)**2)/(2*sigmay**2)
    G = amplitude*np.exp(-(A * (X - mux) ** 2 + 2 * B * (X - mux) * (Y - muy) + C * (Y - muy) ** 2)) + offset  # + slopex * X + slopey * Y + offset
    return G.ravel()  # np.ravel() Return a contiguous flattened array.


# Is this script being run from within an interactive lyse session?
if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

with h5py.File(h5_path, mode='r+') as f:
    g = hz.getGlobalsFromFile(h5_path)
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)


image_types = list(images.keys())
# print(images[image_types[0]])

# Defining the pixel size (um) and imaging from the magnification
# Also the total fov (in um) given the chip size in pixels
pixels = images[image_types[0]].shape[0]
fov = pixels*px/mag


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


# Creating a grid for the Gaussian fit
x_size = roi_x[1]-roi_x[0]
y_size = roi_y[1]-roi_y[0]
x = np.linspace(0, x_size-1, x_size)
y = np.linspace(0, y_size-1, y_size)
x, y = np.meshgrid(x, y)

###### An initial guess (which the results seem to be roughly robust to) and fit
#initial_guess = [ 3.48026329e+02,  2.31670864e+02,  1.92715502e+02,  5.56796476e+01,5.99932795e+01, -2.08146521e-01,  1.17119223e+01]
# initial_guess = np.array([62.3451402 , 111.05955417, 213.15920305,  33.02471839,
#         24.05161137,   5.43720939,  -0.25069135])

# ind = np.unravel_index(np.argmax(roi_MOT, axis=None), roi_MOT.shape)
# initial_guess = np.array([np.max(roi_MOT), *ind, x_size/2, y_size/2, 0, 0])
initial_guess = np.array([np.max(roi_MOT), x_size/2, y_size/2, x_size/2, y_size/2, 0, np.min(roi_MOT)])#0])
# np.array([ 5.50211592e+01,  1.11880793e+02,  1.68233167e+02,  2.70126498e+01,
#         3.78308458e+01, -2.58729750e+00,  1.03113035e-01])
# np.array([ 50.84076777, 261.28397715, 116.98008048,  38.41147601,
#         27.28217021, -13.57208579,   1.6931073 ])

# [2666.35537127,  231.94367821,  186.53784832,   29.84723883, 37.6077219 ,   18.8263272 ,   15.97871513]
# popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), roi_MOT.ravel(), p0=initial_guess)
popt, pcov = opt.curve_fit(gauss2D, (y, x), roi_MOT.ravel(), p0=initial_guess)
# print(repr(popt))
#fit = twoD_Gaussian((x, y), *popt)
fit = gauss2D((y, x), *popt)
# Integrating the Gaussian fit based on the amplitude and standard deviations
atom_number_gaussian = np.abs(2 * np.pi * popt[0] * popt[3] * popt[4] / counts_per_atom)
sigma = np.sort(np.abs([popt[3], popt[4]]))  # gaussian waiast in pixel, [short axis, long axis]
gaussian_waist = np.array(sigma)*px*1e-6/mag # convert from pixel to distance m

tof = g['bm_tof_imaging_delay']
# if g['do_dipole_trap_tof_check'] == True:
#     tof = g['img_tof_imaging_delay']
# print(gaussian_waist)
temperature = m / kB * (gaussian_waist/tof)**2

# Sum the counts from the (background subtracted) MOT in the ROI, and scale by the counts per atom (see Wiki)
# to find the atom number. Save all of these to the counts dictionary which will then be saved into the run's h5 file
electron_counts_mot = roi_MOT.sum()
atom_number_withbkg = electron_counts_mot / counts_per_atom
electron_counts_bkg = roi_bkg.sum()
bkg_number = electron_counts_bkg / counts_per_atom / roi_bkg.size * roi_MOT.size # average bkg floor in the size of roi_MOT
# atom_number = int(atom_number_withbkg)
# atom_number = int(atom_number_withbkg- bkg_number)
atom_number = atom_number_gaussian

counts = {
    'counts per atom' : counts_per_atom,
    'counts': electron_counts_mot,
    'atom number': atom_number,
}


fig, axs = plt.subplots(nrows=2, ncols=2)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs
for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

image_scale = 30 #300
# raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)
raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

ax_mot_raw.set_title('Raw')
ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)

ax_bkg_raw.set_title('Raw, no MOT')
pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
fig.colorbar(pos, ax=ax_bkg_raw)

ax_mot_roi.set_title('MOT ROI')
pos = ax_mot_roi.imshow(
    roi_MOT,
    extent=px/mag*np.array([roi_y[0], roi_y[1], roi_x[0], roi_x[1]]),
    **raw_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)

ax_bkg_roi.set_title('Background ROI')
pos = ax_bkg_roi.imshow(
    roi_bkg,
    vmin=-10,
    vmax=10,
    extent=px/mag*np.array([roi_y_bkg[0], roi_y_bkg[1], roi_x_bkg[0], roi_x_bkg[1]]),
)
fig.colorbar(pos, ax=ax_bkg_roi)


print('atom_number =', atom_number, 'bkg_number =', bkg_number, 'temperature =', temperature)


# Saving the counts dictionary into the run h5 file a group called analysis
with h5py.File(h5_path, mode='a') as f:

    analysis_group = f.require_group('analysis')
    region_group = analysis_group.require_group('counts')

    hz.dictionaryToDatasets(region_group, counts, recursive=True)


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'

with open(count_file_path, 'a') as f_object:
    f_object.write(f'{atom_number}\n')


temp_file_path = folder_path+'\\temp.csv'

with open(temp_file_path, 'a') as f_object:
    f_object.write(f'{tof},{temperature[0]},{temperature[1]}\n')

tof_fit_file_path = folder_path+'\\TOFdata.csv'

with open(tof_fit_file_path, 'a') as f_object:
    f_object.write(f'{atom_number},{tof},{gaussian_waist[0]}, {gaussian_waist[1]}\n')
