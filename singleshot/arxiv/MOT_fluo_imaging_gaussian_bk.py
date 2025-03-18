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
px = 1#5.5 # Pixels size
mag = 1#0.4 # Magnefication
counts_per_atom = 16.6 # Counts per atom 16.6 counts per atom per ms
roi_x = [550, 1350]#roi_x = [850, 1250] # Region of interest of X direction
roi_y = [650, 1450] #[750, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction

def fit_gauss2D(image):
    from scipy.optimize import curve_fit
    def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
        height = data.max()
        rotation = 0
        offset = 0
        return [height, x, y, width_x, width_y, rotation, offset]
    '''
    Parameter:
        image: a 2D array that includes an ellipise
    Note:
        numpy.unravel_index(indices, shape..):
        Converts a flat index or array of flat indices into a tuple of coordinate arrays.
        np.argmax():
        Returns the indices of the maximum values along an axis.
    '''
    L = image.shape[0]
    x_l = np.arange(L)
    y_l = np.arange(L)
    x, y = np.meshgrid(x_l, y_l)
    initial_guess = moments(image) #[amplitude, mux, muy, sigmax, sigmay, rotation, offset]
    print(initial_guess)
    try:
        #data_fitted = gauss2D((y, x), *initial_guess).reshape(len(y_l), len(x_l))
        #ax.contour(x, y, data_fitted.reshape(len(y_l), len(x_l)), 5, colors='w', alpha = 0.3)
        popt, pcov = curve_fit(gauss2D,(y, x),image.ravel(),p0=initial_guess)
        err = np.sqrt(np.diag(pcov))

        sigmax = popt[3]
        sigmay = popt[4]

        sigmax_err = err[3]
        sigmay_err = err[4]

        sigma = np.array([sigmax, sigmay])
        indicies = sigma.argsort()

        sigma_err= np.array([sigmax_err, sigmay_err])

        [sigma_min, sigma_max ]= sigma[indicies]
        [sigma_min_err, sigma_max_err]=sigma_err[indicies]


        return popt, pcov #(sigma_max, sigma_min, sigma_max_err, sigma_min_err)

    except RuntimeError:
        print("Bad Fit")
        return [0, 0]

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
        G = amplitude*np.exp(-(A * (X - mux) ** 2 + 2 * B * (X - mux) * (Y - muy) + C * (Y - muy) ** 2)) + offset #+ slopex * X + slopey * Y + offset
        return G.ravel() #np.ravel() Return a contiguous flattened array.

#2D Gaussian function with the capability to rotate the axes by angle theta. Returns 1D array,instead of 2D, using ravel, which is necessary for fitting.
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


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
    images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)


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


#Creating a grid for the Gaussian fit
x_size = roi_x[1]-roi_x[0]
y_size = roi_y[1]-roi_y[0]
x = np.linspace(0, x_size-1, x_size)
y = np.linspace(0, y_size-1, y_size)
x, y = np.meshgrid(x, y)

###### An initial guess (which the results seem to be roughly robust to) and fit
initial_guess = (roi_MOT.max(), 371.20496207, 278.97278791,  32.40847851,
        27.19507109,  -0.54671273,   1.15411987)
popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), roi_MOT.ravel(), p0=initial_guess)
# popt, pcov = fit_gauss2D(roi_MOT)
print(repr(popt))
fit = twoD_Gaussian((x, y), *popt)
#Integrating the Gaussian fit based on the amplitude and standard deviations
atom_number_gaussian = 2 * np.pi * popt[0] * popt[3] * popt[4] / counts_per_atom


# popt =[184.16508502, 371.20496207, 278.97278791,  32.40847851,
#         27.19507109,  -0.54671273,   1.15411987] # go to (Bx, By, Bz) = (0, 0, 0)mG before imaging
# popt_1, pcov = opt.curve_fit(lambda xy, A, offset: gauss2D(xy, A, popt[1], popt[2],  popt[3],  popt[4], popt[5],  offset), (y, x), roi_MOT.ravel(), p0=(roi_MOT.max() ,0))
# atom_number_gaussian = 2 * np.pi * popt_1[0] * popt[3] * popt[4] / counts_per_atom



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

image_scale = 300
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
    extent=px*mag*np.array([roi_y[0], roi_y[1], roi_x[0], roi_x[1]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)

ax_bkg_roi.set_title('Background ROI')
pos = ax_bkg_roi.imshow(
    roi_bkg,
    vmin=-10,
    vmax=10,
    extent=px*mag*np.array([roi_y_bkg[0], roi_y_bkg[1], roi_x_bkg[0], roi_x_bkg[1]]),
)
fig.colorbar(pos, ax=ax_bkg_roi)


print('atom_number =', atom_number, 'bkg_number=', bkg_number)
#print(g)

# Saving the counts dictionary into the run h5 file a group called analysis
with h5py.File(h5_path, mode='a') as f:

    analysis_group = f.require_group('analysis')
    region_group = analysis_group.require_group('counts')

    hz.dictionaryToDatasets(region_group, counts, recursive=True)


folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\data.csv'

with open(count_file_path, 'a') as f_object:
    f_object.write(f'{atom_number}\n')
