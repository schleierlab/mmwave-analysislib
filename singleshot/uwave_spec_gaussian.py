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
roi_x = [900, 1150]#roi_x = [850, 1250] # Region of interest of X direction
roi_y = [900, 1150] #[750, 1150] # Region of interest of Y direction
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
#     images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
#     uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get('mw_detuning'))
#     run_number = info_dict.get('run number')

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    for group in g:
        for glob in g[group]:
            if g[group][glob][0:2] == "np":
                loop_glob = glob
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
    try:
        uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get(loop_glob))
    except:
        uwave_detuning = float(hz.attributesToDictionary(f).get('globals').get('n_shot'))

    run_number = info_dict.get('run number')




image_types = list(images.keys())

# The MOT image and the backgrounds image are saved into the h5 according to the run file (first_MOT_images.py)
# as elements in the images dictionary. The MOT image is the first one, and the background is the second.
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


L = roi_MOT.shape[0]
x_l = np.arange(L)
y_l = np.arange(L)
x, y = np.meshgrid(x_l, y_l)
# popt = np.array([7.52013393e+02, 1.27215996e+02, 1.26715820e+02, 2.22250282e+01,
#        2.18626235e+01, 3.67469011e-01, 3.92932091e+00]) # go to (Bx, By, Bz) = (0, 0, 0)mG before imaging
popt = [ 1.36839142e+02,  1.47518009e+02,  7.77484432e+01,  2.98988168e+01,
        2.73066766e+01, -1.41966933e-01,  8.83168756e-01]
popt_1, pcov = opt.curve_fit(lambda xy, A, offset: gauss2D(xy, A, popt[1], popt[2],  popt[3],  popt[4], popt[5],  offset), (y, x), roi_MOT.ravel(), p0=(roi_MOT.max() ,0))
gaussian_peak = popt_1[0]
# print(gaussian_peak)


atom_number = gaussian_peak
# atom_number = roi_MOT.sum()-roi_bkg.sum()/roi_bkg.size * roi_MOT.size


fig, axs = plt.subplots(nrows=2, ncols=2)

(ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs
for ax in axs[0]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
for ax in axs[1]:
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

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
    extent=np.array([roi_y[0], roi_y[1], roi_x[0], roi_x[1]]),
    **roi_img_color_kw,
)
fig.colorbar(pos, ax=ax_mot_roi)

ax_bkg_roi.set_title('Background ROI')
pos = ax_bkg_roi.imshow(
    roi_bkg,
    vmin=-10,
    vmax=10,
    extent=np.array([roi_y_bkg[0], roi_y_bkg[1], roi_x_bkg[0], roi_x_bkg[1]]),
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


