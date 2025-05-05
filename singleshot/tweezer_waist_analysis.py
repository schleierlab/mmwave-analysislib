# -*- coding: utf-8 -*-
"""
@author: Lin Xin
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
import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    # print(f"site_roi_x ={site_roi_x}")
    ind = np.argsort(site_roi_x[:, 0])
    site_roi_x = site_roi_x[ind]
    site_roi_y = site_roi_y[ind]

    return np.array(site_roi_x), np.array(site_roi_y)

def plot_shots_avg(data, site_roi_x, site_roi_y, dx, n_shots =2, show_roi = True):
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    # if np.min(site_roi_x) == 0:
    #     roi_x = np.array([np.min(site_roi_x), np.max(site_roi_x)+10])
    # else:
    #     roi_x = np.array([np.min(site_roi_x)-10, np.max(site_roi_x)+10])

    roi_x = np.array([0, 2560])

    #print(f'roi_x = {repr(roi_x)}')
    # roi_x = [1275,1475] #[1250, 1450]
    site_roi_x = site_roi_x - roi_x[0]

    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))

    #fig.suptitle(f'{n_shots} shots average, Mag = 7.424, Pixel = 0.87 um')
    [ax1, ax2] = axs
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')

    ax2.set_xlabel('x [um]')
    ax2.set_ylabel('y [um]')

    (ax_first_image, ax_real_dimension) = axs

    image_scale = np.amax(data[:,roi_x[0]:roi_x[1]])#/2 # 12 bit dept
    raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

    #ax_first_image.set_title('first shot')
    print(roi_x)
    pos = ax_first_image.imshow(data[:,roi_x[0]:roi_x[1]], **raw_img_color_kw)
    fig.colorbar(pos, ax=ax_first_image)
    raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale, extent = [0, data.shape[1]*dx, 0,data.shape[0]*dx], aspect = 'equal')
    ax_real_dimension.imshow(data[:,roi_x[0]:roi_x[1]], **raw_img_color_kw)



    if show_roi == True:
        ax_first_image.add_collection(PatchCollection(rect, match_original=True))

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

def fit_gauss2D(image, plot = True):
    from scipy.optimize import curve_fit
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
    if plot == True:
        fig, ax = plt.subplots()
        ax.imshow(image)
    x_l = np.arange(L)
    y_l = np.arange(L)
    x, y = np.meshgrid(x_l, y_l)
    x_l_fit = np.linspace(0, L-1, 100)
    y_l_fit = np.linspace(0, L-1, 100)
    x_fit, y_fit = np.meshgrid(x_l_fit, y_l_fit)
    ind = np.unravel_index(np.argmax(image, axis=None), image.shape) #This is just (L,L) if we have a square with length L
    initial_guess = [np.max(image),*ind, 1, 1, 0, 0] #[amplitude, mux, muy, sigmax, sigmay, rotation, offset]
    try:
        #data_fitted = gauss2D((y, x), *initial_guess).reshape(len(y_l), len(x_l))
        #ax.contour(x, y, data_fitted.reshape(len(y_l), len(x_l)), 5, colors='w', alpha = 0.3)
        #print(image.ravel().shape)
        popt, pcov = curve_fit(gauss2D,(y, x),image.ravel(),p0=initial_guess)
        err = np.sqrt(np.diag(pcov))
        data_fitted = gauss2D((y_fit, x_fit), *popt).reshape(len(y_l_fit), len(x_l_fit))
        if plot == True:
            ax.contour(x_fit, y_fit, data_fitted.reshape(len(y_l_fit), len(x_l_fit)), 3, colors='w')
            plt.show()
            #print(popt)
            print('error:',err)

        (amplitude, mux, muy, sigmax, sigmay, rotation, offset) = popt
        # sigma_min = np.min([sigmax, sigmay])
        # sigma_max = np.max([sigmax, sigmay])
        # eccentricity = np.sqrt(1 - (sigma_min/sigma_max)**2)

        return (amplitude, mux, muy, sigmax, sigmay, rotation, offset)

    except RuntimeError:
        print("Bad Fit")
        return [0, 0]

def avg_shots_gauss_fit(data, site_roi_x, site_roi_y, plot = True):
    params_lst = []
    for i in np.arange(site_roi_x.shape[0]):
        data_roi = data[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
        params= fit_gauss2D(data_roi, plot)
        params_lst.append(params)
    return np.array(params_lst)


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
    run_number = info_dict.get('run number')
    images = hz.datasetsToDictionary(f['manta419b_tweezer_images'], recursive=True)

image_types = list(images.keys())
data = images[image_types[0]]

neighborhood_size = 10 #10 #6
threshold = 200 #35 #85 #100 #100
roi_y = [0, 1550]# [0, 2560]  #[1300,1860] #
roi_x = [0, 1500]#[0, 2560]  #[1300,1860] #
mag = 750/40.4  # 250/40.4
pix = 5.5  # um
dx = pix/mag

data_new = data[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]

# fig, axs = plt.subplots(nrows=1, ncols=1)

# axs.set_xlabel('x [px]')
# axs.set_ylabel('y [px]')

# image_scale = 3000
# raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

# axs.set_title('Raw')
# pos = axs.imshow(data_new, **raw_img_color_kw)
# fig.colorbar(pos, ax=axs)


site_roi_x, site_roi_y = auto_roi_detection(data_new, neighborhood_size, threshold)

print(repr(site_roi_x), repr(site_roi_y))

# print(repr(site_roi_x), repr(site_roi_y))

# site_roi_x = np.array([[1078, 1088],
#        [1094, 1104],
#        [1110, 1120],
#        [1127, 1137],
#        [1143, 1153],
#        [1159, 1169],
#        [1175, 1185],
#        [1192, 1202],
#        [1208, 1218],
#        [1225, 1235],
#        [1241, 1251],
#        [1257, 1267],
#        [1273, 1283],
#        [1289, 1299],
#        [1306, 1316],
#        [1322, 1332],
#        [1338, 1348],
#        [1354, 1364],
#        [1371, 1381],
#        [1387, 1397]])

# site_roi_y = np.array([[943, 953],
    # [927, 937],
    # [912, 922],
    # [896, 906],
    #    [880, 890],
    #    [865, 875],
    #    [849, 859],
    #    [833, 843],
    #    [818, 828],
    #    [802, 812],
    #    [786, 796],
    #    [771, 781],
    #    [755, 765],
    #    [739, 749],
    #    [724, 734],
    #    [708, 718],
    #    [692, 702],
    #    [677, 687],
    #    [661, 671],
    #    [645, 655]])

plot_shots_avg(data_new, site_roi_x, site_roi_y, n_shots=1, dx=dx, show_roi=True)

roi_number_lst = []
for i in np.arange(site_roi_x.shape[0]):
    site_roi_signal = data_new[site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
    roi_number_lst.append(np.sum(site_roi_signal))

params_lst = avg_shots_gauss_fit(data_new, site_roi_x, site_roi_y, plot = False)
amplitude, mux, muy, sigmax, sigmay, rotation, offset = params_lst[:, 0], params_lst[:, 1], params_lst[:, 2], params_lst[:, 3], params_lst[:, 4], params_lst[:, 5], params_lst[:, 6]

print(f" mux = {repr(mux)}, muy = {repr(muy)}, sigma_x = {repr(sigmax)}, sigmay = {repr(sigmay)}, rotation = {repr(rotation)},offset = {repr(offset)}")
waist_x = 2*np.mean(sigmax)*dx
waist_y = 2*np.mean(sigmay)*dx
amplitude = np.abs(amplitude)

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\tweezer_waist.csv'

# if run_number == 0:
#     with open(count_file_path, 'w') as f_object:
#         f_object.write(f'{waist_x},{waist_y}\n')

# else:
with open(count_file_path, 'a') as f_object:
        f_object.write(f'{waist_x},{waist_y}\n')

folder_path = '\\'.join(h5_path.split('\\')[0:-1])
count_file_path = folder_path+'\\tweezer_amplitude.csv'
roi_number_lst_path = folder_path+'\\roi_number_lst.csv'

np.savetxt(count_file_path, amplitude)
np.savetxt(roi_number_lst_path, roi_number_lst)

count_file_path = folder_path+'\\site_roi_x.csv'
np.savetxt(count_file_path, site_roi_x)

# plt.plot(2*sigmax*dx,'o',label="waist x")
# plt.plot(2*sigmay*dx,'o',label="waist y")
# plt.xlabel("sites")
# plt.ylabel("Gaussian waist (um)")
# plt.legend()