# -*- coding: utf-8 -*-
"""
@author: Lin Xin
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
import glob

from tkinter import Tk
from tkinter.filedialog import askdirectory


def avg_all_shots(folder, shots = 'defult', loop = True):
    n_shots = np.size([i for i in os.listdir(folder) if i.endswith('.h5')])

    if shots == 'defult':
        shots = n_shots
    # roi_number_lst = np.zeros([site_roi_x.shape[0], shots])
    # print(np.shape(roi_number_lst))

    for cnt in (np.arange(shots)):
        if loop == True:
            if cnt % 2 == 0:
                string = glob.glob(folder + f'\*{cnt}.h5')
                h5_path = string[0]
                #h5_path = folder + rf'{cnt:01}.h5'
                # print(h5_path)
                with h5py.File(h5_path, mode='r+') as f:
                    # g = hz.attributesToDictionary(f['globals'])
                    # info_dict = hz.getAttributeDict(f)
                    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)

                    # run_number = info_dict.get('run number')
                image_types = list(images.keys())
                if cnt == 0:
                    print(f'size of the image is {np.shape(images[image_types[0]])}')

                    data = np.zeros((2, ) + np.shape(images[image_types[0]]))
                    bkg = np.zeros((2, ) + np.shape(images[image_types[0]]))
                # print(image_types)
                # image = np.array((images[image_types[0]], images[image_types[1]]))
                # print(np.shape(image))
                try:
                    data[0] = data[0]+images[image_types[0]]
                    data[1] = data[1]+images[image_types[1]]
                except:
                    sys.exit('The data is not created, start from the first shot')

            else:
                string = glob.glob(folder + f'\*{cnt}.h5')
                h5_path = string[0]
                #h5_path = folder + rf'{cnt:01}.h5'
                # print(h5_path)
                with h5py.File(h5_path, mode='r+') as f:
                    # g = hz.attributesToDictionary(f['globals'])
                    # info_dict = hz.getAttributeDict(f)
                    images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
                    # run_number = info_dict.get('run number')

                image_types = list(images.keys())
                # print(image_types)
                # image = np.array((images[image_types[0]], images[image_types[1]]))
                # print(np.shape(image))
                try:
                    bkg[0] = bkg[0]+images[image_types[0]]
                    bkg[1] = bkg[1]+images[image_types[1]]
                except:
                    sys.exit('The bkg is not created, start from the first shot')
        # print(cnt)

    if loop == True:
        N = shots/2
    else:
        N = shots

    return ((data-bkg)/N, bkg/N, N)


def avg_shots_multi_roi_avg_bkg_sub(folder, site_roi_y, site_roi_x, avg_bkg_img, shots = 'defult', loop = True, plot_single_shots = False, image_scale = 'default'):
    n_shots = np.size([i for i in os.listdir(folder) if i.endswith('.h5')])
    roi_x = np.array([np.min(site_roi_x)-10, np.max(site_roi_x)+10])

    if shots == 'defult':
        shots = n_shots

    if loop == True:
        N = int(n_shots/2)
    else:
        N = n_shots

    roi_number_lst = np.zeros([2, site_roi_x.shape[0], N])

    # print(np.shape(roi_number_lst))

    for cnt in (np.arange(shots)):
        if loop == True:
            if cnt % 2 == 1:
                continue
        # print(cnt)
        string = glob.glob(folder + f'\*{cnt}.h5')
        h5_path = string[0]
        #h5_path = folder + rf'{cnt:01}.h5'
        # print(h5_path)
        with h5py.File(h5_path, mode='r+') as f:
            images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)

        image_types = list(images.keys())
        if cnt == 0:
            print(f'size of the image is {np.shape(images[image_types[0]])}')
            data = np.zeros((2, ) + np.shape(images[image_types[0]]))



        image = np.array((images[image_types[0]], images[image_types[1]]))
        sub_image = image - avg_bkg_img

        # data = data + sub_image
        data[0] = data[0]+sub_image[0]
        data[1] = data[1]+sub_image[1]

        if loop == True:
            cnt = int(cnt/2)
        rect = []
        for i in np.arange(site_roi_x.shape[0]):
            roi_signal1 = sub_image[0, site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
            roi_signal2 = sub_image[1, site_roi_y[i,0]:site_roi_y[i,1], site_roi_x[i,0]:site_roi_x[i,1]]
            electron_counts1 = roi_signal1.sum()
            electron_counts2 = roi_signal2.sum()
            roi_number_lst[0, i, cnt] = electron_counts1
            roi_number_lst[1, i, cnt] = electron_counts2

            if plot_single_shots == True:
                rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))

        if plot_single_shots == True:
            fig, axs = plt.subplots(nrows=1, ncols=1)
            #fig.suptitle(f'{len(bkg_number_lst)} shots average, Mag = 7.424, Pixel = 0.87 um')
            axs.set_xlabel('x [px]')
            axs.set_ylabel('y [px]')
            if image_scale == 'default':
                img_scale = np.amax(sub_image[0, :, roi_x[0]:roi_x[1]]) # 12 bit dept
            else:
                img_scale = image_scale
            raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=img_scale)

            pos = axs.imshow(sub_image[0,:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
            fig.colorbar(pos, ax=axs)
            axs.add_collection(PatchCollection(rect, match_original=True))

    return (data/N, roi_number_lst, N)

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

    return np.array(site_roi_x), np.array(site_roi_y)

def plot_shots_avg(data, site_roi_x,site_roi_y, n_shots =2, show_roi = True):
    roi_x = np.array([np.min(site_roi_x)-10, np.max(site_roi_x)+10])

    print(f'roi_x = {repr(roi_x)}')
    # roi_x = [1275,1475] #[1250, 1450]
    site_roi_x = site_roi_x - roi_x[0]

    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))
    fig.suptitle(f'{n_shots} shots average, Mag = 7.424, Pixel = 0.87 um')
    for ax in axs:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    (ax_first_image, ax_second_image) = axs

    image_scale = np.amax(data[0, :, roi_x[0]:roi_x[1]]) # 12 bit dept
    raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

    ax_first_image.set_title('first shot')
    pos = ax_first_image.imshow(data[0,:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
    #fig.colorbar(pos, ax=ax_first_image)

    ax_second_image.set_title('second shot')
    pos = ax_second_image.imshow(data[1,:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
    fig.colorbar(pos, ax=ax_second_image)


    if show_roi == True:
        ax_first_image.add_collection(PatchCollection(rect, match_original=True))
        ax_second_image.add_collection(PatchCollection(rect, match_original=True))


def histagram_fit_and_threshold(roi_number_lst, site_roi_x, plot_histagram = False, plot_double_gaussian_fit = False, plot_gaussian_poisson_fit = False, sub_bkg = False, cpa = 'default', print_value = False, do_neighbour_bkg_sub = True):
    import scipy.optimize as optimize
    def gaussianpoisson_pdf_fit(X, C, mu, sigma, CPA):
        kmax = 20
        from scipy.stats import poisson
        return sum([C*np.exp(-(X/CPA-k)**2/(2*sigma**2))*poisson.pmf(k, mu) for k in np.arange(0, kmax)])

    def double_gaussian_fit( x, c1, mu1, sigma1, c2, mu2, sigma2 ):
        res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
            + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
        return res

    def threshold(c1, mu1, sigma1, c2, mu2, sigma2, cpa):
        #calcuate threshold based on the double gaussian fit.
        x = np.linspace(-0.25*cpa,1.5*cpa,1000)
        f = double_gaussian_fit(x, c1, mu1, sigma1, 0, mu2, sigma2)
        g = double_gaussian_fit(x, 0, mu1, sigma1, c2, mu2, sigma2)
        idx = np.argwhere(np.diff(np.sign(f-g))).flatten() # could have another point at negative value (or much larger than 1cpa?) so we need to exclude it
        # if len(idx) > 1:
        #     x0 = abs(0.5*cpa-x[idx[0]])
        #     x1 = abs(0.5*cpa-x[idx[1]])
        return x[idx]


    def prob_of_one_atom(c0,sigma0, c1, sigma1):
        prob = c1*sigma1/(c0*sigma0+c1*sigma1)
        return prob

    def image_fidelity(mu1, sigma1, mu2, sigma2, ff, th):
        import scipy.special as special
        # Lin's way of calculating fidelity, see wiki:
        # sigma12_sq = sigma1**2 + sigma2**2
        # fidelity = 1-np.exp(-(mu1-mu2)**2/2/sigma12_sq)/np.sqrt(2*np.pi*sigma12_sq)

        # P0(x < th) and P1( x > th )
        cdf0 = 0.5*(1+special.erf((th-mu1)/(np.sqrt(2)*sigma1))) #integrate from -infinity to th
        cdf1 = 0.5*(1+special.erf((th-mu2)/(np.sqrt(2)*sigma2)))
        p = (1-ff)*(1-cdf0)+ff*cdf1
        fidelity = 1-p
        return fidelity





    first_shot_roi_number = roi_number_lst[0,:]
    bkg_number_lst = first_shot_roi_number[0,:]

    if do_neighbour_bkg_sub == True:
        for i in range(roi_number_lst.shape[0]):
            first_shot_roi_number[i,:]  = first_shot_roi_number[i,:] - np.mean(bkg_number_lst)


    if sub_bkg == True:
        bkg_mean = np.mean(bkg_number_lst)
        print(bkg_mean)
    else:
        bkg_mean = 0

    all_roi_number_lst = first_shot_roi_number[1:site_roi_x.shape[0],:].flatten()


    if plot_histagram == True:
        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        fig.suptitle(f'{all_roi_number_lst.shape[0]} samples')
        for ax in axs:
            ax.set_xlabel('counts')
            ax.set_ylabel('frequency')

        (ax_first_image, ax_second_image) = axs
        ax_first_image.hist(all_roi_number_lst-bkg_mean, bins = 250, label = f'all sites')
        ax_first_image.hist(bkg_number_lst-bkg_mean, bins = 10, label = f'0th site, bkg')
        ax_first_image.legend()
        ax_first_image.set_title('first shot')
        #plt.xlim([-5000,10000])

    x_data = all_roi_number_lst
    n_bin = 250
    hist, bin_edges = np.histogram(x_data, bins = n_bin)
    hist=hist/sum(hist)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float)
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2

    y_hist=hist


    #gaussian_poisson_fit
    if plot_gaussian_poisson_fit == True:
        sigma = 0.1
        mu = 1
        p0_cpa = 3000 #6000*200

        param_optimised,param_covariance_matrix = optimize.curve_fit(gaussianpoisson_pdf_fit,x_hist,y_hist,p0=[max(y_hist),mu,sigma,p0_cpa],maxfev=5000)
        if cpa == 'default':
            cpa = param_optimised[3]

        fig = plt.figure()
        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
        plt.plot(x_hist_2/param_optimised[3],gaussianpoisson_pdf_fit(x_hist_2,*param_optimised),'r.:',label='Gaussian-Poissonian fit')
        plt.legend()

        #Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        plt.hist(x_data/param_optimised[3], weights=weights, bins = n_bin)

        #setting the label,title and grid of the plot
        plt.xlabel("Atoms")
        plt.ylabel("Probability")
        plt.grid("on")
        plt.title(f"atom roi")
        plt.show()
        print('[C, mu, sigma, CPA] =', param_optimised)

    #double_gaussian_fit
    cpa = 2000
    sigma1 = 0.1*cpa
    sigma2 = 0.1*cpa
    mu1 = 0
    mu2 = cpa
    c1 = max(y_hist)
    c2 = c1+2000
    param_optimised,param_covariance_matrix = optimize.curve_fit(double_gaussian_fit,x_hist,y_hist,p0=[c1, mu1, sigma1, c2, mu2, sigma2])

    (c0, mu0, s0, c1, mu1, s1) = param_optimised
    s0 = abs(s0)
    s1 = abs(s1)
    cpa = mu1
    th = threshold(c0, mu0, s0, c1, mu1, s1, cpa)
    ff = prob_of_one_atom(c0,s0, c1, s1) #filling fraction
    f = image_fidelity(mu0, s0, mu1, s1, ff, th)

    fig_2, axs_2 = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    (ax_first_image_2, ax_second_image_2) = axs_2

    for ax in axs_2:
        ax.set_xlabel('counts')
        ax.set_ylabel('Probability')

    if plot_double_gaussian_fit == True:
        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
        ax_first_image_2.plot(x_hist_2,double_gaussian_fit(x_hist_2,*param_optimised),'r.:',label='Double-Gaussian fit')
        ax_first_image_2.legend()

        #Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax_first_image_2.hist(x_data, weights=weights, bins = n_bin)

        #setting the label,title and grid of the plot
        ax_first_image_2.grid("on")
        ax_first_image_2.set_title(f"first shot")
        print(' [c1, mu1, sigma1, c2, mu2, sigma2] =', param_optimised)
        #print('mu2/sigma2 =', param_optimised[4]/param_optimised[5])

    if print_value == True:
        print('1st image Probability of one atom = ', prob_of_one_atom(c0,s0, c1, s1))
        print('1st image Imaging fidelity upper limit =', image_fidelity(mu0, s0, mu1, s1, ff, th))
        print('1st image threshold = ', th)


    second_shot_roi_number = roi_number_lst[1,:]
    bkg_number_lst = second_shot_roi_number[0,:]

    if sub_bkg == True:
        bkg_mean = np.mean(bkg_number_lst)
        print(bkg_mean)
    else:
        bkg_mean = 0

    if do_neighbour_bkg_sub == True:
        for i in range(roi_number_lst.shape[0]):
            second_shot_roi_number[i,:]  = second_shot_roi_number[i,:] - np.mean(bkg_number_lst)

    all_roi_number_lst = second_shot_roi_number[1:site_roi_x.shape[0],:].flatten()

    if plot_histagram == True:
        ax_second_image.hist(all_roi_number_lst-bkg_mean, bins = 250, label = f'all sites')
        ax_second_image.hist(bkg_number_lst-bkg_mean, bins = 10, label = f'0th site, bkg')
        ax_second_image.legend()
        ax_second_image.set_title('2nd shot')


    x_data = all_roi_number_lst
    n_bin = 250
    hist, bin_edges = np.histogram(x_data, bins = n_bin)
    hist=hist/sum(hist)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float)
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2

    y_hist=hist


    #gaussian_poisson_fit
    if plot_gaussian_poisson_fit == True:
        sigma = 0.1
        mu = 1
        p0_cpa = 3000 #6000*200

        param_optimised,param_covariance_matrix = optimize.curve_fit(gaussianpoisson_pdf_fit,x_hist,y_hist,p0=[max(y_hist),mu,sigma,p0_cpa],maxfev=5000)
        if cpa == 'default':
            cpa = param_optimised[3]

        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
        ax_second_image_2.plot(x_hist_2/param_optimised[3],gaussianpoisson_pdf_fit(x_hist_2,*param_optimised),'r.:',label='Gaussian-Poissonian fit')
        ax_second_image_2.legend()

        #Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax_second_image_2.hist(x_data/param_optimised[3], weights=weights, bins = n_bin)

        #setting the label,title and grid of the plot
        ax_second_image_2.grid("on")
        print('[C, mu, sigma, CPA] =', param_optimised)

    #double_gaussian_fit
    cpa = 2000
    sigma1 = 0.1*cpa
    sigma2 = 0.1*cpa
    mu1 = 0
    mu2 = cpa
    c1 = max(y_hist)
    c2 = c1+2000
    param_optimised,param_covariance_matrix = optimize.curve_fit(double_gaussian_fit,x_hist,y_hist,p0=[c1, mu1, sigma1, c2, mu2, sigma2])

    (c0, mu0, s0, c1, mu1, s1) = param_optimised
    s0 = abs(s0)
    s1 = abs(s1)
    cpa_2 = mu1
    th_2 = threshold(c0, mu0, s0, c1, mu1, s1, cpa_2)
    ff_2 = prob_of_one_atom(c0,s0, c1, s1) #filling fraction
    f_2 = image_fidelity(mu0, s0, mu1, s1, ff_2, th)

    if plot_double_gaussian_fit == True:
        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
        ax_second_image_2.plot(x_hist_2,double_gaussian_fit(x_hist_2,*param_optimised),'r.:',label='Double-Gaussian fit')
        ax_second_image_2.legend()

        #Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax_second_image_2.hist(x_data, weights=weights, bins = n_bin)

        #setting the label,title and grid of the plot
        ax_second_image_2.grid("on")
        ax_second_image_2.set_title(f"second shot")
        print(' [c1, mu1, sigma1, c2, mu2, sigma2] =', param_optimised)
        #print('mu2/sigma2 =', param_optimised[4]/param_optimised[5])

    if print_value == True:
        print('2nd image Probability of one atom = ', prob_of_one_atom(c0,s0, c1, s1))
        print('2nd image Imaging fidelity upper limit =', image_fidelity(mu0, s0, mu1, s1, ff_2, th_2))
        print('2nd image threshold = ', th_2)






    return th, cpa, ff, f



def survival_rate(roi_number_lst, th, site_roi_x):
    first_shot_atom_number = roi_number_lst[0,1:site_roi_x.shape[0]+1,:]
    second_shot_atom_number = roi_number_lst[1,1:site_roi_x.shape[0]+1,:]

    if th.shape[0] > 1:
        th = th[0]

    survival_points = (first_shot_atom_number>th) & (second_shot_atom_number>th)
    appear_points = (first_shot_atom_number<=th) & (second_shot_atom_number>th)
    lost_points = (first_shot_atom_number>th) & (second_shot_atom_number<=th)
    nothing_points = (first_shot_atom_number<=th) & (second_shot_atom_number<=th)


    first_shot_atom_sum_each_roi = np.array([np.sum(first_shot_atom_number[i,:]>th) for i in range(site_roi_x.shape[0])])
    first_shot_no_atom_sum_each_roi = np.array([np.sum(first_shot_atom_number[i,:]<=th) for i in range(site_roi_x.shape[0])])

    survival_sum_each_roi = np.array([np.sum(survival_points[i,:]) for i in range(site_roi_x.shape[0])])
    appear_sum_each_roi = np.array([np.sum(appear_points[i,:]) for i in range(site_roi_x.shape[0])])
    lost_sum_each_roi = np.array([np.sum(lost_points[i,:]) for i in range(site_roi_x.shape[0])])
    nothing_sum_each_roi = np.array([np.sum(nothing_points[i,:]) for i in range(site_roi_x.shape[0])])

    survival_rate_each_roi = survival_sum_each_roi/first_shot_atom_sum_each_roi
    appear_rate_each_roi = appear_sum_each_roi/first_shot_no_atom_sum_each_roi
    lost_rate_each_roi = lost_sum_each_roi/first_shot_atom_sum_each_roi
    fidelity_each_roi = 1 - lost_rate_each_roi - appear_rate_each_roi




    x_arr = (site_roi_x[:,0]+site_roi_x[:,1])/2

    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    (ax1, ax2), (ax3, ax4) = axs
    ax1.plot(x_arr, survival_rate_each_roi,'o')
    ax1.grid()
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel(f'survival rate')
    ax1.set_title(f'average survival rate: {np.mean(survival_rate_each_roi)}')

    ax2.plot(x_arr, appear_rate_each_roi,'o')
    ax2.grid()
    ax2.set_xlabel('x [px]')
    ax2.set_ylabel(f'appear rate')
    ax2.set_title(f'average appear rate: {np.mean(appear_rate_each_roi)}')

    ax3.plot(x_arr, lost_rate_each_roi,'o')
    ax3.grid()
    ax3.set_xlabel('x [px]')
    ax3.set_ylabel(f'lost rate')
    ax3.set_title(f'average lost rate: {np.mean(lost_rate_each_roi)}')

    ax4.plot(x_arr, fidelity_each_roi,'o')
    ax4.grid()
    ax4.set_xlabel('x [px]')
    ax4.set_ylabel(f'fidelity')
    ax4.set_title(f'average fidelity: {np.mean(fidelity_each_roi)}')



    # axs.suptitle(f'average imaging fidelity: {1-np.mean(lost_rate_each_roi)-np.mean(appear_rate_each_roi)} ' )
    plt.show()

while True:
    try:
        folder = askdirectory(title='Select Folder for averaging the tweezer images') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break


folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
site_roi_x_file_path = folder_path + "\\site_roi_x.npy"
site_roi_y_file_path =  folder_path + "\\site_roi_y.npy"
site_roi_x = np.load(site_roi_x_file_path)
site_roi_y = np.load(site_roi_y_file_path)


folder_path = folder
roi_number_lst_file_path = folder_path + "\\roi_number_lst.npy"
th_file_path = folder_path + "\\th.npy"
roi_number_lst = np.load(roi_number_lst_file_path)


site_roi_x_new = np.concatenate([[np.min(site_roi_x, axis = 0)], site_roi_x])
site_roi_y_new = np.concatenate([[np.min(site_roi_y, axis = 0) + 10], site_roi_y])

th, cpa, ff, f = histagram_fit_and_threshold(roi_number_lst, site_roi_x_new, plot_histagram = True, plot_double_gaussian_fit = True, print_value=True, do_neighbour_bkg_sub= False)
# th = np.load(th_file_path)

survival_rate(roi_number_lst, th, site_roi_x)



root = Tk()
root.destroy()










