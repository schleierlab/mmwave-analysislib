# -*- coding: utf-8 -*-
"""
Created on Aug 16th 2024

@author: Michelle Wu
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
# from tweezer_imaging_fidelity_measurement_alternating_bkg import survival_rate

import scipy.optimize as optimize
from scipy.optimize import curve_fit
from scipy.stats import poisson
import scipy.special as special


def avg_shots_multi_roi_avg_bkg_sub(folder, site_roi_y, site_roi_x, avg_bkg_img, shots = 'defult', loop = True, plot_single_shots = False, image_scale = 'default'):
    '''
    set_threshold = 'default' means not setting threshold
    '''
    h = roi_y[1]-roi_y[0]

    data = np.zeros(shape=(2,h,2400))

    n_shots = np.size([i for i in os.listdir(folder) if i.endswith('.h5')])

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
        # print(image_types)

        # image_1 = images[image_types[0]]
        # # print(np.shape(image_1))
        # image_2 = images[image_types[1]] #2nd shot the camera take
        # sub_image1 = image_1 - avg_bkg_img[0]
        # sub_image2 = image_2 - avg_bkg_img[1]

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

def plot_shots_avg(n_shots,site_roi_x,site_roi_y,data, show_roi = True):
    site_roi_x = site_roi_x - roi_x[0]

    fig, axs = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'{n_shots} shots average, Mag = 7.424, Pixel = 0.87 um')

    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        rect.append(patches.Rectangle((site_roi_x[i,0], site_roi_y[i,0]), site_roi_x[i,1]-site_roi_x[i,0], site_roi_y[i,1]-site_roi_y[i,0], linewidth=1, edgecolor='r', facecolor='none'))
        # print(site_roi_x[i,0], site_roi_y[i,0])

    axs.set_xlabel('x [px]')
    axs.set_ylabel('y [px]')
    image_scale = np.amax(data[:, roi_x[0]:roi_x[1]]) # 12 bit dept
    # print(np.shape(data))
    raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax=image_scale)

    pos = axs.imshow(data[:, roi_x[0]:roi_x[1]], **raw_img_color_kw)
    fig.colorbar(pos, ax=axs)
    if show_roi == True:
        axs.add_collection(PatchCollection(rect, match_original=True))
        # print('show roi')
    plt.show()

def double_gaussian_fit( x, c1, mu1, sigma1, c2, mu2, sigma2 ):
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def gaussianpoisson_pdf_fit(X, C, mu, sigma, CPA):
    kmax = 20
    return sum([C*np.exp(-(X/CPA-k)**2/(2*sigma**2))*poisson.pmf(k, mu) for k in np.arange(0, kmax)])

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
    # Lin's way of calculating fidelity, see wiki:
    # sigma12_sq = sigma1**2 + sigma2**2
    # fidelity = 1-np.exp(-(mu1-mu2)**2/2/sigma12_sq)/np.sqrt(2*np.pi*sigma12_sq)

    # P0(x < th) and P1( x > th )
    cdf0 = 0.5*(1+special.erf((th-mu1)/(np.sqrt(2)*sigma1))) #integrate from -infinity to th
    cdf1 = 0.5*(1+special.erf((th-mu2)/(np.sqrt(2)*sigma2)))
    p = (1-ff)*(1-cdf0)+ff*cdf1
    fidelity = 1-p
    return fidelity

def histagram_fit_and_threshold(roi_number_lst, site_roi_x, plot_histagram = False, plot_double_gaussian_fit = False, plot_gaussian_poisson_fit = False, sub_bkg = False, cpa = 'default', print_value = False):

    bkg_number_lst = roi_number_lst[0,:]

    if sub_bkg == True:
        bkg_mean = np.mean(bkg_number_lst)
        print(bkg_mean)
    else:
        bkg_mean = 0

    all_roi_number_lst = roi_number_lst[1:site_roi_x.shape[0],:].flatten()
    # print(site_roi_x.shape[0])
    # print(np.shape(roi_number_lst))
    # print(np.shape(all_roi_number_lst))
    # print(np.shape(bkg_number_lst))

    if plot_histagram == True:
        plt.hist(all_roi_number_lst-bkg_mean, bins = 250, label = f'all sites')
        plt.hist(bkg_number_lst-bkg_mean, bins = 10, label = f'0th site, bkg')
        plt.legend()
        #plt.xlim([-5000,10000])
        plt.xlabel('counts')
        plt.ylabel('frequency')
        plt.show()

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
    cpa = 1500
    sigma1 = 0.1*cpa
    sigma2 = 0.1*cpa
    mu1 = 10 #0
    mu2 = cpa
    c1 = max(y_hist)
    c2 = c1+cpa
    param_optimised,param_covariance_matrix = optimize.curve_fit(double_gaussian_fit,x_hist,y_hist,p0=[c1, mu1, sigma1, c2, mu2, sigma2])

    (c0, mu0, s0, c1, mu1, s1) = param_optimised
    s0 = abs(s0)
    s1 = abs(s1)
    cpa = mu1
    th = threshold(c0, mu0, s0, c1, mu1, s1, cpa)
    ff = prob_of_one_atom(c0,s0, c1, s1) #filling fraction
    f = image_fidelity(mu0, s0, mu1, s1, ff, th)

    # fig_2, axs_2 = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    # (ax_first_image_2, ax_second_image_2) = axs_2

    # for ax in axs_2:
    #         ax.set_xlabel('counts')
    #         ax.set_ylabel('Probability')

    # if plot_double_gaussian_fit == True:
    #     print(' [c1, mu1, sigma1, c2, mu2, sigma2] =', param_optimised)
    #     x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
    #     ax_first_image_2.plot(x_hist_2,double_gaussian_fit(x_hist_2,*param_optimised),'r.:',label='Double-Gaussian fit')
    #     ax_first_image_2.legend()

    #     #Normalise the histogram values
    #     weights = np.ones_like(x_data) / len(x_data)
    #     ax_first_image_2.hist(x_data, weights=weights, bins = n_bin)

    #     #setting the label,title and grid of the plot
    #     ax_first_image_2.grid("on")
    #     ax_first_image_2.set_title(f"first shot,p = {ff:.2f},\n th = {th[0]:.1f}, f = {f[0]:.4f}")

    if plot_double_gaussian_fit == True:
        fig = plt.figure()
        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
        plt.plot(x_hist_2,double_gaussian_fit(x_hist_2,*param_optimised),'r.:',label='Double-Gaussian fit')
        plt.legend()

        #Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        plt.hist(x_data, weights=weights, bins = n_bin)

        #setting the label,title and grid of the plot
        plt.xlabel("Atoms")
        plt.ylabel("Probability")
        plt.grid("on")
        plt.title(f"atom roi")
        plt.show()
        print(' [c1, mu1, sigma1, c2, mu2, sigma2] =', param_optimised)
        #print('mu2/sigma2 =', param_optimised[4]/param_optimised[5])

    if print_value == True:
        print('Probability of one atom = ', prob_of_one_atom(c0,s0, c1, s1))
        print('Imaging fidelity =', image_fidelity(mu0, s0, mu1, s1, ff, th))
        print('threshold = ', th)

    return th, cpa, ff, f

def rearrangement_success_rate(roi_number_lst, thold, site_roi_x, site_roi_y, n_shots, target_array, shots = 'defult',loop = False, plot = True): #thold means threshold
    target_array = np.array(target_array)
    n_target = len(target_array)

    if shots == 'defult':
        shots = n_shots

    if loop == True:
        N = int(n_shots/2)
    else:
        N = n_shots

    n = len(site_roi_x)-1
    surv_arr = np.zeros((n, shots))
    lost_arr = np.zeros((n, shots))
    appear_arr = np.zeros((n, shots))
    nothing_arr = np.zeros((n, shots))

    rearrange_shot = [] # shots where rearrangement happens
    for cnt in (np.arange(shots)): # cnt is shot number
        above_thold_1 = 0
        for i in np.arange(site_roi_x.shape[0])[1:]: #exclude the first site because it is the background site
            electron_counts1 = roi_number_lst[0, i, cnt]
            electron_counts2 = roi_number_lst[1, i, cnt]
            if electron_counts1> thold:
                above_thold_1 += 1
            if electron_counts1> thold and electron_counts2> thold:
                surv_arr[i-1,cnt] = 1
            elif electron_counts1> thold and electron_counts2<= thold:
                lost_arr[i-1,cnt] = 1
            elif electron_counts1<= thold and electron_counts2> thold:
                appear_arr[i-1,cnt] = 1
            elif electron_counts1<= thold and electron_counts2<= thold:
                nothing_arr[i-1,cnt] = 1
        if above_thold_1 >= len(target_array):
            rearrange_shot.append(cnt)

    # atom loading rate for each site, suming over different shots
    tot_surv = np.sum(surv_arr, axis = 1)
    tot_lost = np.sum(lost_arr, axis = 1)
    tot_appear = np.sum(appear_arr, axis = 1)
    tot_nothing = np.sum(nothing_arr, axis = 1)
    total = tot_surv+tot_lost+tot_appear+tot_nothing

    loading1 = (tot_surv+tot_lost)/total # atoms appear on 1st shot
    loading2 = (tot_surv+tot_appear)/total # atoms appear on 2nd shot
    # rearrangement success rate calculation
    success_number_lst = []
    success_arr = np.zeros((n_target, shots)) # success rearrange for each site, each shot
    fail_arr = np.zeros((n_target, shots))
    for cnt in rearrange_shot:
        for i in target_array:
            success_arr[i,cnt] = surv_arr[i,cnt] + appear_arr[i,cnt] # successful rearrangement
            fail_arr[i,cnt] = lost_arr[i,cnt] + nothing_arr[i,cnt] # unsuccessful rearrangement. lost atom or no atom
        success_number = np.sum(success_arr[:,cnt])
        success_number_lst.append(success_number)

    # Number to count
    # n_target = len(target_array)
    # Count occurrences of the number
    success_rearrange = success_number_lst.count(n_target)
    # print(success_number_lst)

    # site success rate
    tot_success = np.sum(success_arr, axis = 1)
    tot_fail = np.sum(fail_arr, axis = 1)
    total = tot_success + tot_fail
    # print(total)
    # print(tot_success)

    site_success_rate = np.zeros(len(total))
    site_success_rate_lst = []
    for i in np.arange(n_target):
        site_success_rate[i] = tot_success[i]/total[i]
        site_success_rate_lst.append(site_success_rate[i])

    # where_0 = np.where(total == 0)
    # total[where_0] = 1 # avoid division by zero
    # site_success_rate = np.divide(tot_success, total) #success rate for each site
    avg_site_success_rate = np.mean(site_success_rate_lst)

    print('rearrangement shots ratio:', len(rearrange_shot)/shots)
    print('rearrange success rate:',success_rearrange/len(rearrange_shot))
    print('average success rate for each site:',avg_site_success_rate)
    # print('average survival rate in rearrangement:',np.sum(surv_rate_arr)/len(surv_rate_arr))
    # print('average new appear rate in rearrangement:',np.sum(appear_rate_arr)/len(appear_rate_arr))

    if plot == True:
        fig, axs = plt.subplots(nrows=1, ncols=1)
        site_roi_x = site_roi_x - roi_x[0]
        n_site_roi_x = site_roi_x[1:]
        # n_site_roi_y = site_roi_y[1:]
        x_arr = np.sum(n_site_roi_x, axis = 1)/2

        n_site_roi_x_target = []
        for i in target_array:
            n_site_roi_x_target.append(n_site_roi_x[i])
        n_site_roi_x_target = np.array(n_site_roi_x_target)
        x_arr_target = np.sum(n_site_roi_x_target, axis = 1)/2

        axs.plot(x_arr, loading1,'o', label = '1st shot')#'before rearrangement')
        axs.plot(x_arr, loading2,'o', label = '2nd shot')#'after rearrangement') # we can't say this is after rearrangement because sometimes rearrangement doesn't happen
        axs.grid()
        axs.legend()
        axs.set_xlabel('x [px]')
        axs.set_ylabel(f'tweezer loading rate, {n_shots} shots average')
        plt.show()

        # Calculate the frequency of each unique element
        unique_elements, counts = np.unique(success_number_lst, return_counts=True)
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.bar(unique_elements, counts, width=0.5) # Create a bar plot
        axs.grid(axis = 'y')
        axs.set_xlabel('Number of sites after rearrangement')
        axs.set_ylabel('Frequency')
        axs.set_title(f'{n_target} target sites, \n rearrange shot ratio: {len(rearrange_shot)/shots} , success rate: {success_rearrange/len(rearrange_shot):.3f}')
        axs.set_xticks(unique_elements) # Adjust the x-axis ticks to be at the center of the bars
        plt.show()

        # site success rate
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.plot(x_arr_target, site_success_rate,'o')
        axs.grid()
        axs.set_xlabel('x [px]')
        axs.set_ylabel(f'rearrangement success rate, {n_shots} shots')
        axs.set_title(f'target sites success rate avg: {avg_site_success_rate:.3f}')
        plt.show()



while True:
    try:
        folder = askdirectory(title='Select Folder for averaging the tweezer images') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break

cnt = 0 # check the first shot (In this case, we are getting the globals that are the same for all shots, so it shouldnt matter which shot we checko. As long as they are in the same folder)
string = glob.glob(folder + f'\*{cnt}.h5')
h5_path = string[0]

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    info_dict = hz.getAttributeDict(f)
    kinetix_roi_row= np.array(hz.attributesToDictionary(f).get('globals').get('kinetix_roi_row'))
    target_array = np.array(hz.attributesToDictionary(f).get('globals').get('TW_target_array'))
    #print(f'run_number = {run_number} ')

print('target array:', target_array)
roi_y = [kinetix_roi_row[0], kinetix_roi_row[0]+kinetix_roi_row[1]]
load_roi = True # this will load roi_x, site_roi_x, and site_roi_y

folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'

if load_roi == True:
    site_roi_x_file_path = folder_path + "\\site_roi_x.npy"
    site_roi_y_file_path =  folder_path + "\\site_roi_y.npy"
    roi_x_file_path =  folder_path + "\\roi_x.npy"
    roi_x = np.load(roi_x_file_path)
    site_roi_x = np.load(site_roi_x_file_path)
    site_roi_y = np.load(site_roi_y_file_path)

# arrange site_roi_x from big to small (then add the background roi)
# Get the indices that would sort the array based on the 0th element
sorted_indices = np.argsort(site_roi_x[:, 0])[::-1]

# Use the indices to sort the array
site_roi_x = site_roi_x[sorted_indices]
site_roi_y = site_roi_y[sorted_indices]

# add background roi
site_roi_x = np.insert(site_roi_x, 0, site_roi_x[-1], axis=0)
site_roi_y = np.insert(site_roi_y, 0, site_roi_y[-1]+12, axis=0)

print(f'site_roi_x={site_roi_x}, site_roi_y={site_roi_y}')

#load background image
avg_shot_bkg_file_path =  folder_path + "\\avg_shot_bkg.npy"
avg_bkg_img = np.load(avg_shot_bkg_file_path)

(data, roi_number_lst, n_shots) = avg_shots_multi_roi_avg_bkg_sub(folder, site_roi_y, site_roi_x, avg_bkg_img, loop = False)

plot_shots_avg(n_shots,site_roi_x,site_roi_y,data[0])
th, cpa, ff, f = histagram_fit_and_threshold(roi_number_lst[0], site_roi_x, plot_histagram = False, plot_double_gaussian_fit = True, print_value=True)

rearrangement_success_rate(roi_number_lst, th, site_roi_x, site_roi_y, n_shots, target_array, shots = 'defult',loop = False, plot = True)

# folder_path = folder
# roi_number_lst_file_path = folder_path + "\\roi_number_lst.npy"
th_file_path = folder_path + "\\th.npy"
# np.save(roi_number_lst_file_path, roi_number_lst)
np.save(th_file_path, th)

root = Tk() # try to close the small window. Works when run in terminal, but somehow doesn't work on lyse...
root.destroy()









