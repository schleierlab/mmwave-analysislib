import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp

from tkinter import Tk
from tkinter.filedialog import askdirectory
import easygui
from pathlib import Path

import sys
root_path = r"X:\userlib\analysislib"
#root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"

if root_path not in sys.path:
    sys.path.append(root_path)

try:
    lyse
except:
    import lyse

from tweezer_imaging_fidelity_measurement_avg_bkg_sub_sweep_parameters import histagram_fit_and_threshold as hft

while True:
    try:
        folder = askdirectory(title='Select Folder for averaging the tweezer images') # shows dialog box and return the path
        print(folder)

    except:
        continue
    break

def lorentzian(x, x0, w, a, offset):
    return offset + a * w/2/np.pi/((w/2)**2 + (x - x0)**2)


# # Reading in sweep parameters
# msg = "Enter the detunings, sweep parameters, and sweep param names"
# title = "Input"
# fieldNames = ["numpy array", "numpy array", "str"]
# fieldValues = []  # we start with blanks for the values
# fieldValues = easygui.multenterbox(msg,title, fieldNames)

# # make sure that none of the fields was left blank
# while 1:
#     if fieldValues == None: break
#     errmsg = ""
#     for i in range(len(fieldNames)):
#       if fieldValues[i].strip() == "":
#         errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
#     if errmsg == "": break # no problems found
#     fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
# #print("Reply was:", fieldValues)

# dets, sweep_params, sweep_param_name = (eval(fieldValues[0]), eval(fieldValues[1]), fieldValues[2])

dets = np.linspace(1.725, 1.925, 41)
sweep_params = np.array([3e-3, 5e-3, 10e-3])
sweep_param_name = "mw_field_wait_dur (ms)"




# Reading in roi_number_lst
folder_path = folder
roi_number_lst_file_path = folder_path + "\\roi_number_lst.npy"
roi_number_lst = np.load(roi_number_lst_file_path)

folder_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
site_roi_x_file_path = folder_path + "\\site_roi_x.npy"
site_roi_y_file_path =  folder_path + "\\site_roi_y.npy"

site_roi_x = np.load(site_roi_x_file_path)
site_roi_y = np.load(site_roi_y_file_path)
site_roi_x_new = np.concatenate([[np.min(site_roi_x, axis = 0)], site_roi_x])
site_roi_y_new = np.concatenate([[np.min(site_roi_y, axis = 0) + 10], site_roi_y])
print(f'site_roi_x={site_roi_x_new}, site_roi_y={site_roi_y_new}')


# Calculating threshold
# TODO: refactor this and handle poor fitting when the cpa (counts per atom) is wrong
th, cpa, ff, f = hft(roi_number_lst, site_roi_x_new, plot_histagram = True, plot_double_gaussian_fit = True, print_value=True)
print(f'th = {th}, cpa = {cpa}, ff = {ff}, f = {f}')

# Reshaping roi_number_lst
reshaped = np.array(roi_number_lst).transpose(2,0,1)

n_dets = np.shape(dets)[0]
actual_dets = dets #dets[1:int(n_dets / 2)]
n_params = np.shape(sweep_params)[0]
n_rois = np.shape(reshaped)[-1]
n_reps = int(np.shape(reshaped)[0] / (np.shape(dets)[0] * np.shape(sweep_params)[0]))



#Finding atom counts
atom_counts = []
for shot in reshaped:
    shot_1 = shot[0]>th
    shot_2 = shot[1]>th
    atom_counts.append([shot_1*1, shot_2*1])

atom_counts = np.array(atom_counts).reshape(n_reps, n_params, n_dets, 2, n_rois)
atom_counts = atom_counts.transpose(1,2,3,0,4)

#Summing atom counts over shots with same params and using a rough asymmetric error bar
spectra = []
p_tildes = []
error_bars = []
for i in range(n_params):
    spectrum = []
    p_tilde = []
    error_bar = []
    for j in range(n_dets):
        # first_shot = (sum(sum(atom_counts[i][j][0])) + sum(sum(atom_counts[i][n_dets - j][0]))) / 2
        # second_shot = (sum(sum(atom_counts[i][j][1])) + sum(sum(atom_counts[i][n_dets - j][1]))) / 2
        first_shot = sum(sum(atom_counts[i][j][0]))
        second_shot = sum(sum(atom_counts[i][j][1]))
        spectrum.append(second_shot / first_shot)

        p_t = (second_shot + 1) / (first_shot + 2)
        e_bar = np.sqrt(p_t * (1-p_t) / first_shot)

        p_tilde.append(p_t)
        error_bar.append(e_bar)

    # plt.plot(actual_dets, spectrum, f"C{i}.")
    # plt.errorbar(actual_dets, p_tilde, yerr=error_bar, fmt=' ', ecolor= f"C{i}")

    spectra.append(spectrum)
    p_tildes.append(p_tilde)
    error_bars.append(error_bar)

#Fitting each spectrum
center_guess = np.average(actual_dets)
freqs = actual_dets - center_guess #center_guess - actual_dets


upopts = []
popts = []
pconvs = []
x0s = []
offsets = []
for i, spectrum in enumerate(spectra):
    indices = np.argsort(spectrum, axis=0)
    guess = [np.mean(freqs[indices[-2:]]),
            0.02,
            (max(spectrum) - min(spectrum))*0.02*np.pi/2,
            min(spectrum)]
    bounds_setting =([np.min(freqs), 0, 0], [np.max(freqs),np.max(freqs)-np.min(freqs), np.pi/2*(np.max(freqs)-np.min(freqs))])
    popt, pconv = curve_fit(
        lambda x, x0, w, a: lorentzian(x, x0, w, a, guess[3]),
        freqs,
        spectrum,
        p0=guess[0:3],
        bounds = bounds_setting,
    )
    popts.append(popt)
    pconvs.append(pconv)
    upopts.append(uncertainties.correlated_values(popt, pconv))
    offsets.append(guess[3])


gammas = np.abs(np.array(upopts).transpose()[1])
x0s = np.array(upopts).transpose()[0]
# offsets = np.array(upopts).transpose()[3]
depths = np.abs(offsets + np.array(upopts).transpose()[2]/(np.pi * gammas / 2))

plt.figure()
# freqs = freqs - np.average(unp.nominal_values(x0s))
freqs_smooth = np.linspace(freqs[0], freqs[-1], 500)
labels = sweep_params

for i, spectrum in enumerate(spectra):
    indices = np.argsort(spectrum, axis=0)
    guess = [np.mean(freqs[indices[-2:]]),
    0.02,
    (max(spectrum) - min(spectrum))*0.02*1.5,
    min(spectrum)]
    plt.plot(freqs, spectrum, f"C{i}.", label = f"{sweep_param_name} = {labels[i]}")
    plt.errorbar(freqs, p_tildes[i], yerr=error_bars[i], fmt=' ', ecolor= f"C{i}")
    plt.plot(freqs_smooth, lorentzian(freqs_smooth, *popts[i], guess[3]), f"C{i}", label = f"linewdith: ${gammas[i]*1e+3:LS}$ kHz")
    # plt.plot(freqs_smooth, lorentzian(freqs_smooth, *guess), f"C{i}")

plt.legend()
plt.xlabel(f'Frequency from resonance [MHz]')
plt.ylabel("Survival rate")


#Plotting fits of each spectrum
plt.figure()
labels = sweep_params*1e3

x_axis = sweep_param_name

fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True)

axs[0,0].errorbar(labels, unp.nominal_values(gammas), yerr=unp.std_devs(gammas))
axs[0,0].set(ylabel = "FWHM [MHz]")
axs[0,0].set_xlim([0, 1.05*max(labels)])
axs[0,0].set_ylim([0, max(unp.nominal_values(gammas)) + max(unp.std_devs(gammas))])
axs[0,0].set_title("FWHM fits")

axs[0,1].errorbar(labels, unp.nominal_values(x0s), yerr=unp.std_devs(x0s))
axs[0,1].set(ylabel = "Center [MHz]")
axs[0,1].set_xlim([0, 1.05*max(labels)])
y_max = max(np.abs(unp.nominal_values(x0s))) + max(unp.std_devs(x0s))
axs[0,1].set_ylim([-y_max, y_max])
axs[0,1].set_title("Center frequency fits")

axs[1,1].errorbar(labels, unp.nominal_values(offsets), yerr=unp.std_devs(offsets))
axs[1,1].set(xlabel = x_axis, ylabel = "Off-resonant surival rate")
axs[1,1].set_xlim([0, 1.05*max(labels)])
# axs[1,1].set_ylim([0.9, max(unp.nominal_values(offsets)) + max(unp.std_devs(offsets))])
axs[1,1].set_title("Offset fits")

axs[1,0].errorbar(labels, unp.nominal_values(depths), yerr=unp.std_devs(depths))
axs[1,0].set(xlabel = x_axis, ylabel = "On-resonant surival rate")
axs[1,0].set_xlim([0, 1.05*max(labels)])
axs[1,0].set_ylim([0, max(unp.nominal_values(depths)) + max(unp.std_devs(depths))])
axs[1,0].set_title("On-resonant survival from fits")

plt.tight_layout()
