# -*- coding: utf-8 -*-
"""
Created on Thu March 3 2025

@author: lin
"""
import sys
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
import csv


class bulk_gas_analysis():
    px = 5.5 # Pixels size
    n_px = 2048 # Number of pixels in one direction
    quantum_efficiency = 0.4 # Quantum efficiency
    gain = 1.028 # Gain from line filter FBH850-10
    scattering_rate = 2*np.pi*5.2227e6 # 5.2227 MHz

    def __init__(self):
        pass

    def get_magnification(self, i, o, debug = False):
        self.M = i/o
        return self.M

    def get_counts_per_atom(self, d, f, t, debug = False):
        omega = (d/2)**2/(4*f**2) # solid angle divided by 4pi
        self.counts_per_atom = self.scattering_rate/2*omega*self.quantum_efficiency/self.gain * t # counts per atom
        return self.counts_per_atom


    def load_image_in_lyse(self, debug = False):
        # Is this script being run from within an interactive lyse session?
        if lyse.spinning_top:
            # If so, use the filepath of the current h5_path
            h5_path = lyse.path
        else:
            # If not, get the filepath of the last h5_path of the lyse DataFrame
            df = lyse.data()
            h5_path = df.filepath.iloc[-1]

        self.h5_path = h5_path

        with h5py.File(h5_path, mode='r+') as f:
            g = hz.attributesToDictionary(f['globals'])
            info_dict = hz.getAttributeDict(f)
            images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
            self.images = images
            self.run_number = f.attrs['run number']

        return images

    def get_image_bkg_sub(self, debug = False):
        images = self.images
        image_types = list(images.keys())
        if debug:
            print("image type is", images[image_types[0]])
        MOT_image = images[image_types[0]] # 1st shot is signal
        background_image = images[image_types[1]] # 2nd shot is background
        sub_image = MOT_image - background_image # subtraction of the background

        self.MOT_image = MOT_image
        self.background_image = background_image
        self.sub_image = sub_image

        return sub_image

    def get_images_roi(self, roi, debug = False):
        [roi_x, roi_y, roi_x_bkg, roi_y_bkg] = roi
        self.roi = roi
        sub_image = self.sub_image
        roi_MOT = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]

        self.roi_MOT = roi_MOT
        self.roi_bkg = roi_bkg

        return roi_MOT, roi_bkg

    def get_atom_number(self, debug = False):
        roi_MOT = self.roi_MOT
        roi_bkg = self.roi_bkg
        electron_counts_mot = roi_MOT.sum()
        electron_counts_bkg = roi_bkg.sum()
        atom_number_withbkg = electron_counts_mot / self.counts_per_atom
        bkg_number = electron_counts_bkg / self.counts_per_atom / roi_bkg.size * roi_MOT.size # average bkg floor in the size of roi_MOT
        atom_number = int(atom_number_withbkg) - int(bkg_number)
        self.atom_number = atom_number

        return atom_number

    def plot_images(self, image_scale = 100,debug = False):
        MOT_image = self.MOT_image
        background_image = self.background_image
        roi_MOT = self.roi_MOT
        roi_bkg = self.roi_bkg
        [roi_x, roi_y, roi_x_bkg, roi_y_bkg] = self.roi

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        (ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs
        for ax in axs[0]:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
        for ax in axs[1]:
            ax.set_xlabel('x [um]')
            ax.set_ylabel('y [um]')



        raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax= image_scale)
        ax_mot_raw.set_title('Raw')
        pos = ax_mot_raw.imshow(MOT_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_mot_raw)

        ax_bkg_raw.set_title('Raw, no MOT')
        pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_raw)

        ax_mot_roi.set_title('MOT ROI')
        pos = ax_mot_roi.imshow(
            roi_MOT,
            extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*self.px*self.M,
            **raw_img_color_kw,
        )
        fig.colorbar(pos, ax=ax_mot_roi)

        ax_bkg_roi.set_title('Background ROI')
        pos = ax_bkg_roi.imshow(
            roi_bkg,
            vmin=-10,
            vmax=10,
            extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*self.px*self.M, #factor: px*mag
        )
        fig.colorbar(pos, ax=ax_bkg_roi)

    def save_atom_number(self, debug = False):
        h5_path = self.h5_path
        atom_number = self.atom_number
        run_number = self.run_number

        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        if run_number == 0:
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{atom_number}\n')
        else:
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{atom_number}\n')

    def load_atom_number(self, debug = False):
        h5_path = self.h5_path
        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        with open(count_file_path, newline='') as csvfile:
            counts = [list(map(float, row))[0] for row in csv.reader(csvfile)]

        return counts


    def plot_atom_number(self, debug = False):
        counts = self.load_atom_number()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(counts)
        ax.set_xlabel('Shot number')
        ax.set_ylabel('MOT atom count')

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
