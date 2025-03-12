from dataclasses import dataclass
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import os
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    lyse
except:
    import lyse
from analysis.data import h5lyze as hz
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from image_preprocessor import ImagePreProcessor
from .image_config import AnalysisConfig, ImagingSystem

class BulkGasAnalysis(ImagePreProcessor):
    """Analysis class for bulk gas imaging data.
    
    This class provides functionality for analyzing bulk gas imaging data, including
    ROI-based analysis, background subtraction, and atom number calculations.
    
    The class uses a configuration-based approach where all analysis parameters
    are specified through an AnalysisConfig object, which includes imaging system
    setup, ROI definitions, and analysis parameters.
    """
    
    scattering_rate = 2 * np.pi * 5.2227e+6  # rad/s
    """ Cesium scattering rate """
    # TODO: for what transition? Where did you get this number?

    m = 2.20694650e-25
    """ Cesium mass in kg"""

    kB = 1.3806503e-23
    """J/K Boltzman constant"""

    def __init__(
            self,
            config: AnalysisConfig,
            load_type: str = 'lyse',
            h5_path: str = None
            ):
        """Initialize BulkGasAnalysis with analysis configuration.

        Parameters
        ----------
        config : AnalysisConfig
            Configuration object containing all analysis parameters including:
            - imaging_system: ImagingSystem configuration
            - exposure_time: Imaging exposure time in seconds
            - atoms_roi: ROI for atoms [[x_min, x_max], [y_min, y_max]]
            - bkg_roi: ROI for background [[x_min, x_max], [y_min, y_max]]
            - method: Background subtraction method
        load_type : str, default='lyse'
            Type of loading to perform
             # TODO: what are the options for load_type?
        h5_path : str, optional
            Path to h5 file to load
        """
        if config.exposure_time is None:
            raise ValueError("exposure_time must be provided in AnalysisConfig for bulk gas analysis")
        if config.atoms_roi is None or config.bkg_roi is None:
            raise ValueError("atoms_roi and bkg_roi must be provided in AnalysisConfig for bulk gas analysis")

        # Initialize parent class first
        super().__init__(
            imaging_setup=config.imaging_system,
            load_type=load_type,
            h5_path=h5_path
        )

        # Store config
        self.analysis_config = config

        # Set class-specific attributes
        self.atoms_roi = config.atoms_roi
        self.background_roi = config.bkg_roi
        self.counts_per_atom = self.imaging_setup.counts_per_atom(
            self.scattering_rate,
            config.exposure_time,
        )

        # Process images
        self.atom_image, self.background_image, self.sub_image = self.get_image_bkg_sub()
        self.roi_atoms, self.roi_bkg = self.get_images_roi()

    @staticmethod
    def get_scanning_params_static(h5_path):
        """
        get scanning parameters from globals

        Returns
        -------
        params: list
        """
        with h5py.File(h5_path, mode='r+') as f:
            globals = f['globals']
            params = {}
            for group in globals.keys():
                expansion = hz.getAttributeDict(globals[group]['expansion'])
                for key, value in expansion.items():
                    if value == 'outer' or value == 'inner':
                        global_var = hz.getAttributeDict(globals[group])[key]
                        global_unit = hz.getAttributeDict(globals[group]['units'])[key]
                        params[key] = [global_var, global_unit]

            rep_str =params['n_shots'][0]
            if rep_str[0:2] != 'np':
                rep_str = 'np.' + rep_str
            rep = eval(rep_str)
            n_rep = rep.shape[0]
            del params['n_shots']
            if len(params) == 0:
                params['n_shots'] = [rep_str,'Shots']
                n_rep = 1

        return params, n_rep



    def get_image_bkg_sub(self, debug = False):
        """
        get background subtracted image: signal - background
        atom_image = 1st key of images, background_image = 2nd key of images

        Returns
        -------
        sub_image = atom_image - background_image
        array like, shape [n_px, n_px]
        """
        images = self.images
        image_types = list(images.keys())
        if debug:
            print("image type is", images[image_types[0]])
        atom_image = images[image_types[0]] # 1st shot is signal
        background_image = images[image_types[1]] # 2nd shot is background
        sub_image = atom_image - background_image # subtraction of the background

        # self.atom_image = atom_image
        # self.background_image = background_image
        # self.sub_image = sub_image

        return atom_image, background_image, sub_image

    def get_images_roi(self,):
        """
        get the images and background in the roi defined by atoms_roi and background_roi
        note that the y direction is the row direction of image matrix and
        x direction is the col direction

        Returns
        -------
        roi_atoms, roi_bkg as the images and background in the roi
        array like, shape [roi_y.size, roi_x.size] and [roi_y_bkg.size, roi_x_bkg.size]
        """
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi
        sub_image = self.sub_image
        roi_atoms = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]

        # self.roi_atoms = roi_atoms
        # self.roi_bkg = roi_bkg

        return roi_atoms, roi_bkg

    def get_atom_number(self,):
        """
        sum of counts in roi/counts_per_atom - average background atom per px* roi size

        Returns
        -------
        atom_number: int
        """

        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        electron_counts_atom = roi_atoms.sum()
        electron_counts_bkg = roi_bkg.sum()
        atom_number_withbkg = electron_counts_atom / self.counts_per_atom
        bkg_number = electron_counts_bkg / self.counts_per_atom / roi_bkg.size * roi_atoms.size # average bkg floor in the size of roi_atoms
        atom_number = int(atom_number_withbkg) - int(bkg_number)
        # self.atom_number = atom_number

        self.save_atom_number(atom_number)

        return atom_number

    def gauss2D(self,x, amplitude, mux, muy, sigmax, sigmay, rotation, offset):
        """
        2D Gaussian, see: https://en.wikipedia.org/wiki/Gaussian_function
        Parameters:
        -----------
        x: tuple of 2 arrays, x[0] for x and x[1] for y
        amplitude: amplitude of the gaussian, float
        mux: center of the gaussian in x, float
        muy: center of the gaussian in y, float
        sigmax: width of the gaussian in x, float
        sigmay: width of the gaussian in y, float
        rotation: rotation of the gaussian, float
        offset: offset of the gaussian, float

        Returns
        -------
        G: array like shape[1,:],
          1d ravel of the gaussian
        """
        assert len(x) == 2
        X = x[0]
        Y = x[1]
        A = (np.cos(rotation)**2)/(2*sigmax**2) + (np.sin(rotation)**2)/(2*sigmay**2)
        B = (np.sin(rotation*2))/(4*sigmay**2) - (np.sin(2*rotation))/(4*sigmax**2)
        C = (np.sin(rotation)**2)/(2*sigmax**2) + (np.cos(rotation)**2)/(2*sigmay**2)
        G = amplitude*np.exp(-(A * (X - mux) ** 2 + 2 * B * (X - mux) * (Y - muy) + C * (Y - muy) ** 2)) + offset  # + slopex * X + slopey * Y + offset
        return G.ravel()  # np.ravel() Return a contiguous flattened array.

    def get_atom_gaussian_fit(self, option = "all parameters", gaussian_fit_params = None):
        """
        measure the atom number, temperature of atom through time of flight
        the waist of the cloud is determined through 2D gaussian fitting
        all fitting parameters are saved in "data.csv"

        Parameters
        ----------
        option: str, "all parameters" or "amplitude only"
        "all parameters" will use fitting with all free parameters
        "amplitude only" will use fitting with only free amplitude, the center, waist, rotation and offset is fixed
        gaussian_fit_params: array_like, shape [7]
        only used when option is "amplitude only", the center, waist, rotation and offset is fixed using the values in gaussian_fit_params

        Returns
        -------
        time_of_flight: float
            extracted from global variable "bm_tof_imaging_delay"
        atom_number_gaussian: float
            atom number through gaussian fitting
        guassian_waist: array_like, shape [2]
        temperature: array_like, shape [2]
            instant temperature assuming initial cloud size is 0
            T = m / k_B * (waist/time_of_flight)^2
        """
        # Creating a grid for the Gaussian fit
        [roi_x, roi_y] = self.atoms_roi
        x_size = roi_x[1]-roi_x[0]
        y_size = roi_y[1]-roi_y[0]
        x = np.linspace(0, x_size-1, x_size)
        y = np.linspace(0, y_size-1, y_size)
        x, y = np.meshgrid(x, y)

        # Fitting the 2d Gaussian
        initial_guess = np.array([
            np.max(self.roi_atoms),
            x_size/2,
            y_size/2,
            x_size/2,
            y_size/2,
            0,
            np.min(self.roi_atoms)])

        if option == "all parameters":
            popt, pcov = opt.curve_fit(self.gauss2D, (y, x), self.roi_atoms.ravel(), p0=initial_guess)
            atom_number_gaussian = np.abs(2 * np.pi * popt[0] * popt[3] * popt[4] / self.counts_per_atom)
            sigma = np.sort(np.abs([popt[3], popt[4]]))  # gaussian waiast in pixel, [short axis, long axis]
        elif option == "amplitude only":
            if gaussian_fit_params is None:
                raise ValueError('when choose amplitude only option, '
                'you need to put in the gaussain_fit_params')
            popt, pcov = opt.curve_fit(
            lambda xy, A, offset: self.gauss2D(
                xy, A, *gaussian_fit_params[1:6], offset
            ),
            (y, x),
            self.roi_atoms.ravel(),
            p0=(self.roi_atoms.max(), 0),
            )
            atom_number_gaussian = np.abs(2 * np.pi * (popt[0]-popt[1]) * gaussian_fit_params[3] * gaussian_fit_params[4] / self.counts_per_atom)
            sigma = np.sort(np.abs([gaussian_fit_params[3], gaussian_fit_params[4]]))  # gaussian waiast in pixel, [short axis, long axis]
        else:
            raise NotImplementedError('option should be "all parameters" or "amplitude only"')

        gaussian_waist = np.array(sigma)*self.imaging_setup.pixel_size_before_maginification# convert from pixel to distance m
        time_of_flight = self.globals['bm_tof_imaging_delay']
        temperature = self.m / self.kB * (gaussian_waist/time_of_flight)**2

        self.save_atom_temperature(atom_number_gaussian, time_of_flight, gaussian_waist, temperature)

        return time_of_flight, atom_number_gaussian, gaussian_waist, temperature

    def plot_images(self, raw_image_scale = 100, roi_image_scale = 100):
        """
        plot the images of raw image/background image full frame and background subtracted images in rois
        and also save the images in the folder path

        Parameters
        ----------
        raw_image_scale: float
            scale of raw image in the plot
        roi_image_scale: float
            scale of roi image in the plot

        """
        atom_image = self.atom_image
        background_image = self.background_image
        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        (ax_atom_raw, ax_bkg_raw), (ax_bkg_roi, ax_atom_roi) = axs
        for ax in axs[0]:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
        for ax in axs[1]:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')



        raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax= raw_image_scale)
        ax_atom_raw.set_title('Raw, with atom')
        pos = ax_atom_raw.imshow(atom_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_atom_raw)

        ax_bkg_raw.set_title('Raw, no atom')
        pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_raw)

        pixel_size_before_maginification = self.imaging_setup.pixel_size_before_maginification
        ax_atom_roi.set_title('Atom ROI')
        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax= roi_image_scale)
        pos = ax_atom_roi.imshow(
            roi_atoms,
            extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*pixel_size_before_maginification,
            **roi_img_color_kw,
        )
        fig.colorbar(pos, ax=ax_atom_roi)

        ax_bkg_roi.set_title('Background ROI')
        pos = ax_bkg_roi.imshow(
            roi_bkg,
            vmin=-10,
            vmax=10,
            extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*pixel_size_before_maginification, #factor: px*mag
        )
        fig.colorbar(pos, ax=ax_bkg_roi)
        fig.savefig(self.folder_path+'\\atom_cloud.png')
        return

    def save_atom_number(self, atom_number):
        """
        save atom number to data.csv
        if run number is 0, it will create a new file
        otherwise it will append
        """
        run_number = self.run_number

        count_file_path = self.folder_path+'\\data.csv'
        file_open_mode = 'w' if run_number == 0 else 'a'
        with open(count_file_path, file_open_mode) as f_object:
                f_object.write(f'{atom_number}\n')
        return

    def save_atom_temperature(self, atom_number, time_of_flight, gaussian_waist, temperature):
        """
        save atom temperature to data.csv
        if run number is 0, it will create a new file
        otherwise it will append
        """
        run_number = self.run_number


        count_file_path = self.folder_path+'\\data.csv'
        file_open_mode = 'w' if run_number == 0 else 'a'
        with open(count_file_path, file_open_mode) as f_object:
            f_object.write(f'{atom_number},'
                           f'{time_of_flight},'
                           f'{gaussian_waist[0]},'
                           f'{gaussian_waist[1]},'
                           f'{temperature[0]},'
                           f'{temperature[1]}\n')

        return

    def load_atom_number(self,):
        """
        load the atom number that is already saved in data.csv

        Returns
        -------
        counts: float

        """
        h5_path = self.h5_path
        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        try:
            with open(count_file_path, newline='') as csvfile:
                counts = [list(map(float, row))[0] for row in csv.reader(csvfile)]
        except:
            raise FileExistsError('Please run get_atom_number first')

        return counts


    def plot_atom_number(self,):
        """
        To run this function, we need to run get_atom_number first and have data.csv in the same folder
        plot atom number from data.csv vs the shot number and also save the image in the folder path
        this plot all the shots in the same queue start with run number = 0
        so that we can see the trend

        """
        counts = self.load_atom_number()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(counts,'o-')
        ax.set_xlabel('Shot number')
        ax.set_ylabel('atom count')

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
        fig.savefig(self.folder_path+'\\atom_number.png')

        return

    def load_atom_temperature(self,):
        """
        load the atom number that is already saved in data.csv

        Returns
        -------
        counts: float

        """
        h5_path = self.h5_path
        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        with open(count_file_path, newline='') as csvfile:
            matrix = [list(map(float, row)) for row in csv.reader(csvfile)]
        matrix = np.array(matrix)
        time_of_flight, waist_x, waist_y, temperature_x, temperature_y = [matrix[:,i] for i in np.arange(1,6)]
        return time_of_flight, waist_x, waist_y, temperature_x, temperature_y

    def plot_atom_temperature(self,):
        """
        plot atom temperature from data.csv vs the shot number
        fitting more accurate temperature with different time of flight through linear fit
        since the size of the cloud is not zero at the begining of time of flight
        plot the fitting result
        save all the plots in the folder path
        """
        time_of_flight, waist_x, waist_y, temperature_x, temperature_y = self.load_atom_temperature()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(np.array(temperature_x)*1e6, label='X temperature')
        ax.plot(np.array(temperature_y)*1e6, label='Y temperature')
        ax.set_xlabel('Shot number')
        ax.set_ylabel('Temperature (uK)')
        ax.legend()

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
        fig.savefig(self.folder_path+'\\temperature_single_shot.png')

        if self.run_number >= 2:
            # linear fit to extrac the velocity of the atoms through sphereical expansion
            slope_lst = []
            intercept_lst = []
            slope_uncertainty_lst = []
            for waist in [waist_x, waist_y]:
                coefficient, covariance = np.polyfit(time_of_flight**2, np.array(waist)**2, 1, cov = True)
                slope_lst.append(coefficient[0])
                intercept_lst.append(coefficient[1])
                error = np.sqrt(np.diag(covariance))
                slope_error = error[0]
                slope_uncertainty_lst.append(ufloat(coefficient[0],slope_error))

            slope_lst = np.array(slope_lst)
            intercept_lst = np.array(intercept_lst)
            slope_uncertainty_lst = np.array(slope_uncertainty_lst)
            T_lst = slope_uncertainty_lst*self.m/self.kB*1e6

            t_plot = np.arange(0,max(time_of_flight)+1e-3,1e-3)
            fig2, ax2 = plt.subplots(constrained_layout=True)
            for slope,intercept,T in zip(slope_lst,intercept_lst,T_lst):
                ax2.plot(t_plot**2*1e6 , (slope*t_plot**2 + intercept)*1e12, label=rf'$T = {T:LS}$ $\mu$K')

            i=0
            for waist in [waist_x, waist_y]:
                ax2.plot(time_of_flight**2*1e6, np.array(waist)**2*1e12, 'oC'+str(i), label = 'data')
                i+=1

            ax2.set_xlabel('Shot number')
            ax2.set_xlabel(r'Time$^2$ (ms$^2$)')
            ax2.set_ylabel(r'$\sigma^2$ ($\mu$m$^2$)')
            ax2.legend()
            ax2.set_ylim(ymin=0)
            ax2.tick_params(axis = 'x')
            ax2.tick_params(axis = 'y')

            ax2.grid(color='0.7', which='major')
            ax2.grid(color='0.9', which='minor')
            fig2.savefig(self.folder_path+'\\temperature_fit.png')
        return

    def plot_amplitude_vs_parameter(self):
        """
        plot amplitude vs parameter that is scanned, the image will be saved in the folder path

        the amplitude will be first column of data.csv, eg. atom number, survival rate, etc
        the data will be plotted against the first parameter for now

        this can be modified for general use with multiple repetitions and multiple parameters

        """
        amplitude = self.load_atom_number()
        fig, ax = plt.subplots(constrained_layout=True)

        for key, value in self.params.items():
            para = eval(value[0])
            unit = value[1]
            label_string = f'{key} ({unit})'

        if self.n_rep > 1:
            #TODO support multiple repetitions
            raise NotImplementedError("multiple repetitions is not supported yet")

        if len(self.params.keys()) == 1:
            # 1D scanning case
            amplitude_resize = np.resize(amplitude, para.shape[0])
            amplitude_resize[len(amplitude):] = np.nan
        else:
            #TODO support 2D scanning
            raise NotImplementedError("2 scanning parameters is not supported yet")

        ax.plot(para, amplitude_resize, '-o')
        ax.set_xlabel(label_string)
        ax.set_ylabel('amplitude')
        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')

        fig.savefig(self.folder_path+'\\amplitude_vs_parameter.png')
