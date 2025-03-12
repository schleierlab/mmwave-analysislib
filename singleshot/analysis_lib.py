# -*- coding: utf-8 -*-
"""
Created on Thu March 3 2025

@author: lin, tony
"""
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
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


@dataclass
class ImagingCamera:
    pixel_size: float
    """Pixel size, in m"""

    image_size: float
    """Size of sensor, in pixels (assumes square sensor)"""

    quantum_efficiency: float
    """Quantum efficiency of the camera"""

    gain: float
    """
    Additional conversion from detected electron counts to camera counts,
    (for cameras like EMCCDs).
    """

    image_name: str
    """
    Name of the image file to load.
    """


@dataclass
class ImagingSetup:
    imaging_f: float
    """Focal length of imaging lens, in m"""

    objective_f: float
    """Focal length of objective lens, in m"""

    lens_diameter: float  # diameter of light collection lenses
    """Diameter of imaging and objective lenses (or the smaller of the two), in m"""

    imaging_loss: float
    """Fractional power loss along imaging path."""

    camera: ImagingCamera
    """Detection camera"""

    @property
    def magnification(self):
        return self.imaging_f / self.objective_f

    @property
    def pixel_size_before_maginification(self):
        return  self.camera.pixel_size/self.magnification

    @property
    def solid_angle_fraction(self):
        """
        Solid angle captured by the imaging setup as a fraction of 4\pi.
        Makes a small angle approximation for the imaging NA.
        """
        return (self.lens_diameter / 2)**2 / (4 * self.imaging_f**2)

    def counts_per_atom(self, scattering_rate, exposure_time):
        """
        For an image of atoms with given scattering rate taken with a given exposure time,
        find the camera counts per atom we expect with this imaging setup.
        Note that the scattering rate is for resonance light not for the detuned light.

        Parameters
        ----------
        scattering_rate: float
            Scattering rate (\Gamma) for atoms being imaged.
        exposure_time: float
            Exposure time of the image in seconds.

        Returns
        -------
        float
            Camera counts per atom for this imaging setup.
        """
        count_rate_per_atom = (
            scattering_rate/2
            * self.solid_angle_fraction
            * self.imaging_loss
            * self.camera.quantum_efficiency
            * self.camera.gain
        )
        return count_rate_per_atom * exposure_time


manta_camera = ImagingCamera(
    pixel_size=5.5e-6,
    image_size=2048,
    quantum_efficiency=0.4,
    gain=1,
    image_name='manta419b_mot_images'
)


manta_setup = ImagingSetup(
    imaging_f=50e-3,
    objective_f=125e-3,
    lens_diameter=25.4e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=manta_camera,
)

kinetix_camera = ImagingCamera(
    pixel_size=6.5e-6,
    image_size=2400,
    quantum_efficiency=0.58,
    gain=1,
    image_name='kinetix_images'
)

kinetix_setup = ImagingSetup(
    imaging_f=40.4e-3,
    objective_f=300e-3,
    lens_diameter=50.8e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=kinetix_camera,
)

class ImagePreProcessor:
    def __init__(self, imaging_setup: ImagingSetup, load_type: str = 'lyse', h5_path: Optional[str] = None):
        """Initialize image preprocessing.
        
        Parameters
        ----------
        imaging_setup : ImagingSetup
            Imaging setup configuration
        load_type : str, default='lyse'
            'lyse' for h5 file active in lyse, 'h5' for h5 file with input h5_path
        h5_path : str, optional
            Path to h5 file, only used if load_type='h5'
        """
        self.imaging_setup = imaging_setup
        self.h5_path, self.folder_path = self.get_h5_path(load_type=load_type, h5_path=h5_path)
        self.params, self.n_rep = self.get_scanning_params()
        self.images, self.run_number, self.globals = self.load_images()
    
    def get_h5_path(self, load_type, h5_path):
        """
        get h5_path based on load_type
        Parameters
        ----------
        load_type: str
            'lyse' for h5 file active in lyse, 'h5' for h5 file with input h5_path
        h5_path: str
            path to h5 file, only used if load_type='h5'

        Returns
        -------
        h5_path: str
        The actual h5 file path used based on the selected load_type
        """
        if load_type == 'lyse':
            # Is this script being run from within an interactive lyse session?
            if lyse.spinning_top:
                # If so, use the filepath of the current h5_path
                h5_path = lyse.path
            else:
                # If not, get the filepath of the last h5_path of the lyse DataFrame
                df = lyse.data()
                h5_path = df.filepath.iloc[-1]
        elif load_type == 'h5':
            if h5_path is None:
                raise ValueError("When load_type is h5, please provide exact h5 path")
            h5_path = h5_path
        else:
            raise ValueError("load_type must be 'lyse' or 'h5'")

        # Get folder path from h5 path
        folder_path = os.path.dirname(h5_path)

        return h5_path, folder_path

    def get_scanning_params(self,):
        """
        get scanning parameters and number of repetitions based
        on the expansion of each globals, if the expansion is 'inner' or 'outer'
        then it is a scanning parameter, number of repetitions is determined by n_shots
        in globals. If only n_shots is present, then n_rep = 1 and scanning params = n_shots

        Returns
        -------
        params: list
            list of scanning parameters
            list.key = parameter name, list.value = [parameter value, unit]
        n_rep: int
            number of repetitions
        """
        h5_path = self.h5_path
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

    def load_images(self,):
        """
        load image inside the h5 file, return current run number and globals

        Returns
        -------
            images: dictionary with keys based on different imaging cameras used
                images.keys(): list of all keys of images, which is based on different imaging cameras
                images.values(): list of all values of images, array like, shape [n_px, n_px]
            run_number: int
                current run number
            globals: dictionary with keys based on different global variables used

        """
        with h5py.File(self.h5_path, mode='r+') as f:
            globals = hz.getGlobalsFromFile(self.h5_path)
            images = hz.datasetsToDictionary(f[self.imaging_setup.camera.image_name], recursive=True)
            run_number = f.attrs['run number']
        return images, run_number, globals

class BulkGasAnalysis(ImagePreProcessor):
    scattering_rate = 2 * np.pi * 5.2227e+6 # rad/s
    """ Cesium scattering rate """

    m = 2.20694650e-25
    """ Cesium mass in kg"""

    kB = 1.3806503e-23
    """J/K Boltzman constant"""

    def __init__(
            self,
            imaging_setup: ImagingSetup,
            exposure_time,
            atoms_roi,
            bkg_roi,
            load_type='lyse',
            h5_path=None
            ):
        """
        Parameters
        ----------
        imaging_setup: ImagingSetup class
         chose manta_path or kinetix_path for the current imaging setup
        exposure_time: float
            imaging exposure time in s
        atoms_roi, bkg_roi: array_like, shape (2, 2)
            Specification of the regions of interest for the atoms and for the background image.
            The specification should take the form
            [
                [x_min, x_max],
                [y_min, y_max],
            ],
            where all numbers are given in pixels.
        load_type: str
            'lyse' for h5 file active in lyse, 'h5' for h5 file with input h5_path
        h5_path: str
            path to h5 file, only used if load_type='h5'
        """
        # Initialize parent class first
        super().__init__(
            imaging_setup=imaging_setup,
            load_type=load_type,
            h5_path=h5_path
        )

        # Set class-specific attributes
        self.atoms_roi = atoms_roi
        self.background_roi = bkg_roi
        self.counts_per_atom = self.imaging_setup.counts_per_atom(
            self.scattering_rate,
            exposure_time,
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


@dataclass
class ROIConfig:
    """Configuration for ROIs in tweezer analysis.

    Parameters
    ----------
    roi_x : List[int]
        X-coordinates [start, end] for atom imaging region
        Only required when load_roi=False
    site_roi : dict,
        Dictionary containing site ROI coordinates
        Only required when load_roi=False
        Must have 'site_roi_x' and 'site_roi_y' arrays
    """
    def __init__(self, roi_x: List[int] = None, site_roi: dict = None):
        self.roi_x = roi_x
        self.site_roi = site_roi

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ROIConfig":
        """Load ROI configuration from YAML file.
        
        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file
            
        Returns
        -------
        ROIConfig
            ROI configuration object
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract ROI coordinates if present
        roi_x = config.get('roi_x', None)
        site_roi = config.get('site_roi', None)

        return cls(roi_x=roi_x, site_roi=site_roi)


class TweezerAnalysis(ImagePreProcessor):

    def __init__(
            self,
            imaging_setup: ImagingSetup,
            method: str = 'average',
            bkg_roi_x: List[int] = [1900, 2400],
            load_roi: bool = True,
            roi_config_path: str = None,
            load_threshold: bool = True,
            threshold: Optional[float] = None,
            load_type: str = 'lyse',
            h5_path: str = None,):
        """Initialize TweezerAnalysis with ROI configuration and analysis parameters.

        Parameters
        ----------
        imaging_setup : ImagingSetup
            Imaging setup configuration object
        bkg_roi_x : List[int]
            X-coordinates [start, end] for background region
        roi_config_path : str
            Path to YAML configuration file for ROIs
        method : str, default='alternative'
            Method for background subtraction, one of:
            - 'alternative': Use alternative background subtraction
            - 'standard': Use standard background subtraction
        load_roi : bool, default=True
            If True, load roi_x and site ROIs from standard .npy files:
            - roi_x.npy: Main ROI x-coordinates
              - site_roi_x.npy: Site ROI x-coordinates
              - site_roi_y.npy: Site ROI y-coordinates
            If False, load from YAML config (requires roi_x and site_roi)
        load_type : str, optional
            Type of loading to perform
        h5_path : str, optional
            Path to h5 file to load
        load_threshold : bool, default=True
            Whether to load threshold from file
        threshold : float, optional
            Threshold value to use if not loading from file
        """
        # Standard file locations
        self.multishot_path = 'X:\\userlib\\analysislib\\scripts\\multishot'

        # Initialize parent class first
        super().__init__(
            imaging_setup=imaging_setup,
            load_type=load_type,
            h5_path=h5_path
        )

        # Load ROIs and set class attributes
        self.atom_roi, self.background_roi, self.site_roi = self.load_roi(
            roi_config_path=roi_config_path,
            bkg_roi_x=bkg_roi_x,
            load_roi=load_roi
        )
            
        # Set threshold
        self.threshold = self.load_threshold(load_threshold, threshold)

        # Process images
        self.atom_images, self.background_images, self.sub_images = self.get_image_bkg_sub(method=method)
        self.roi_atoms, self.roi_bkgs = self.get_images_roi()

    def load_roi(self, roi_config_path: str, bkg_roi_x: List[int], load_roi: bool):
        """Load ROI and site ROI either from file or from provided configuration.
        
        Parameters
        ----------
        roi_config_path : str
            Path to YAML configuration file containing ROI specifications.
            When load_roi=False:
            - Must contain 'roi_x' key with [start, end] coordinates
            - Must contain 'site_roi' section with 'site_roi_x' and 'site_roi_y'
        bkg_roi_x : List[int]
            X-coordinates [start, end] for background region.
            This is passed directly from __init__ rather than loaded from YAML.
        load_roi : bool
            If True, load ROIs from standard file locations
            If False, use ROIs from YAML config
            
        Returns
        -------
        atom_roi : List[List[int]]
            [[x_min, x_max], [y_min, y_max]] for atom ROI
        background_roi : List[List[int]]
            [[x_min, x_max], [y_min, y_max]] for background ROI
        site_roi : List[List[List[int]]]
            [site_roi_x, site_roi_y] for site-specific ROIs
        """
        # Get y-coordinates from globals
        try:
            roi_y = self.globals["tw_kinetix_roi_row"]
            print(f"Using tw_kinetix_roi_row from globals: {roi_y}")
        except KeyError:
            print("Warning: tw_kinetix_roi_row not found in globals, using full camera height")
            roi_y = [0, self.imaging_setup.camera.image_size]

        # Standard file locations for ROIs
        roi_paths = {
            'site_roi_x': os.path.join(self.multishot_path, "site_roi_x.npy"),
            'site_roi_y': os.path.join(self.multishot_path, "site_roi_y.npy"),
            'roi_x': os.path.join(self.multishot_path, "roi_x.npy")
        }

        if load_roi:
            # Load ROIs from standard .npy files
            try:
                roi_x = np.load(roi_paths['roi_x']).tolist()
                site_roi_x = np.load(roi_paths['site_roi_x'])  # Keep as numpy array for calculations
                site_roi_y = np.load(roi_paths['site_roi_y'])
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load ROI files from {self.multishot_path}: {e}")

            # Adjust site_roi_x relative to roi_x
            site_roi_x = site_roi_x - roi_x[0]

            # Add bkg site roi
            site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], np.array(site_roi_x)])
            site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) - 10], np.array(site_roi_y)])

        else:
            roi_config = ROIConfig.from_yaml(roi_config_path)
            # When not loading from files, validate YAML config
            if roi_config.roi_x is None:
                raise ValueError("When load_roi is False, the YAML config must include 'roi_x' coordinates")
            if roi_config.site_roi is None:
                raise ValueError("When load_roi is False, the YAML config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

            # Load ROIs from YAML
            roi_x = roi_config.roi_x
            site_roi_config = roi_config.site_roi
            site_roi_x = np.array(site_roi_config['site_roi_x'])
            site_roi_y = np.array(site_roi_config['site_roi_y'])

        # Always get roi_y from globals
        roi_y = self.globals["tw_kinetix_roi_row"]

        # Return ROIs in the format expected by the class
        return [roi_x, roi_y], [bkg_roi_x, roi_y], [site_roi_x, site_roi_y]    

    def load_threshold(self, load_threshold, default_threshold):
        """Load threshold value from file or use default."""
        if load_threshold:
            threshold_path = os.path.join(self.multishot_path, "th.npy")
            try:
                threshold = np.load(threshold_path)[0]
                print(f'Loaded threshold {threshold:.2f} from {threshold_path}')
                return threshold
            except FileNotFoundError:
                print(f'Warning: Threshold file not found at {threshold_path}, using default value')
                threshold = default_threshold
        else:
            if default_threshold is None:
                raise ValueError(
                    'When load_threshold is False, default_threshold must be provided'
                )
            return default_threshold

    def get_image_bkg_sub(self, method: str = 'average'):
        """Get background-subtracted images.

        Parameters
        ----------
        method : str, default='average'
            Method for background subtraction:
            - 'average': Use average background subtraction
            - 'alternative': Use alternative background subtraction

        Returns
        -------
        atom_images : ndarray
            Array of atom images
        background_images : ndarray
            Array of background images
        sub_images : ndarray
            Array of background-subtracted images
        """
        images = self.images
        folder_path = self.folder_path

        # Set up paths for background images
        alternative_bkg_path = os.path.join(folder_path, 'alternative_bkg')
        average_bkg_path = os.path.join(folder_path, 'avg_shot_bkg.npy')
        last_bkg_sub_path = os.path.join(folder_path, 'last_bkg_sub')

        if method == 'alternative':
            if self.globals['mot_do_coil']:
                atom_images = images
                background_images = np.load(alternative_bkg_path)
                sub_images = atom_images - background_images
                np.save(last_bkg_sub_path, sub_images)
            else:
                background_images = images
                np.save(alternative_bkg_path, images)
                sub_images = np.load(last_bkg_sub_path)
                # load last background subtracted images
                # during background taking shot to make
                # sure there is something to plot
        elif method == 'average':
            atom_images = images
            background_images = np.load(average_bkg_path)
            sub_images = atom_images - background_images
            np.save(last_bkg_sub_path, sub_images)
        else:
            raise NotImplementedError
            #TODO implement the method here

        return atom_images, background_images, sub_images
    
    def convert_dict_to_array(self, dict):
        """
        convert the images from dictionary to np.array
        with shape(# shots in a single sequence, size of images)
        """
        dict_types = list(dict.keys())
        array = []
        for item in dict_types:
            array.append(dict[item])
        return np.array(array)

    def get_images_roi(self):
        """Get ROI images for atoms and background regions.

        Returns
        -------
        roi_atoms : array
            Images cropped to atoms ROI region
        roi_bkgs : array
            Images cropped to background ROI region
        """
        [roi_x, _] = self.atom_roi
        [roi_x_bkg, _] = self.background_roi
        sub_images = self.sub_images
        roi_atoms = sub_images[:, :, roi_x[0]:roi_x[1]]
        roi_bkgs = sub_images[:, :, roi_x_bkg[0]:roi_x_bkg[1]]
        return roi_atoms, roi_bkgs
    
    def get_site_counts(self, sub_image):
        """Get counts in each tweezer site ROI.

        Returns
        -------
        roi_number_lst : list
            List of summed counts in each site ROI
        """
        roi_number_lst = []
        site_roi_x = self.site_roi[0]
        site_roi_y = self.site_roi[1]

        for i in np.arange(site_roi_x.shape[0]):
            y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
            x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]

            site_roi_signal = sub_image[y_start:y_end, x_start:x_end]
            signal_sum = np.sum(site_roi_signal)
            roi_number_lst.append(signal_sum)

        return roi_number_lst

    def analyze_site_existence(self, roi_number_lst):
        """Analyze atom existence in each site and create visualization rectangles.

        Parameters
        ----------
        roi_number_lst : list
            List of summed counts in each site ROI from get_site_counts

        Returns
        -------
        rect_sig : list
            List of Rectangle patches for visualization
        atom_exist_lst : list
            List of booleans indicating atom presence in each site
        """
        rect_sig = []
        atom_exist_lst = []
        site_roi_x = self.site_roi[0]
        site_roi_y = self.site_roi[1]

        for i, signal_sum in enumerate(roi_number_lst):
            y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
            x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]

            if signal_sum > self.threshold:
                rect = patches.Rectangle(
                    (x_start, y_start),
                    x_end - x_start,
                    y_end - y_start,
                    linewidth=1,
                    edgecolor='gold',
                    facecolor='none',
                    alpha=0.5,
                    fill=False)
                rect_sig.append(rect)
                atom_exist_lst.append(1)
            else:
                atom_exist_lst.append(0)

        return rect_sig, atom_exist_lst

    def plot_images(self, roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True):
        """Plot the analysis results with optional site ROIs and background ROIs.
        
        Parameters
        ----------
        roi_image_scale : int, default=150
            Scale factor for ROI image colormaps
        show_site_roi : bool, default=True
            Whether to show site ROI rectangles
        plot_bkg_roi : bool, default=True
            Whether to plot background ROI images
        """
        num_of_imgs = len(self.sub_images)
        
        # Get site counts and analyze existence for each image
        rect_sig = []
        atom_exist_lst = []
        for i in range(num_of_imgs):
            sub_image = self.sub_images[i]
            roi_number_lst_i = self.get_site_counts(sub_image)
            rect_sig_i, atom_exist_lst_i = self.analyze_site_existence(roi_number_lst_i)
            rect_sig.append(rect_sig_i)
            atom_exist_lst.append(atom_exist_lst_i)

        # Create figure with enough rows for all shots plus background ROIs
        num_rows = num_of_imgs + (2 if plot_bkg_roi else 0)
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 5*num_rows), constrained_layout=True)
        
        # If only one subplot, wrap it in a list for consistent indexing
        if num_rows == 1:
            axs = [axs]

        plt.rcParams.update({'font.size': 14})
        fig.suptitle('Tweezer Array Imaging Analysis', fontsize=16)

        # Configure all axes
        for ax in axs:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
            ax.tick_params(axis='both', which='major', labelsize=12)

        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

        # Plot tweezer ROIs for each shot
        for i in range(num_of_imgs):
            ax_tweezer = axs[i]
            ax_tweezer.set_title(f'Shot {i+1} Tweezer ROI', fontsize=14, pad=10)
            pos = ax_tweezer.imshow(self.roi_atoms[i], **roi_img_color_kw)
            
            if show_site_roi:
                # Draw the base ROI rectangles
                site_roi_x = self.site_roi[0]
                site_roi_y = self.site_roi[1]
                base_rects = []
                for j in range(site_roi_x.shape[0]):
                    y_start, y_end = site_roi_y[j,0], site_roi_y[j,1]
                    x_start, x_end = site_roi_x[j,0], site_roi_x[j,1]
                    rect = patches.Rectangle(
                        (x_start, y_start),
                        x_end - x_start,
                        y_end - y_start,
                        linewidth=1,
                        edgecolor='blue',
                        facecolor='none',
                        alpha=0.3)
                    base_rects.append(rect)
                pc_base = PatchCollection(base_rects, match_original=True)
                ax_tweezer.add_collection(pc_base)
                
                # Draw the signal rectangles
                if i < len(rect_sig):
                    pc_sig = PatchCollection(rect_sig[i], match_original=True)
                    ax_tweezer.add_collection(pc_sig)
            
            fig.colorbar(pos, ax=ax_tweezer).ax.tick_params(labelsize=12)

        # Plot background ROIs if requested
        if plot_bkg_roi:
            for i in range(min(2, num_of_imgs)):  # Plot up to 2 background ROIs
                ax_bkg = axs[num_of_imgs + i]
                ax_bkg.set_title(f'Shot {i+1} Background ROI', fontsize=14, pad=10)
                pos = ax_bkg.imshow(self.roi_bkgs[i], **roi_img_color_kw)
                fig.colorbar(pos, ax=ax_bkg).ax.tick_params(labelsize=12)

        # # Save the figure
        # fig.savefig(os.path.join(self.folder_path, 'atom_cloud.png'))
        # plt.close(fig)
