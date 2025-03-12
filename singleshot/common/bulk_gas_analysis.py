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
from .analysis_config import AnalysisConfig, ImagingSystem

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
        all fitting parameters are saved in processed_quantities.h5

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

    def save_processed_quantities(self, **quantities):
        """Save processed quantities to an HDF5 file.

        Parameters
        ----------
        **quantities : dict
            Dictionary of quantities to save. Each key-value pair will be saved
            as a dataset in the HDF5 file. Values must be numpy arrays or scalars.
            Common quantities include:
            - atom_number : number of atoms in the cloud
            - time_of_flight : time of flight in seconds
            - gaussian_waist : tuple of (x, y) waist sizes
            - temperature : tuple of (x, y) temperatures
        """
        h5_path = os.path.join(self.folder_path, 'processed_quantities.h5')
        
        # Always write mode since each folder represents a single run
        with h5py.File(h5_path, 'w') as f:
            # Save each quantity
            for name, value in quantities.items():
                if isinstance(value, (tuple, list)):
                    # Convert tuples/lists to numpy arrays
                    value = np.array(value)
                f.create_dataset(name, data=value)

    def save_atom_number(self, atom_number):
        """Save atom number to the HDF5 file."""
        self.save_processed_quantities(atom_number=atom_number)

    def save_atom_temperature(self, atom_number, time_of_flight, gaussian_waist, temperature):
        """Save temperature measurement data to the HDF5 file."""
        self.save_processed_quantities(
            atom_number=atom_number,
            time_of_flight=time_of_flight,
            gaussian_waist=gaussian_waist,
            temperature=temperature
        )

    def load_processed_quantities(self, *quantities):
        """Load processed quantities from the HDF5 file.

        Parameters
        ----------
        *quantities : str
            Names of quantities to load. If none specified, loads all available quantities.

        Returns
        -------
        dict
            Dictionary of loaded quantities, with quantity names as keys
        """
        h5_path = os.path.join(self.folder_path, 'processed_quantities.h5')
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No processed quantities file found at {h5_path}")
            
        results = {}
        with h5py.File(h5_path, 'r') as f:
            # If no specific quantities requested, load all available
            if not quantities:
                quantities = f.keys()
            
            # Load each requested quantity
            for quantity in quantities:
                if quantity in f:
                    results[quantity] = f[quantity][()]
                
        return results

    def load_atom_number(self):
        """Load atom number from the HDF5 file.

        Returns
        -------
        float
            Atom number for this run
        """
        try:
            result = self.load_processed_quantities('atom_number')
            return result['atom_number']
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load atom number: {e}")
            return None

    def load_atom_temperature(self):
        """Load temperature measurement data from the HDF5 file.

        Returns
        -------
        tuple
            (time_of_flight, waist_x, waist_y, temperature_x, temperature_y)
        """
        try:
            result = self.load_processed_quantities(
                'time_of_flight', 
                'gaussian_waist', 
                'temperature'
            )
            
            time_of_flight = result['time_of_flight']
            gaussian_waist = result['gaussian_waist']
            temperature = result['temperature']
            
            # Split waist and temperature into x,y components
            waist_x = gaussian_waist[0]
            waist_y = gaussian_waist[1]
            temperature_x = temperature[0]
            temperature_y = temperature[1]
            
            return time_of_flight, waist_x, waist_y, temperature_x, temperature_y
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load temperature data: {e}")
            return None, None, None, None, None

    def get_atom_temperature(self, time_of_flight, gaussian_waist):
        """Get atom temperature from cloud size at different time of flight.

        The cloud size is fitted with a Gaussian function to extract the waist.
        The temperature is calculated from the waist size vs time of flight.
        All fitting parameters are saved in processed_quantities.h5.

        Parameters
        ----------
        time_of_flight : float
            Time of flight in seconds
        gaussian_waist : tuple
            (x_waist, y_waist) in meters

        Returns
        -------
        temperature : tuple
            (x_temperature, y_temperature) in Kelvin
        """
        temperature = self.m / self.kB * (gaussian_waist/time_of_flight)**2
        return temperature
