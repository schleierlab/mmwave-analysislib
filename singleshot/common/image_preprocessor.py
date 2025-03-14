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
from .analysis_config import ImagingSystem

class ImagePreProcessor:
    """Base class for image preprocessing.
    
    This class handles the basic image loading and preprocessing operations
    common to both tweezer and bulk gas analysis.
    
    Parameters
    ----------
    imaging_setup : ImagingSystem
        Configuration object for the imaging system setup
    load_type : str, default='lyse'
        Type of data loading to use
    h5_path : Optional[str], default=None
        Path to H5 file for data loading
    """
    def __init__(
            self,
            imaging_setup: ImagingSystem,
            load_type: str = 'lyse',
            h5_path: Optional[str] = None):
        """Initialize image preprocessing.
        
        Parameters
        ----------
        imaging_setup : ImagingSystem
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
