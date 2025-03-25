from abc import abstractmethod, ABC
import os
from typing import Any, Optional, Literal

import numpy as np

import h5py

try:
    lyse
except:
    import lyse

from .analysis_config import ImagingSystem
from analysislib.analysis.data import h5lyze as hz


class ImagePreprocessor(ABC):
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

    run_number: int
    h5_path: str
    folder_path: str

    n_runs: int
    '''Total number of runs for this runmanager expansion.'''

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
        self.exposures, self.run_number, self.globals = self.load_images()
        self.params, self.n_rep, self.current_params = self.get_scanning_params()

        with h5py.File(self.h5_path, mode='r') as f:
            self.n_runs = f.attrs['n_runs']

    # TODO migrate to using pathlib Paths instead
    def get_h5_path(self, load_type: Literal['lyse', 'h5'], h5_path: Optional[str] = None) -> tuple[str, str]:
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

    # TODO unit tests for this
    def get_scanning_params(self,):
        """
        get scanning parameters and number of repetitions based
        on the expansion of each globals, if the expansion is 'inner' or 'outer'
        then it is a scanning parameter, number of repetitions is determined by n_shots
        in globals. If only n_shots is present, then n_rep = 1 and scanning params = n_shots

        Returns
        -------
        params: dict
            Scanning parameters; key is the parameter name
            and value is a tuple (parameter_values, unit)
        n_rep: int
            number of repetitions
        """
        h5_path = self.h5_path
        with h5py.File(h5_path, mode='r+') as f:
            globals = f['globals']
            params = {}
            current_params = []
            for group in globals.keys():
                expansion = hz.getAttributeDict(globals[group]['expansion'])
                for key, value in expansion.items():
                    if value == 'outer' or value == 'inner':
                        global_var = hz.getAttributeDict(globals[group])[key]
                        global_unit = hz.getAttributeDict(globals[group]['units'])[key]
                        params[key] = (global_var, global_unit)
            if 'n_shots' in params:
                rep_str, _ = params['n_shots']
                if rep_str[0:2] != 'np':
                    rep_str = 'np.' + rep_str
                rep = eval(rep_str)
                n_rep = rep.shape[0]
                del params['n_shots']
                if len([key for key in params if 'do' not in key]) == 0:
                    params['n_shots'] = [rep_str, 'Shots']
                    n_rep = 1
            else:
                n_rep = 1
            for key, value in params.items():
                current_params.append(self.globals[key])

        return params, n_rep, np.array(current_params)

    def get_current_scanning_params(self,):
        """
        get current scanning parameters from globals

        Returns
        -------
        current_param_vals: dict
            Current scanning parameters; key is the parameter name
            and value is the parameter value
        """

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

    def load_images(self,) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
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
            images = hz.datasetsToDictionary(f[self.imaging_setup.camera.image_group_name], recursive=True)
            run_number = f.attrs['run number']
        return images, run_number, globals

    # def load_processed_quantities(self, *quantities):
    #     """Load processed quantities from the HDF5 file.

    #     Parameters
    #     ----------
    #     *quantities : str
    #         Names of quantities to load. If none specified, loads all available quantities.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of loaded quantities, with quantity names as keys
    #     """
    #     h5_path = os.path.join(self.folder_path, 'processed_quantities.h5')

    #     if not os.path.exists(h5_path):
    #         raise FileNotFoundError(f"No processed quantities file found at {h5_path}")

    #     results = {}
    #     with h5py.File(h5_path, 'r') as f:
    #         # If no specific quantities requested, load all available
    #         if not quantities:
    #             quantities = f.keys()

    #         # Load each requested quantity
    #         for quantity in quantities:
    #             if quantity in f:
    #                 results[quantity] = f[quantity][()]

    #     return results

    # def save_processed_quantities(self, **quantities):
    #     """Save processed quantities to an HDF5 file.

    #     Parameters
    #     ----------
    #     **quantities : dict
    #         Dictionary of quantities to save. Each key-value pair will be saved
    #         as a dataset in the HDF5 file. Values must be numpy arrays or scalars.
    #         Common quantities include:
    #         - atom_number : number of atoms in the cloud
    #         - time_of_flight : time of flight in seconds
    #         - gaussian_waist : tuple of (x, y) waist sizes
    #         - temperature : tuple of (x, y) temperatures
    #     """
    #     h5_path = os.path.join(self.folder_path, 'processed_quantities.h5')

    #     # Always write mode since each folder represents a single run
    #     with h5py.File(h5_path, 'w') as f:
    #         # Save each quantity
    #         for name, value in quantities.items():
    #             if isinstance(value, (tuple, list)):
    #                 # Convert tuples/lists to numpy arrays
    #                 value = np.array(value)
    #             f.create_dataset(name, data=value)

    @abstractmethod
    def process_shot(self,) -> None:
        """
        Abstract method that should be overriden by subclasses to process the image data from a single shot.
        """
        raise NotImplementedError
