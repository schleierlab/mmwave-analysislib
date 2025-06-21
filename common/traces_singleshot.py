from abc import abstractmethod, ABC
import os
from typing import Any, Optional, Literal

import numpy as np

import h5py

from .analysis_config import ImagingSystem
from analysislib.analysis.data import h5lyze as hz
from typing import Union
from os import PathLike
from pathlib import Path


class TraceSingleshotAnalysis(ABC):
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
    h5_path: Path

    n_runs: int
    '''Total number of runs for this runmanager expansion.'''

    def __init__(
            self,
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
        self.h5_path = self.get_h5_path(load_type=load_type, h5_path=h5_path)
        self.traces_dict, self.run_number= self.load_traces()
        self.traces_name, self.traces_time, self.traces_value = self.convert_dict_to_name_time_value(self.traces_dict)

        with h5py.File(self.h5_path, mode='r') as f:
            self.n_runs = f.attrs['n_runs']

    # TODO migrate to using pathlib Paths instead
    def get_h5_path(self, load_type: Literal['lyse', 'h5'], h5_path: Optional[str] = None) -> Path:
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
        h5_path: Path
            The actual h5 file path used based on the selected load_type
        """
        if load_type == 'lyse':
            import lyse
            # Is this script being run from within an interactive lyse session?
            if lyse.spinning_top:
                # If so, use the filepath of the current h5_path
                h5_path = Path(lyse.path)
            else:
                # If not, get the filepath of the last h5_path of the lyse DataFrame
                df = lyse.data()
                h5_path = Path(df.filepath.iloc[-1])
        elif load_type == 'h5':
            if h5_path is None:
                raise ValueError("When load_type is h5, please provide exact h5 path")
            h5_path = Path(h5_path)
        else:
            raise ValueError("load_type must be 'lyse' or 'h5'")

        return h5_path



    def load_traces(self,) -> tuple[tuple[np.ndarray, ...], int, dict[str, Any]]:
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
            hz.getDefaultParamsFromFile
            traces_dict = hz.datasetsToDictionary(f["data"]["traces"], recursive=True)
            run_number = f.attrs['run number']

        return traces_dict, run_number, globals,

    def convert_dict_to_name_time_value(self, traces_dict):
        traces_name = []
        traces_time = []
        traces_value = []
        for key, value in traces_dict.items():
            traces_name.append(key)
            traces_time.append(value['time'])
            traces_value.append(value['value'])
        return traces_name, traces_time, traces_value


    @abstractmethod
    def process_shot(self,) -> None:
        """
        Abstract method that should be overriden by subclasses to process the image data from a single shot.
        """
        raise NotImplementedError
