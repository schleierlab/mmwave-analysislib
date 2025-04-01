from abc import abstractmethod, ABC
from typing import Optional
from matplotlib.figure import Figure
import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    lyse
except NameError:
    import lyse # needed for MLOOP

from .plot_config import PlotConfig
from .image import ROI

class BaseStatistician(ABC):
    """Base class for statistical analysis of tweezer or bulk gas imaging data."""
    def __init__(self):
        # TODO: move common init tasks here from child classes
        pass

    @abstractmethod
    def _load_processed_quantities(self, preproc_h5_path: str) -> None:
        """Load processed quantities from an h5 file."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _save_mloop_params(self, shot_h5_path: str) -> None:
        """Save values and uncertainties to be used by MLOOP for optimization."""
        raise NotImplementedError("Subclasses must implement this method.")