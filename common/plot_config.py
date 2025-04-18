"""Configuration class for plotting parameters used across different plotters."""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PlotConfig:
    """Configuration for plot parameters used across different plotters.
    
    Parameters
    ----------
    font_size : int, default=14
        Base font size for all text elements
    title_font_size : int, default=16
        Font size for plot titles
    label_font_size : int, default=12
        Font size for axis labels and colorbar ticks
    figure_size : tuple[float, float], default=(10, 5)
        Base figure size (width, height) in inches
    colormap : str, default='viridis'
        Colormap to use for image plots
    grid_color_major : str, default='0.7'
        Color for major grid lines
    grid_color_minor : str, default='0.9'
        Color for minor grid lines
    constrained_layout : bool, default=True
        Whether to use constrained_layout for automatic figure layout
    raw_image_scale : float, default=100
        Scale factor for raw image colormaps
    roi_image_scale : float, default=100
        Scale factor for ROI image colormaps
    """
    font_size: int = 14
    title_font_size: int = 16
    label_font_size: int = 12
    figure_size: Tuple[float, float] = (10, 5)
    colormap: str = 'viridis'
    grid_color_major: str = '0.7'
    grid_color_minor: str = '0.9'
    constrained_layout: bool = True
    raw_image_scale: float = 100
    roi_image_scale: float = 100
