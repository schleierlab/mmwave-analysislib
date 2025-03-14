import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

try:
    import lyse
except ImportError:
    from analysis.data import h5lyze as lyse

from .plot_config import PlotConfig
from .tweezer_analysis import TweezerAnalysis

# Add analysis library root to Python path
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)

class TweezerPlotter:
    """Class for plotting tweezer analysis results.
    
    This class provides methods for visualizing tweezer analysis results,
    including raw images, background-subtracted images, and ROI counts.
    
    Parameters
    ----------
    tweezer_analyzer : TweezerAnalysis
        Analyzer object containing the data to plot
    plot_config : PlotConfig, optional
        Configuration object for plot styling
    """
    def __init__(self, tweezer_analyzer, plot_config: PlotConfig = None):
        self.tweezer_analyzer = tweezer_analyzer
        self.plot_config = plot_config or PlotConfig()

    def plot_images(self, show_site_roi=True, plot_bkg_roi=True):
        """Plot the analysis results with optional site ROIs and background ROIs.
        
        Parameters
        ----------
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
            roi_number_lst_i = self.tweezer_analyzer.get_site_counts(sub_image)
            rect_sig_i, atom_exist_lst_i = self.tweezer_analyzer.analyze_site_existence(roi_number_lst_i)
            rect_sig.append(rect_sig_i)
            atom_exist_lst.append(atom_exist_lst_i)

        # Create figure with enough rows for all shots plus background ROIs
        num_rows = num_of_imgs + (2 if plot_bkg_roi else 0)
        fig_height = self.plot_config.figure_size[1] * num_rows
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, 
                               figsize=(self.plot_config.figure_size[0], fig_height), 
                               constrained_layout=self.plot_config.constrained_layout)
        
        # If only one subplot, wrap it in a list for consistent indexing
        if num_rows == 1:
            axs = [axs]

        plt.rcParams.update({'font.size': self.plot_config.font_size})
        fig.suptitle('Tweezer Array Imaging Analysis', fontsize=self.plot_config.title_font_size)

        # Configure all axes
        for ax in axs:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
            ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)

        roi_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.roi_image_scale)

        # Plot tweezer ROIs for each shot
        for i in range(num_of_imgs):
            ax_tweezer = axs[i]
            ax_tweezer.set_title(f'Shot {i+1} Tweezer ROI', fontsize=self.plot_config.title_font_size, pad=10)
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
                        edgecolor='r',
                        facecolor='none',
                        alpha=0.5,
                        fill=False)
                    base_rects.append(rect)
                pc_base = PatchCollection(base_rects, match_original=True)
                ax_tweezer.add_collection(pc_base)
                
                # Draw the signal rectangles
                if i < len(rect_sig):
                    pc_sig = PatchCollection(rect_sig[i], match_original=True)
                    ax_tweezer.add_collection(pc_sig)
            
            fig.colorbar(pos, ax=ax_tweezer).ax.tick_params(labelsize=self.plot_config.label_font_size)

        # Plot background ROIs if requested
        if plot_bkg_roi:
            for i in range(min(2, num_of_imgs)):  # Plot up to 2 background ROIs
                ax_bkg = axs[num_of_imgs + i]
                ax_bkg.set_title(f'Shot {i+1} Background ROI', fontsize=self.plot_config.title_font_size, pad=10)
                pos = ax_bkg.imshow(self.roi_bkgs[i], **roi_img_color_kw)
                fig.colorbar(pos, ax=ax_bkg).ax.tick_params(labelsize=self.plot_config.label_font_size)
