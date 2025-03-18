from typing import Optional, cast
import h5py
from matplotlib import colors, patches, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from uncertainties import ufloat

from .bulk_gas_analysis import BulkGasPreprocessor
from .image import ROI, Image
from .plot_config import PlotConfig


class BulkGasPlotter:
    def __init__(self, h5_path, plot_config: PlotConfig = None):
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(h5_path)

    def _load_processed_quantities(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.atom_numbers = f['atom_numbers'][:]
            self.params_list = f['params'][:]
            print(self.params_list)
            self.current_params = f['current_params'][:]
            

    # def plot_images(self):
    #     """
    #     plot the images of raw image/background image full frame and background subtracted images in rois
    #     and also save the images in the folder path
    #     """
    #     atom_image = self.atom_image
    #     background_image = self.background_image
    #     roi_atoms = self.roi_atoms
    #     roi_bkg = self.roi_bkg
    #     [roi_x, roi_y] = self.atoms_roi
    #     [roi_x_bkg, roi_y_bkg] = self.background_roi

    #     fig, axs = plt.subplots(nrows=2, ncols=2,
    #                            figsize=self.plot_config.figure_size,
    #                            constrained_layout=self.plot_config.constrained_layout)
    #     (ax_atom_raw, ax_bkg_raw), (ax_bkg_roi, ax_atom_roi) = axs
    #     for ax in axs[0]:
    #         ax.set_xlabel('x [px]')
    #         ax.set_ylabel('y [px]')
    #         ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
    #     for ax in axs[1]:
    #         ax.set_xlabel('x [m]')
    #         ax.set_ylabel('y [m]')
    #         ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)

    #     raw_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.raw_image_scale)
    #     ax_atom_raw.set_title('Raw, with atom', fontsize=self.plot_config.title_font_size)
    #     pos = ax_atom_raw.imshow(atom_image, **raw_img_color_kw)
    #     fig.colorbar(pos, ax=ax_atom_raw).ax.tick_params(labelsize=self.plot_config.label_font_size)

    #     ax_bkg_raw.set_title('Raw, no atom', fontsize=self.plot_config.title_font_size)
    #     pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
    #     fig.colorbar(pos, ax=ax_bkg_raw).ax.tick_params(labelsize=self.plot_config.label_font_size)

    #     pixel_size_before_magnification = self.imaging_setup.pixel_size_before_magnification
    #     ax_atom_roi.set_title('Atom ROI', fontsize=self.plot_config.title_font_size)
    #     roi_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.roi_image_scale)
    #     pos = ax_atom_roi.imshow(
    #         roi_atoms,
    #         extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*pixel_size_before_magnification,
    #         **roi_img_color_kw,
    #     )
    #     fig.colorbar(pos, ax=ax_atom_roi).ax.tick_params(labelsize=self.plot_config.label_font_size)

    #     ax_bkg_roi.set_title('Background ROI', fontsize=self.plot_config.title_font_size)
    #     pos = ax_bkg_roi.imshow(
    #         roi_bkg,
    #         vmin=-10,
    #         vmax=10,
    #         extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*pixel_size_before_magnification,
    #         cmap=self.plot_config.colormap
    #     )
    #     fig.colorbar(pos, ax=ax_bkg_roi).ax.tick_params(labelsize=self.plot_config.label_font_size)
    #     fig.savefig(self.folder_path+'\\atom_cloud.png')
    #     return

    def plot_atom_number(self, fig: Optional[Figure] = None):
        """Plot atom number vs the shot number and save the image in the folder path.

        This function requires that get_atom_number has been run first to save the
        atom number to processed_quantities.h5.
        """
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
        else:
            ax = fig.subplots()
        
        loop_params = np.transpose(np.array(self.current_params))[0]
        ax.plot(
            loop_params, self.atom_numbers,
            marker='.',
        )
        ax.set_xlabel(
            f"{self.params_list[0][0].decode('utf-8')} [{self.params_list[0][1].decode('utf-8')}]",
            fontsize=self.plot_config.label_font_size,
        )
        ax.set_ylabel(
            'atom count',
            fontsize=self.plot_config.label_font_size,
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.plot_config.label_font_size,
        )

        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')

    # def plot_atom_temperature(self):
    #     """Plot atom temperature vs shot number and fit temperature with time of flight.

    #     This function requires that get_atom_temperature has been run first to save
    #     the temperature data to processed_quantities.h5.

    #     Fits temperature by analyzing cloud size at different time of flight values
    #     through linear fit, accounting for non-zero initial cloud size.
    #     """
    #     time_of_flight, waist_x, waist_y, temperature_x, temperature_y = self.load_atom_temperature()
    #     fig, ax = plt.subplots(figsize=self.plot_config.figure_size,
    #                           constrained_layout=self.plot_config.constrained_layout)

    #     ax.plot(
    #         np.array(temperature_x)*1e6,
    #         label='$x$ temperature',
    #     )
    #     ax.plot(np.array(temperature_y)*1e6, label='$y$ temperature')

    #     ax.set_xlabel('Shot number', fontsize=self.plot_config.label_font_size)
    #     ax.set_ylabel('Temperature (uK)', fontsize=self.plot_config.label_font_size)
    #     ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
    #     ax.legend(fontsize=self.plot_config.label_font_size)

    #     ax.grid(color=self.plot_config.grid_color_major, which='major')
    #     ax.grid(color=self.plot_config.grid_color_minor, which='minor')
    #     fig.savefig(self.folder_path+'\\temperature_single_shot.png')

    #     if self.run_number >= 2:
    #         # linear fit to extract the velocity of the atoms through sphereical expansion
    #         slope_lst = []
    #         intercept_lst = []
    #         slope_uncertainty_lst = []
    #         for waist in [waist_x, waist_y]:
    #             coefficient, covariance = np.polyfit(time_of_flight**2, np.array(waist)**2, 1, cov = True)
    #             slope_lst.append(coefficient[0])
    #             intercept_lst.append(coefficient[1])
    #             error = np.sqrt(np.diag(covariance))
    #             slope_error = error[0]
    #             slope_uncertainty_lst.append(ufloat(coefficient[0],slope_error))

    #         slope_lst = np.array(slope_lst)
    #         intercept_lst = np.array(intercept_lst)
    #         slope_uncertainty_lst = np.array(slope_uncertainty_lst)
    #         T_lst = slope_uncertainty_lst*self.m/self.kB*1e6

    #         t_plot = np.arange(0,max(time_of_flight)+1e-3,1e-3)
    #         fig2, ax2 = plt.subplots(figsize=self.plot_config.figure_size,
    #                                 constrained_layout=self.plot_config.constrained_layout)
    #         for slope,intercept,T in zip(slope_lst,intercept_lst,T_lst):
    #             ax2.plot(t_plot**2*1e6 , (slope*t_plot**2 + intercept)*1e12, label=rf'$T = {T:LS}$ $\mu$K')

    #         i=0
    #         for waist in [waist_x, waist_y]:
    #             ax2.plot(time_of_flight**2*1e6, np.array(waist)**2*1e12, 'oC'+str(i), label = 'data')
    #             i+=1

    #         ax2.set_xlabel(r'Time$^2$ (ms$^2$)', fontsize=self.plot_config.label_font_size)
    #         ax2.set_ylabel(r'$\sigma^2$ ($\mu$m$^2$)', fontsize=self.plot_config.label_font_size)
    #         ax2.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
    #         ax2.legend(fontsize=self.plot_config.label_font_size)
    #         ax2.set_ylim(ymin=0)

    #         ax2.grid(color=self.plot_config.grid_color_major, which='major')
    #         ax2.grid(color=self.plot_config.grid_color_minor, which='minor')
    #         fig2.savefig(self.folder_path+'\\temperature_fit.png')
    #     return

    # def plot_amplitude_vs_parameter(self):
        # """Plot amplitude vs scanned parameter.

        # The amplitude (e.g. atom number, survival rate) is loaded from processed_quantities.h5.
        # The data is plotted against the first parameter for now.

        # This can be modified for general use with multiple repetitions and multiple parameters.
        # """
        # amplitude = self.gas_analyzer.load_processed_quantities('atom_number')
        # fig, ax = plt.subplots(figsize=self.plot_config.figure_size,
        #                       constrained_layout=self.plot_config.constrained_layout)

        # for key, value in self.params.items():
        #     para = eval(value[0])
        #     unit = value[1]
        #     label_string = f'{key} ({unit})'

        # if self.n_rep > 1:
        #     #TODO support multiple repetitions
        #     raise NotImplementedError("multiple repetitions is not supported yet")

        # if len(self.params.keys()) == 1:
        #     # 1D scanning case
        #     amplitude_resize = np.resize(amplitude, para.shape[0])
        #     amplitude_resize[len(amplitude):] = np.nan
        # else:
        #     #TODO support 2D scanning
        #     raise NotImplementedError("2 scanning parameters is not supported yet")

        # ax.plot(para, amplitude_resize, '-o')
        # ax.set_xlabel(label_string, fontsize=self.plot_config.label_font_size)
        # ax.set_ylabel('amplitude', fontsize=self.plot_config.label_font_size)
        # ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
        # ax.grid(color=self.plot_config.grid_color_major, which='major')
        # ax.grid(color=self.plot_config.grid_color_minor, which='minor')

        # fig.savefig(self.folder_path+'\\amplitude_vs_parameter.png')