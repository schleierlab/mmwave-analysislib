from typing import Optional

import h5py
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy import optimize
import uncertainties

from .plot_config import PlotConfig


globals_friendly_names = {
    'mw_detuning': 'Microwave detuning',
}


class BulkGasPlotter:
    def __init__(self, h5_path, plot_config: PlotConfig = None):
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(h5_path)

    def _load_processed_quantities(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.atom_numbers = f['atom_numbers'][:]
            self.params_list = f['params'][:]
            self.n_runs = f.attrs['n_runs']
            self.current_params = f['current_params'][:]

    def plot_atom_number(self, fig: Optional[Figure] = None, plot_lorentz = True):
        """Plot atom number vs the shot number and save the image in the folder path.

        This function requires that get_atom_number has been run first to save the
        atom number to processed_quantities.h5.

        Parameters
        ----------
        fig : Optional[Figure], default=None
            Figure to plot on, if None a new figure is created
        plot_lorentz : bool, default=True
            Whether to plot a Lorentzian fit to the atom number data
        """
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
        else:
            ax = fig.subplots()

        loop_params = np.squeeze(self.current_params)
        
        # Group data points by x-value and calculate statistics
        unique_params = np.unique(loop_params)
        means = np.array([
            np.mean(self.atom_numbers[loop_params == x]) 
            for x in unique_params
        ])
        stds = np.array([
            np.std(self.atom_numbers[loop_params == x]) 
            for x in unique_params
        ])
        
        ax.errorbar(
            unique_params,
            means,
            yerr=stds,
            marker='.',
            linestyle='-',
            alpha=0.5,
            capsize=3,
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

        # doing the fit at the end of the run
        if self.atom_numbers.shape[0] == self.n_runs and plot_lorentz:
            popt, pcov = self.fit_lorentzian(unique_params, means)
            upopt = uncertainties.correlated_values(popt, pcov)

            x_plot = np.linspace(
                np.min(unique_params),
                np.max(unique_params),
                1000,
            )

            ax.plot(x_plot, self.lorentzian(x_plot, *popt))
            fig.suptitle(
                f'Center frequency: ${upopt[0]:SL}$ MHz; '
                f'Width: ${1e+3 * upopt[1]:SL}$ kHz'
            )

    @staticmethod
    def lorentzian(x, x0, width, a, offset):
        """
        Returns a Lorentzian function.

        Parameters
        ----------
        x : float or array
            Input values
        x0 : float
            Central frequency
        width : float
            Width of the Lorentzian
        a : float
            Amplitude of the Lorentzian
        offset : float
            Offset of the Lorentzian

        Returns
        -------
        float or array
            Lorentzian function values
        """
        detuning = x - x0
        return a * width / ((width / 2)**2 + detuning**2) + offset

    def fit_lorentzian(self, x_data, y_data):
        """
        Fits a Lorentzian function to the atom number data.
        """
        x0_guess = x_data[np.argmax(y_data)]
        x_resolution = (x_data[1] - x_data[0])
        width_guess = 2*x_resolution
        y_data_range = np.max(y_data) - np.min(y_data)
        a_guess = y_data_range / width_guess * (width_guess/2)**2

        offset_guess = np.min(y_data)
        p0 = [x0_guess, width_guess, a_guess, offset_guess]
        return optimize.curve_fit(self.lorentzian, x_data, y_data, p0=p0)

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