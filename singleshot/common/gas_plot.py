from .plot_config import PlotConfig

class BulkGasPlotter:
    def __init__(self, bulk_gas_analyzer, plot_config: PlotConfig = None):
        self.bulk_gas_analyzer = bulk_gas_analyzer
        self.plot_config = plot_config or PlotConfig()

    def plot_images(self):
        """
        plot the images of raw image/background image full frame and background subtracted images in rois
        and also save the images in the folder path
        """
        atom_image = self.atom_image
        background_image = self.background_image
        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi

        fig, axs = plt.subplots(nrows=2, ncols=2, 
                               figsize=self.plot_config.figure_size,
                               constrained_layout=self.plot_config.constrained_layout)
        (ax_atom_raw, ax_bkg_raw), (ax_bkg_roi, ax_atom_roi) = axs
        for ax in axs[0]:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
            ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
        for ax in axs[1]:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)

        raw_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.raw_image_scale)
        ax_atom_raw.set_title('Raw, with atom', fontsize=self.plot_config.title_font_size)
        pos = ax_atom_raw.imshow(atom_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_atom_raw).ax.tick_params(labelsize=self.plot_config.label_font_size)

        ax_bkg_raw.set_title('Raw, no atom', fontsize=self.plot_config.title_font_size)
        pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_raw).ax.tick_params(labelsize=self.plot_config.label_font_size)

        pixel_size_before_maginification = self.imaging_setup.pixel_size_before_maginification
        ax_atom_roi.set_title('Atom ROI', fontsize=self.plot_config.title_font_size)
        roi_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.roi_image_scale)
        pos = ax_atom_roi.imshow(
            roi_atoms,
            extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*pixel_size_before_maginification,
            **roi_img_color_kw,
        )
        fig.colorbar(pos, ax=ax_atom_roi).ax.tick_params(labelsize=self.plot_config.label_font_size)

        ax_bkg_roi.set_title('Background ROI', fontsize=self.plot_config.title_font_size)
        pos = ax_bkg_roi.imshow(
            roi_bkg,
            vmin=-10,
            vmax=10,
            extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*pixel_size_before_maginification,
            cmap=self.plot_config.colormap
        )
        fig.colorbar(pos, ax=ax_bkg_roi).ax.tick_params(labelsize=self.plot_config.label_font_size)
        fig.savefig(self.folder_path+'\\atom_cloud.png')
        return

    def plot_atom_number(self):
        """
        To run this function, we need to run get_atom_number first and have data.csv in the same folder
        plot atom number from data.csv vs the shot number and also save the image in the folder path
        this plot all the shots in the same queue start with run number = 0
        so that we can see the trend
        """
        counts = self.load_atom_number()
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size,
                              constrained_layout=self.plot_config.constrained_layout)

        ax.plot(counts,'o-')
        ax.set_xlabel('Shot number', fontsize=self.plot_config.label_font_size)
        ax.set_ylabel('atom count', fontsize=self.plot_config.label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)

        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')
        fig.savefig(self.folder_path+'\\atom_number.png')
        return

    def plot_atom_temperature(self):
        """
        plot atom temperature from data.csv vs the shot number
        fitting more accurate temperature with different time of flight through linear fit
        since the size of the cloud is not zero at the begining of time of flight
        plot the fitting result
        save all the plots in the folder path
        """
        time_of_flight, waist_x, waist_y, temperature_x, temperature_y = self.load_atom_temperature()
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size,
                              constrained_layout=self.plot_config.constrained_layout)

        ax.plot(np.array(temperature_x)*1e6, label='X temperature')
        ax.plot(np.array(temperature_y)*1e6, label='Y temperature')
        ax.set_xlabel('Shot number', fontsize=self.plot_config.label_font_size)
        ax.set_ylabel('Temperature (uK)', fontsize=self.plot_config.label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
        ax.legend(fontsize=self.plot_config.label_font_size)

        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')
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
            fig2, ax2 = plt.subplots(figsize=self.plot_config.figure_size,
                                    constrained_layout=self.plot_config.constrained_layout)
            for slope,intercept,T in zip(slope_lst,intercept_lst,T_lst):
                ax2.plot(t_plot**2*1e6 , (slope*t_plot**2 + intercept)*1e12, label=rf'$T = {T:LS}$ $\mu$K')

            i=0
            for waist in [waist_x, waist_y]:
                ax2.plot(time_of_flight**2*1e6, np.array(waist)**2*1e12, 'oC'+str(i), label = 'data')
                i+=1

            ax2.set_xlabel(r'Time$^2$ (ms$^2$)', fontsize=self.plot_config.label_font_size)
            ax2.set_ylabel(r'$\sigma^2$ ($\mu$m$^2$)', fontsize=self.plot_config.label_font_size)
            ax2.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
            ax2.legend(fontsize=self.plot_config.label_font_size)
            ax2.set_ylim(ymin=0)

            ax2.grid(color=self.plot_config.grid_color_major, which='major')
            ax2.grid(color=self.plot_config.grid_color_minor, which='minor')
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
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size,
                              constrained_layout=self.plot_config.constrained_layout)

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
        ax.set_xlabel(label_string, fontsize=self.plot_config.label_font_size)
        ax.set_ylabel('amplitude', fontsize=self.plot_config.label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)
        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')

        fig.savefig(self.folder_path+'\\amplitude_vs_parameter.png')