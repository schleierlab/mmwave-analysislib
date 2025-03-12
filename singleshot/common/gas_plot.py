class BulkGasPlotter:
    def __init__(self, bulk_gas_analyzer):
        self.bulk_gas_analyzer = bulk_gas_analyzer

    def plot_images(self, raw_image_scale = 100, roi_image_scale = 100):
        """
        plot the images of raw image/background image full frame and background subtracted images in rois
        and also save the images in the folder path

        Parameters
        ----------
        raw_image_scale: float
            scale of raw image in the plot
        roi_image_scale: float
            scale of roi image in the plot

        """
        atom_image = self.atom_image
        background_image = self.background_image
        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        (ax_atom_raw, ax_bkg_raw), (ax_bkg_roi, ax_atom_roi) = axs
        for ax in axs[0]:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
        for ax in axs[1]:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')



        raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax= raw_image_scale)
        ax_atom_raw.set_title('Raw, with atom')
        pos = ax_atom_raw.imshow(atom_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_atom_raw)

        ax_bkg_raw.set_title('Raw, no atom')
        pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_raw)

        pixel_size_before_maginification = self.imaging_setup.pixel_size_before_maginification
        ax_atom_roi.set_title('Atom ROI')
        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax= roi_image_scale)
        pos = ax_atom_roi.imshow(
            roi_atoms,
            extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*pixel_size_before_maginification,
            **roi_img_color_kw,
        )
        fig.colorbar(pos, ax=ax_atom_roi)

        ax_bkg_roi.set_title('Background ROI')
        pos = ax_bkg_roi.imshow(
            roi_bkg,
            vmin=-10,
            vmax=10,
            extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*pixel_size_before_maginification, #factor: px*mag
        )
        fig.colorbar(pos, ax=ax_bkg_roi)
        fig.savefig(self.folder_path+'\\atom_cloud.png')
        return

    def plot_atom_number(self,):
        """
        To run this function, we need to run get_atom_number first and have data.csv in the same folder
        plot atom number from data.csv vs the shot number and also save the image in the folder path
        this plot all the shots in the same queue start with run number = 0
        so that we can see the trend

        """
        counts = self.load_atom_number()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(counts,'o-')
        ax.set_xlabel('Shot number')
        ax.set_ylabel('atom count')

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
        fig.savefig(self.folder_path+'\\atom_number.png')

        return

    def plot_atom_temperature(self,):
        """
        plot atom temperature from data.csv vs the shot number
        fitting more accurate temperature with different time of flight through linear fit
        since the size of the cloud is not zero at the begining of time of flight
        plot the fitting result
        save all the plots in the folder path
        """
        time_of_flight, waist_x, waist_y, temperature_x, temperature_y = self.load_atom_temperature()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(np.array(temperature_x)*1e6, label='X temperature')
        ax.plot(np.array(temperature_y)*1e6, label='Y temperature')
        ax.set_xlabel('Shot number')
        ax.set_ylabel('Temperature (uK)')
        ax.legend()

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')
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
            fig2, ax2 = plt.subplots(constrained_layout=True)
            for slope,intercept,T in zip(slope_lst,intercept_lst,T_lst):
                ax2.plot(t_plot**2*1e6 , (slope*t_plot**2 + intercept)*1e12, label=rf'$T = {T:LS}$ $\mu$K')

            i=0
            for waist in [waist_x, waist_y]:
                ax2.plot(time_of_flight**2*1e6, np.array(waist)**2*1e12, 'oC'+str(i), label = 'data')
                i+=1

            ax2.set_xlabel('Shot number')
            ax2.set_xlabel(r'Time$^2$ (ms$^2$)')
            ax2.set_ylabel(r'$\sigma^2$ ($\mu$m$^2$)')
            ax2.legend()
            ax2.set_ylim(ymin=0)
            ax2.tick_params(axis = 'x')
            ax2.tick_params(axis = 'y')

            ax2.grid(color='0.7', which='major')
            ax2.grid(color='0.9', which='minor')
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
        fig, ax = plt.subplots(constrained_layout=True)

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
        ax.set_xlabel(label_string)
        ax.set_ylabel('amplitude')
        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')

        fig.savefig(self.folder_path+'\\amplitude_vs_parameter.png')