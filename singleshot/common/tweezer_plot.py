class TweezerPlotter:
    def __init__(self, tweezer_analyzer):
        self.tweezer_analyzer = tweezer_analyzer

    def plot_images(self, roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True):
        """Plot the analysis results with optional site ROIs and background ROIs.
        
        Parameters
        ----------
        roi_image_scale : int, default=150
            Scale factor for ROI image colormaps
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
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 5*num_rows), constrained_layout=True)
        
        # If only one subplot, wrap it in a list for consistent indexing
        if num_rows == 1:
            axs = [axs]

        plt.rcParams.update({'font.size': 14})
        fig.suptitle('Tweezer Array Imaging Analysis', fontsize=16)

        # Configure all axes
        for ax in axs:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
            ax.tick_params(axis='both', which='major', labelsize=12)

        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

        # Plot tweezer ROIs for each shot
        for i in range(num_of_imgs):
            ax_tweezer = axs[i]
            ax_tweezer.set_title(f'Shot {i+1} Tweezer ROI', fontsize=14, pad=10)
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
                        edgecolor='blue',
                        facecolor='none',
                        alpha=0.3)
                    base_rects.append(rect)
                pc_base = PatchCollection(base_rects, match_original=True)
                ax_tweezer.add_collection(pc_base)
                
                # Draw the signal rectangles
                if i < len(rect_sig):
                    pc_sig = PatchCollection(rect_sig[i], match_original=True)
                    ax_tweezer.add_collection(pc_sig)
            
            fig.colorbar(pos, ax=ax_tweezer).ax.tick_params(labelsize=12)

        # Plot background ROIs if requested
        if plot_bkg_roi:
            for i in range(min(2, num_of_imgs)):  # Plot up to 2 background ROIs
                ax_bkg = axs[num_of_imgs + i]
                ax_bkg.set_title(f'Shot {i+1} Background ROI', fontsize=14, pad=10)
                pos = ax_bkg.imshow(self.roi_bkgs[i], **roi_img_color_kw)
                fig.colorbar(pos, ax=ax_bkg).ax.tick_params(labelsize=12)
