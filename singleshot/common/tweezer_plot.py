import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

from .plot_config import PlotConfig
from .image import ROI


class TweezerPlotter:
    """Class for plotting tweezer analysis results.

    This class provides methods for visualizing tweezer analysis results.

    Parameters
    ----------
    plot_config : PlotConfig, optional
        Configuration object for plot styling
    """
    def __init__(self, h5_path, plot_config: PlotConfig = None):
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(h5_path)

    def _load_processed_quantities(self, h5_path):
        """Load processed quantities from an h5 file.

        Parameters
        ----------
        h5_path : str
            Path to the processed quantities h5 file
        """
        with h5py.File(h5_path, 'r') as f:
            self.camera_counts = f['camera_counts'][:]
            self.site_occupancies = f['site_occupancies'][:]
            self.site_rois = ROI.fromarray(f['site_rois'])

    # def calculate_survival_rate(self, site_occupancy, method='default') -> tuple[float, float]:
    #         """Calculate survival rate (and uncertainty thereof) from atom existence lists.

    #         Parameters
    #         ----------
    #         atom_exist_lst_1, atom_exist_lst_2 : array_like, shape (n_sites,)
    #             Lists of atom existence values for the two tweezer images.
    #         method : {'default', 'laplace'}, optional
    #             Method for calculating the survival rate and uncertainty.
    #             Defaults to 'default' (which does the usual thing)

    #             'laplace': estimate the survival rate (and associated uncertainty)
    #             using Laplace's rule of succession (https://en.wikipedia.org/wiki/Rule_of_succession),
    #             whereby we inflate the number of atoms by 2 and the number of survivors by 1.
    #             Seems to be a better metric for closed-loop optimization by M-LOOP.
    #         """
    #         atom_exist_lst = site_occupancy
    #         n_initial_atoms = np.sum(atom_exist_lst[0])
    #         survivors = sum(1 for x,y in zip(atom_exist_lst[0], atom_exist_lst[1])
    #                 if x == 1 and y == 1)

    #         if method == 'default':
    #             survival_rate = survivors / n_initial_atoms
    #             uncertainty = np.sqrt(survival_rate * (1 - survival_rate) / n_initial_atoms)
    #         elif method == 'laplace':
    #             # expectation value of posterior beta distribution
    #             survival_rate = (survivors + 1) / (n_initial_atoms + 2)

    #             # calculate based on binomial distribution
    #             # uncertainty = np.sqrt(survival_rate * (1 - survival_rate) / (n_initial_atoms + 2))

    #             # sqrt of variance of the posterior beta distribution
    #             uncertainty = np.sqrt((survivors + 1) * (n_initial_atoms - survivors + 1) / ((n_initial_atoms + 3) * (n_initial_atoms + 2) ** 2))
    #         return survival_rate, uncertainty

    # def plot_images(self, show_site_roi=True, plot_bkg_roi=True):
    #     """Plot the analysis results with optional site ROIs and background ROIs.

    #     Parameters
    #     ----------
    #     show_site_roi : bool, default=True
    #         Whether to show site ROI rectangles
    #     plot_bkg_roi : bool, default=True
    #         Whether to plot background ROI images
    #     """
    #     num_of_imgs = len(self.sub_images)

    #     # Get site counts and analyze existence for each image
    #     # TODO: load atom_exist_lst from file instead
    #     rect_sig = []
    #     atom_exist_lst = []
    #     for i in range(num_of_imgs):
    #         sub_image = self.sub_images[i]
    #         roi_number_lst_i = self.tweezer_analyzer.get_site_counts(sub_image)
    #         rect_sig_i, atom_exist_lst_i = self.tweezer_analyzer.analyze_site_existence(roi_number_lst_i)
    #         rect_sig.append(rect_sig_i)
    #         atom_exist_lst.append(atom_exist_lst_i)

    #     # Create figure with enough rows for all shots plus background ROIs
    #     num_rows = num_of_imgs + (2 if plot_bkg_roi else 0)
    #     fig_height = self.plot_config.figure_size[1] * num_rows
    #     fig, axs = plt.subplots(nrows=num_rows, ncols=1,
    #                            figsize=(self.plot_config.figure_size[0], fig_height),
    #                            constrained_layout=self.plot_config.constrained_layout)

    #     # If only one subplot, wrap it in a list for consistent indexing
    #     if num_rows == 1:
    #         axs = [axs]

    #     plt.rcParams.update({'font.size': self.plot_config.font_size})
    #     fig.suptitle('Tweezer Array Imaging Analysis', fontsize=self.plot_config.title_font_size)

    #     # Configure all axes
    #     for ax in axs:
    #         ax.set_xlabel('x [px]')
    #         ax.set_ylabel('y [px]')
    #         ax.tick_params(axis='both', which='major', labelsize=self.plot_config.label_font_size)

    #     roi_img_color_kw = dict(cmap=self.plot_config.colormap, vmin=0, vmax=self.plot_config.roi_image_scale)

    #     # Plot tweezer ROIs for each shot
    #     for i in range(num_of_imgs):
    #         ax_tweezer = axs[i]
    #         ax_tweezer.set_title(f'Shot {i+1} Tweezer ROI', fontsize=self.plot_config.title_font_size, pad=10)
    #         pos = ax_tweezer.imshow(self.roi_atoms[i], **roi_img_color_kw)

    #         if show_site_roi:
    #             # Draw the base ROI rectangles
    #             site_roi_x = self.site_roi[0]
    #             site_roi_y = self.site_roi[1]
    #             base_rects = []
    #             for j in range(site_roi_x.shape[0]):
    #                 y_start, y_end = site_roi_y[j,0], site_roi_y[j,1]
    #                 x_start, x_end = site_roi_x[j,0], site_roi_x[j,1]
    #                 rect = patches.Rectangle(
    #                     (x_start, y_start),
    #                     x_end - x_start,
    #                     y_end - y_start,
    #                     linewidth=1,
    #                     edgecolor='r',
    #                     facecolor='none',
    #                     alpha=0.5,
    #                     fill=False)
    #                 base_rects.append(rect)
    #             pc_base = PatchCollection(base_rects, match_original=True)
    #             ax_tweezer.add_collection(pc_base)

    #             # Draw the signal rectangles
    #             if i < len(rect_sig):
    #                 pc_sig = PatchCollection(rect_sig[i], match_original=True)
    #                 ax_tweezer.add_collection(pc_sig)

    #         fig.colorbar(pos, ax=ax_tweezer).ax.tick_params(labelsize=self.plot_config.label_font_size)

    #     # Plot background ROIs if requested
    #     if plot_bkg_roi:
    #         for i in range(min(2, num_of_imgs)):  # Plot up to 2 background ROIs
    #             ax_bkg = axs[num_of_imgs + i]
    #             ax_bkg.set_title(f'Shot {i+1} Background ROI', fontsize=self.plot_config.title_font_size, pad=10)
    #             pos = ax_bkg.imshow(self.roi_bkgs[i], **roi_img_color_kw)
    #             fig.colorbar(pos, ax=ax_bkg).ax.tick_params(labelsize=self.plot_config.label_font_size)

    def plot_survival_rate(self, fig=None, ax=None):
        if (fig is None) != (ax is None):
            raise ValueError
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
        initial_atoms = self.site_occupancies[:, 0].sum(axis=-1)

        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.product(self.site_occupancies[:, :2], axis=1).sum(axis=-1)

        survival_rates = surviving_atoms / initial_atoms
        ax.plot(
            np.arange(len(self.site_occupancies)),
            survival_rates,
            marker='.',
        )
