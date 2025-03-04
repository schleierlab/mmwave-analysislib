# -*- coding: utf-8 -*-
"""
Created on Thu March 3 2025

@author: lin, tony
"""
from dataclasses import dataclass
import sys
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    lyse
except:
    import lyse
from analysis.data import h5lyze as hz
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv


@dataclass
class ImagingCamera:
    pixel_size: float
    """Pixel size, in m"""

    image_size: float
    """Size of sensor, in pixels (assumes square sensor)"""

    quantum_efficiency: float

    gain: float
    """
    Additional conversion from detected electron counts to camera counts,
    (for cameras like EMCCDs).
    """


@dataclass
class ImagingSetup:
    imaging_f: float
    """Focal length of imaging lens, in m"""

    objective_f: float
    """Focal length of objective lens, in m"""

    lens_diameter: float  # diameter of light collection lenses
    """Diameter of imaging and objective lenses (or the smaller of the two), in m"""

    imaging_loss: float
    """Fractional power loss along imaging path."""

    camera: ImagingCamera
    """Detection camera"""

    @property
    def magnification(self):
        return self.imaging_f / self.objective_f

    @property
    def solid_angle_fraction(self):
        """
        Solid angle captured by the imaging setup as a fraction of 4\pi.
        Makes a small angle approximation for the imaging NA.
        """
        return (self.lens_diameter / 2)**2 / (4 * self.imaging_f**2)

    def counts_per_atom(self, scattering_rate, exposure_time):
        """
        For an image of atoms with given scattering rate taken with a given exposure time,
        find the camera counts per atom we expect with this imaging setup.

        Parameters
        ----------
        scattering_rate: float
            Scattering rate (\Gamma) for atoms being imaged.
        exposure_time: float
            Exposure time of the image in seconds.

        Returns
        -------
        float
            Camera counts per atom for this imaging setup.
        """
        count_rate_per_atom = (
            scattering_rate/2
            * self.solid_angle_fraction
            * self.imaging_loss
            * self.camera.quantum_efficiency
            * self.camera.gain
        )
        return count_rate_per_atom * exposure_time


manta = ImagingCamera(
    pixel_size=5.5e-6,
    image_size=2048,
    quantum_efficiency=0.4,
    gain=1,
)


manta_path = ImagingSetup(
    imaging_f=50e-3,
    objective_f=125e-3,
    lens_diameter=25.4e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=manta,
)


class BulkGasAnalysis:
    scattering_rate = 2 * np.pi * 5.2227e+6
    """ Cesium scattering rate """

    def __init__(self, imaging_setup: ImagingSetup, exposure_time, atoms_roi, bkg_roi):
        """
        Parameters
        ----------
        imaging_f, objective_f: float
            focal lengths of the imaging and objective lenses, in m
        lens_diameter: float
            imaging lens diameter in m
        exposure_time: float
            imaging exposure time in s
        atoms_roi, bkg_roi: array_like, shape (2, 2)
            Specification of the regions of interest for the atoms and for the background image.
            The specification should take the form
            [
                [x_min, x_max],
                [y_min, y_max],
            ],
            where all numbers are given in pixels.
        """
        self.atoms_roi = atoms_roi
        self.background_roi = bkg_roi

        self.imaging_setup = imaging_setup
        self.counts_per_atom = self.imaging_setup.counts_per_atom(
            self.scattering_rate,
            exposure_time,
        )

        self.images, self.run_number = self.load_image_in_lyse()
        self.mot_image, self.background_image, self.sub_image = self.get_image_bkg_sub()
        self.roi_atoms, self.roi_bkg = self.get_images_roi()

    def load_image_in_lyse(self):
        """
        load image using the h5 file path that is active in lyse

        Returns
        -------
        images: list with keys as manta0, manta1 as the 1st image and second image
        can be called by images[image_types[0,1]] as the signal and background
        """
        # Is this script being run from within an interactive lyse session?
        if lyse.spinning_top:
            # If so, use the filepath of the current h5_path
            h5_path = lyse.path
        else:
            # If not, get the filepath of the last h5_path of the lyse DataFrame
            df = lyse.data()
            h5_path = df.filepath.iloc[-1]

        self.h5_path = h5_path

        with h5py.File(h5_path, mode='r+') as f:
            globals = hz.attributesToDictionary(f['globals'])
            info_dict = hz.getAttributeDict(f)
            images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)
            run_number = f.attrs['run number']
            # self.images = images
            # self.run_number = f.attrs['run number']

        return images, run_number



    def get_image_bkg_sub(self, debug = False):
        """
        get background subtracted image: signal - background

        Returns
        -------
        sub_image = mot_image - background_image
        array like, shape [n_px, n_px]
        """
        images = self.images
        image_types = list(images.keys())
        if debug:
            print("image type is", images[image_types[0]])
        mot_image = images[image_types[0]] # 1st shot is signal
        background_image = images[image_types[1]] # 2nd shot is background
        sub_image = mot_image - background_image # subtraction of the background

        # self.mot_image = mot_image
        # self.background_image = background_image
        # self.sub_image = sub_image

        return mot_image, background_image, sub_image

    def get_images_roi(self,):
        """
        Returns
        -------
        roi_atoms, roi_bkg as the images and background in the roi
        array like, shape [roi_y.size, roi_x.size] and [roi_y_bkg.size, roi_x_bkg.size]
        """
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi
        sub_image = self.sub_image
        roi_atoms = sub_image[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        roi_bkg = sub_image[roi_y_bkg[0]:roi_y_bkg[1], roi_x_bkg[0]:roi_x_bkg[1]]

        # self.roi_atoms = roi_atoms
        # self.roi_bkg = roi_bkg

        return roi_atoms, roi_bkg

    def get_atom_number(self,):
        """
        Returns
        -------
        int
        atom number = sum of atom number in roi - average background atom* roi size
        """

        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        electron_counts_mot = roi_atoms.sum()
        electron_counts_bkg = roi_bkg.sum()
        atom_number_withbkg = electron_counts_mot / self.counts_per_atom
        bkg_number = electron_counts_bkg / self.counts_per_atom / roi_bkg.size * roi_atoms.size # average bkg floor in the size of roi_atoms
        atom_number = int(atom_number_withbkg) - int(bkg_number)
        # self.atom_number = atom_number

        self.save_atom_number(atom_number)

        return atom_number

    def plot_images(self, image_scale = 100):
        """
        Parameters
        ----------
        image_scale: float
            scale of the image, max value of the colorbar

        plot images of raw image/background image full frame and background subtracted image in rois
        """
        mot_image = self.mot_image
        background_image = self.background_image
        roi_atoms = self.roi_atoms
        roi_bkg = self.roi_bkg
        [roi_x, roi_y] = self.atoms_roi
        [roi_x_bkg, roi_y_bkg] = self.background_roi

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        (ax_mot_raw, ax_bkg_raw), (ax_bkg_roi, ax_mot_roi) = axs
        for ax in axs[0]:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
        for ax in axs[1]:
            ax.set_xlabel('x [um]')
            ax.set_ylabel('y [um]')



        raw_img_color_kw = dict(cmap='viridis', vmin=0, vmax= image_scale)
        ax_mot_raw.set_title('Raw')
        pos = ax_mot_raw.imshow(mot_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_mot_raw)

        ax_bkg_raw.set_title('Raw, no mot')
        pos = ax_bkg_raw.imshow(background_image, **raw_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_raw)

        pixel_size_after_maginification = self.imaging_setup.camera.pixel_size*self.imaging_setup.magnification
        ax_mot_roi.set_title('mot ROI')
        pos = ax_mot_roi.imshow(
            roi_atoms,
            extent=np.array([roi_x[0], roi_x[1], roi_y[0], roi_y[1]])*pixel_size_after_maginification,
            **raw_img_color_kw,
        )
        fig.colorbar(pos, ax=ax_mot_roi)

        ax_bkg_roi.set_title('Background ROI')
        pos = ax_bkg_roi.imshow(
            roi_bkg,
            vmin=-10,
            vmax=10,
            extent=np.array([roi_x_bkg[0], roi_x_bkg[1], roi_y_bkg[0], roi_y_bkg[1]])*pixel_size_after_maginification, #factor: px*mag
        )
        fig.colorbar(pos, ax=ax_bkg_roi)

        return

    def save_atom_number(self, atom_number):
        """
        save atom number to data.csv
        if run number is 0, it will create a new file
        otherwise it will append
        """
        h5_path = self.h5_path
        run_number = self.run_number

        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        if run_number == 0:
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{atom_number}\n')
        else:
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{atom_number}\n')
        return

    def load_atom_number(self,):
        """
        load the atom number that is already saved in data.csv

        Returns
        -------
        counts: float

        """
        h5_path = self.h5_path
        folder_path = '\\'.join(h5_path.split('\\')[0:-1])
        count_file_path = folder_path+'\\data.csv'
        with open(count_file_path, newline='') as csvfile:
            counts = [list(map(float, row))[0] for row in csv.reader(csvfile)]

        return counts


    def plot_atom_number(self,):
        """
        plot atom number from data.csv vs the shot number

        """
        counts = self.load_atom_number()
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(counts)
        ax.set_xlabel('Shot number')
        ax.set_ylabel('mot atom count')

        ax.grid(color='0.7', which='major')
        ax.grid(color='0.9', which='minor')


