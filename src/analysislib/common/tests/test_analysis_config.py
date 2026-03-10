import pytest
import numpy as np

from analysislib.common.analysis_config import ImagingCamera, ImagingSystem


class ImagingSystemTest:
    def test_solid_angle_lens_choice(self):
        camera = ImagingCamera(
            pixel_size=5.5e-6,
            image_size=2048,
            quantum_efficiency=0.4,
            gain=1,
            image_group_name='manta419b_mot_images',
            image_name_stem='manta419b_mot',
        )
        system = ImagingSystem(
            imaging_f=200e-3,
            objective_f=100e-3,
            lens_diameter=(100e-3 * np.sqrt(3)),  # gives 60 degree half angle
            imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
            camera=camera,
        )

        assert system.solid_angle_fraction == 0.25
