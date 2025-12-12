import cv2
import numpy as np
from scipy.constants import pi


class Contour:
    def __init__(self, opencv_contour):
        self._contour = opencv_contour
        self._moments = cv2.moments(self._contour)

    @property
    def area(self):
        return cv2.contourArea(self._contour)

    @property
    def perimeter(self):
        return cv2.arcLength(self._contour, True)

    @property
    def circularity(self):
        return 4 * pi * (self.area / self.perimeter**2)

    @property
    def centroid(self):
        if self.area == 0:
            return self._contour.mean(axis=0)[0]

        cx = self._moments['m10'] / self._moments['m00']
        cy = self._moments['m01'] / self._moments['m00']
        return np.array([cx, cy])
