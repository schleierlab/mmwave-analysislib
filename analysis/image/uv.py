# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:39:38 2022

@author: Quantum Engineer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import datetime
from scipy import interpolate
from scipy import misc
from PIL import Image, ImageOps

from analysis.data import autolyze as az
from analysis.data.info import getTweezerRunInfo
from analysis.image import andorlyze as andlz
from analysis.image import mantalyze as mtl
from analysis.image.process import extractReferenceBlobs,extractBlobLocations,\
                                    getSummedDataArray,showBlobsImage
from analysis.image.process import cropImageArray
from tweezerlyze.detection import DetectionBot
from analysis.graphic.plotting import drawGrid
import re

plt.rcParams.update({'font.size':14})
plt.rcParams['image.cmap'] = 'viridis'


def uvSignal(fp, image, roi, radius=5, threshold = 0.4, sequenceIdx = 'singleSequence', saveBlobs = False):
    
    if saveBlobs:
        blobPath = os.path.join(fp, "blobs\\") 
        if not os.path.isdir(blobPath):
            os.mkdir(blobPath)

    date = datetime.datetime.today().strftime('%Y%m%d')
    roi = roi
    
    radius = radius
    method = 'sum'
    blob_kwargs = {'min_sigma':radius, 'max_sigma':radius,'threshold': threshold}
    
    im = Image.open(os.path.join(fp, image))
    # grayIm = ImageOps.grayscale(im)
    im = im.convert('L')
    # im = misc.imread(os.path.join(fp, image), flatten= 1)
    # imageArray = mtl.getImageArray(shots).astype(float)
    imageArray = np.array(im)
    
    # imageArray = np.flip(imageArray, axis=2)

    imageArray = cropImageArray(imageArray, roi)
    

    #extract background
    bgArray = cropImageArray(imageArray, ((0,2*radius),(0,2*radius)))


    blob_locations, blob_rois = extractReferenceBlobs(imageArray,
                                                      roi_pad=2,**blob_kwargs)
    
    # blob_locations = blob_locations + np.array([roi[0][0], roi[1][0]])
    # #just show blobs
    fig, ax = showBlobsImage(imageArray,**blob_kwargs)
    # if saveBlobs:
    #     fig.savefig(os.path.join(blobPath,f"{date}_DetectedBlobs_imidx_{sequenceIdx}_{len(blob_locations)}_traps.png"),bbox_inches='tight',\
    #             dpi=250)
    
    # signals = {} #np.zeros((imageArray.shape[0],len(blob_locations)))
    # for ridx, blob_roi in enumerate(blob_rois):
    #     signals_roi,_ = getSummedDataArray(imageArray, blob_roi)
    #     signals_roi=np.array(signals_roi)*(blob_roi[0][1]-blob_roi[0][0])*(blob_roi[1][1]-blob_roi[1][0])

    #     signals[ridx] = signals_roi.squeeze()
    

    # return blob_locations, signals
    
    return blob_locations