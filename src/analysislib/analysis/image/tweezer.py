# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:22:33 2020

@author: Quantum Engineer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import datetime
from scipy import interpolate
from PIL import Image

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


def tweezerSignals(fp, shots, sequenceIdx = 'singleSequence', saveBlobs = False, radius=15):
    
    if saveBlobs:
        blobPath = os.path.join(fp, "blobs\\") 
        if not os.path.isdir(blobPath):
            os.mkdir(blobPath)

    date = datetime.datetime.today().strftime('%Y%m%d')
    roi = ((200, 1600),(270, 470))
    
    radius = radius
    method = 'sum'
    blob_kwargs = {'min_sigma':radius, 'max_sigma':radius,'threshold':0.2}
    


    imageArray = mtl.getImageArray(shots).astype(float)
    
    imageArray = np.flip(imageArray, axis=2)
        
    imageArray = cropImageArray(imageArray, roi)
    


    #extract background
    bgArray = cropImageArray(imageArray, ((0,2*radius),(0,2*radius)))


    blob_locations, blob_rois = extractReferenceBlobs(np.mean(imageArray,axis=0)\
    
                                                       ,roi_pad=2,**blob_kwargs)
    
    #just show blobs
    fig, ax = showBlobsImage(np.mean(imageArray,axis=0),**blob_kwargs)
    if saveBlobs:
        fig.savefig(os.path.join(blobPath,f"{date}_DetectedBlobs_imidx_{sequenceIdx}_{len(blob_locations)}_traps.png"),bbox_inches='tight',\
                dpi=250)
    
    signals = {} #np.zeros((imageArray.shape[0],len(blob_locations)))
    for ridx, blob_roi in enumerate(blob_rois):
        signals_roi,_ = getSummedDataArray(imageArray, blob_roi)
        signals_roi=np.array(signals_roi)*(blob_roi[0][1]-blob_roi[0][0])*(blob_roi[1][1]-blob_roi[1][0])

        signals[ridx] = signals_roi.squeeze()
    

    return blob_locations, signals

def tweezerSignalsFromDirectory(fp, averageOverSequences=False, averageOverTweezers=False, saveBlobs=False, paramIdx=0):
    
    folders = [os.path.join(fp, name) for name in os.listdir(fp) if os.path.isdir(os.path.join(fp, name)) and name!="blobs"]

    signals = {}
    averagedSignals = {}
    
    ### Select parameters ####
    # paramDict, units, singleSeqParamDict = az.getSequencesParameter(folders)
    paramDict, units, singleSeqParamDict = az.getSequencesParameter(folders)
    if len(paramDict)==0:
        if len(singleSeqParamDict)>0:
            #in this case select the parameter changing within a sequence
            param = list(singleSeqParamDict.keys())[0]
            paramVals = singleSeqParamDict[param]
        else:
            param = 'none'
            paramVals = np.array([0])
    else:
        param = list(paramDict.keys())[paramIdx]
        paramVals = paramDict[param]
        
    ### convert any string paramVals to numbers
    plottingParams = {}
    for seqDesc in paramVals:
        if seqDesc.dtype.type==np.str_:
            plottingParams[seqDesc] = [int(s) for s in re.findall(r'\d+', seqDesc)][-1]
        else:
            plottingParams[seqDesc] = seqDesc
    
    for fidx, folder in enumerate(folders):
        shots = az.getShots(folder)
        singleSeqSignals = tweezerSignals(fp, shots, sequenceIdx = paramVals[fidx], saveBlobs=saveBlobs)
        signals[paramVals[fidx]] = singleSeqSignals
        
    if averageOverSequences:
        print('Averaging over sequences...')
        for key, val in signals.items():
            for tweezer in val:
                val[tweezer] = np.mean(val[tweezer])
    elif averageOverTweezers:
        print('Averaging over tweezers...')
        singleSequenceParam = list(singleSeqParamDict.keys())[0]
        singleSequenceParamVals = singleSeqParamDict[singleSequenceParam]
        
        for paramVal in paramVals:
            averagedSignals[paramVal] = {}
            for pidx, seqParamVal in enumerate(singleSequenceParamVals):
                averagedSignals[paramVal][seqParamVal] = np.mean([signals[paramVal][ii][pidx] for ii in range(len(signals[paramVal]))])
        signals = averagedSignals
    else:
        print('Not averaging...')
        signals = signals
            
    return signals, plottingParams

