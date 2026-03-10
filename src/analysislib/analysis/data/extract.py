# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:02:48 2020

@author: Quantum Engineer
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import inspect

from analysis.data import h5lyze as hz
from analysis.image.roi import  separateROIs



def loadNoiseData(fp, data_filename):
    with h5py.File(os.path.join(fp, data_filename), 'a') as f:
        dataGroup = f['data']
        
        paramDict = hz.datasetsToDictionary(dataGroup['parameters'])

        sequenceDict = hz.datasetsToDictionary(f['sequences'], recursive=True,
                                               ignore=['globals'])
        
        roiDict = hz.datasetsToDictionary(dataGroup['rois'])

                    
        return sequenceDict, paramDict, roiDict
    
def extractFraction(sequenceDict, rois, signalCalibration=None, clipFraction=True, 
                previewROI=False, previewKey='F4',fp='', returnStd=False, signalThreshold=None):
    '''
    Extract the F4 fraction for different sequences and tweezers from sequenceDict.

    Parameters
    ----------
    sequenceDict : dict
        Full sequence data dictionary.
    rois : dict
        Dictionary of ROI values for each saved roi. Only the keys of
        the rois are used here.
    signalCalibration : dict, optional
        If present, the dictionary with a and b parameters 
        for each tweezer. The default is None.
    clipFraction : bool, optional
        Clip the negative fraction values. The default is True.
    previewROI : bool, optional
         The default is False.
    previewKey : string, optional
        'F4' or 'F3'. The default is 'F4'.
    fp : string, optional
        File path to save the preview image. The default is ''.

    Returns
    -------
    meanFraction : TYPE
        DESCRIPTION.
    fractionArrays : TYPE
        DESCRIPTION.

    '''
    meanFraction = dict()
    stdFraction = dict()
    fractionArrays = dict()

    #get keys of separate tweezers and ROI keys for F=3 and F=4 in each tweezer
    splitROI = separateROIs(rois)            
    for twKey in splitROI:
        meanFraction[twKey] = np.zeros([len(sequenceDict)])
        stdFraction[twKey] = np.zeros([len(sequenceDict)])
        
    # for seqIndex, (sequence, sequenceSubDict) in enumerate(sorted(sequenceDict.items())):
    #     fractionArrays[seqIndex]={}
    #     data = sequenceSubDict['dataFullROI_extraBGremoval'] #TODO: change back to 'dataFullROI' without extra BG removal
                
    for seqIndex, (sequence, sequenceSubDict) in enumerate(sorted(sequenceDict.items())):
        fractionArrays[seqIndex]={}
        data = sequenceSubDict['dataFullROI'] #TODO: change back to 'dataFullROI' without extra BG removal (dataFullROI_extraBGremoval)
        
        for twKey, roiKeys in splitROI.items():
            F4 = np.copy(data[roiKeys['F4']])
            F3 = np.copy(data[roiKeys['F3']])
            F4 = np.transpose(F4, axes=[1,2,0])
            F3 = np.transpose(F3, axes=[1,2,0])
            #After this: axis 0: Y, axis 1: X, axis 2: Shot
            
            #signal scaling
            if signalCalibration and 'T' in twKey:
                idx = int(twKey[1])
                F3 = signalCalibration['a'][idx]*F3 - signalCalibration['b'][idx]*F4
                
            d = {'F4': np.copy(F4), 'F3': np.copy(F3)}
            previewImage = np.mean(d[previewKey],axis=2)
        
            #Average the signals in the ROI:
            for F in d:            
                d[F] = np.mean(d[F],axis=(0,1))
            
            if signalThreshold:
                totalSignal = d['F4']+d['F3']
                for F in d:            
                    d[F] = d[F][totalSignal>signalThreshold]
            
            fractionArrays[seqIndex][twKey]=1.*d['F4']/(d['F4']+d['F3'])
            meanFraction[twKey][seqIndex] = np.mean(d['F4']/(d['F4']+d['F3']))
            stdFraction[twKey][seqIndex] = np.std(d['F4']/(d['F4']+d['F3']))
            
            #fix values <0 or >1 (can reduce variance!)
            if clipFraction:
                #this way of clipping for consistency with the old code
                fractionArrays[seqIndex][twKey] = np.clip(fractionArrays[seqIndex][twKey],0,None)
                meanFraction[twKey][seqIndex] =  np.clip(meanFraction[twKey][seqIndex],0,1)

            if previewROI and (seqIndex==1):
                fig, ax = plt.subplots(figsize=(15,6))
                plt.imshow(previewImage)
                plt.title(previewKey + ' for ROI name \"'+twKey+'\"')
            
                plt.savefig(os.path.join(fp, sequence+'_PreviewOfROI_key'+twKey+'.png'))
                plt.show()
            
    if returnStd:
        return meanFraction, fractionArrays, stdFraction
    else:
        return meanFraction, fractionArrays

def extractMeanSignal(sequenceDict, rois, signalCalibration=None):
    '''
    Extracts mean signal from sequence dict.

    Parameters
    ----------
   sequenceDict : dict
        Full sequence data dictionary.
    rois : dict
        Dictionary of ROI values for each saved roi. Only the keys of
        the rois are used here.
    signalCalibration : dict, optional
        If present, the dictionary with a and b parameters 
        for each tweezer. The default is None.

    Returns
    -------
    meanSignal : dict
        Mean over repeats.
    signalArrays : dict
        All arrays.

    '''
    meanSignal = dict()
    meanSignalErr = dict()
    meanSignalF4 = dict()
    meanSignalErrF4 = dict()
    meanSignalDiff = dict()
    meanSignalDiffErr = dict()
    
    
    signalArrays = dict()
    signalArraysErr = dict()
    signalArraysF4 = dict()
    signalArraysF3 = dict()
    signalArraysDiff = dict()
    signalArraysDiffErr = dict()

    #get keys of separate tweezers and ROI keys for F=3 and F=4 in each tweezer
    splitROI = separateROIs(rois)            
        
    for twKey in splitROI:
        meanSignal[twKey] = np.zeros([len(sequenceDict)])
        meanSignalErr[twKey] = np.zeros([len(sequenceDict)])
        meanSignalF4[twKey] = np.zeros([len(sequenceDict)])
        meanSignalErrF4[twKey] = np.zeros([len(sequenceDict)])
        meanSignalDiff[twKey] = np.zeros([len(sequenceDict)])
        meanSignalDiffErr[twKey] = np.zeros([len(sequenceDict)])
    
    for seqIndex, (sequence, sequenceSubDict) in enumerate(sorted(sequenceDict.items())):
        data = sequenceSubDict['dataFullROI']
        
        signalArrays[seqIndex]={}
        signalArraysErr[seqIndex]={}
        signalArraysF4[seqIndex]={}
        signalArraysF3[seqIndex]={}
        signalArraysDiff[seqIndex]={}
        signalArraysDiffErr[seqIndex]={}
        
        for twKey, roiKeys in splitROI.items():
            F4 = np.copy(data[roiKeys['F4']])
            F3 = np.copy(data[roiKeys['F3']])
            #After this: axis 0: Shot, axis 1: y, axis 2: x
            
            #signal scaling
            if signalCalibration and 'T' in twKey:
                idx = int(twKey[1])
                F3 = signalCalibration['a'][idx]*F3 - signalCalibration['b'][idx]*F4
            
            #Average the signals in the ROI:
            meanSignal[twKey][seqIndex]=np.mean(F4+F3)
            meanSignalErr[twKey][seqIndex] = np.std(F4+F3)
            meanSignalF4[twKey][seqIndex]=np.mean(F4)
            meanSignalErrF4[twKey][seqIndex] = np.std(F4)
            meanSignalDiff[twKey][seqIndex] = np.mean(F4-F3)
            meanSignalDiffErr[twKey][seqIndex] = np.std(F4-F3)
            
            
            signalArrays[seqIndex][twKey]=np.mean(F4+F3,axis=(1,2))
            signalArraysErr[seqIndex][twKey] = np.std(F4+F3, axis=(1,2))
            signalArraysF4[seqIndex][twKey] = np.mean(F4, axis=(1,2))
            signalArraysF3[seqIndex][twKey] = np.mean(F3, axis=(1,2))
            signalArraysDiff[seqIndex][twKey]=np.mean(F4-F3,axis=(1,2))
            signalArraysDiffErr[seqIndex][twKey]=np.std(F4-F3,axis=(1,2))
            
    return meanSignal, meanSignalErr, meanSignalF4, meanSignalErrF4, signalArrays, signalArraysErr, signalArraysDiff, signalArraysDiffErr


def extractSignalFromAnalysisFile(data_filename,mode='signal',normTweezer='T0',verbose=False):
    #extract data
    with h5py.File(data_filename, 'a') as f:
        dataGroup = f['data']
        
        roiDict = hz.datasetsToDictionary(dataGroup['rois'])
    
        sequenceGroup = f['sequences']
        sequenceDict = hz.datasetsToDictionary(sequenceGroup, recursive=True, ignore=['globals'])
        
    areas = separateROIs(roiDict,'B')#[roi for roi in roiDict if f'T4' in roi]
    colors = {roi:c for roi,c in zip(areas,plt.cm.rainbow(np.linspace(0, 1, len(areas))))}
    
    n_tweezers = len(areas)
    
    data = {}
    for sequenceIdx, sequenceKey in enumerate(sorted(sequenceDict.keys())):
        data[sequenceIdx]={}
        dataDict = sequenceDict[sequenceKey]['data']
        
        signal={}
        #Analyze total counts
        for area, roi in areas.items():
            signal[area] = {}
            signal[area]['F3'] = dataDict[roi['F3']]
            signal[area]['F4'] = dataDict[roi['F4']]
            if len(dataDict[roi['F3']].shape)>1:
                if  dataDict[roi['F3']].shape[1]>1:
                    if verbose:
                        print('Data shape has a second dimension. Averaging across the second dimension.')
                
                    signal[area]['F3'] = np.mean(signal[area]['F3'],axis=1)
                    signal[area]['F4']  = np.mean(signal[area]['F4'] ,axis=1)
        
        #Get the appropriate signal
        for area, roi in areas.items():
            f3 = signal[area]['F3']  
            f4 = signal[area]['F4']
            
            denom=1
            num=f4

            if mode:
                if mode=='signal':
                    denom = 1
                elif mode=='ratio':
                    denom=f3+f4
                elif mode=='norm':
                    denom=signal[normTweezer]['F3']  +signal[normTweezer]['F4']
                elif mode=='sum':
                    num=f4+f3
                    
            data[sequenceIdx][area]=f4/denom
            
    return areas, data

def averageTweezerSignals(sequenceDict,tweezers_to_average):
    '''
    Average tweezer signals in sequenceDict based on tweezer indices in tweezers_to_average.

    Parameters
    ----------
    sequenceDict : dict
        Sequence data dictionary.
    tweezers_to_average : list
        List of lists of tweezer indices to average.

    Returns
    -------
    sequenceDict : dict
        Modified sequenceDict.
    avgROIDict : dict
        Added average tweezer keys.

    '''
    avgROIDict = {}
    
    for seq in sequenceDict:
        data = sequenceDict[seq]['dataFullROI']
    
        arr_shape = list(data.values())[0].shape
        
        for avgIdx, twIdxs in enumerate(tweezers_to_average):
            for manifold in ['3','4']:
                
                avgTweezerKey =f'A{manifold}_{avgIdx}'
                sequenceDict[seq]['dataFullROI'][avgTweezerKey]=np.zeros(arr_shape)   
                
                for twIdx in twIdxs:
                    sequenceDict[seq]['dataFullROI'][avgTweezerKey]+=\
                        data[f'T{manifold}_{twIdx}']
                    
               
                avgROIDict[avgTweezerKey] = twIdxs 
                
    return sequenceDict, avgROIDict

def tweezerFitAnalysis(data_filename, selectedParam, fitType, fitInitialConditions,mode='signal',
                  previewFits=False,colors=None,
                  fitLabel='',
                  fitCoefficient=None,
                  scaleFactor=1,normTweezer='T0',verbose=False, exclude_tw=[]):
    '''
    

    Parameters
    ----------
    data_filename : string
        Filename of the analysis h5 file.
    selectedParam : string
        Parameter name string.
    fitType : func
        Fit function.
    fitInitialConditions : list
        initial conditions.
    mode : string, optional
        Type of signal you want to fit. The default is 'signal'.
    previewFits : bool, optional
         The default is False.
    colors : dict, optional
        Dictionary of colors for different tweezers. The default is None.
    fitLabel : string, optional
        String for the fit description. The default is ''.
    fitCoefficient : int, optional
        Which fit coefficient to write on the plot. The default is None.
    scaleFactor : float, optional
        Scale factor. The default is 1.
    normTweezer : string, optional
        If mode is 'norm' use this tweezer to normalize the signals. The default is 'T0'.
    verbose : bool, optional
        The default is False.

    Returns
    -------
    paramVals : array
        DESCRIPTION.
    returnCoeffs : array
        DESCRIPTION.
    returnCoeffsErr : array
        DESCRIPTION.

    '''
    #extract data
    
    with h5py.File(data_filename, 'a') as f:
        dataGroup = f['data']
    
        paramDict = hz.datasetsToDictionary(dataGroup['parameters'])
        
        units = dict()
        for param in paramDict:
            units[param] = dataGroup['parameters'][param].attrs['unit']
    
        sequenceGroup = f['sequences']
        sequenceDict = hz.datasetsToDictionary(sequenceGroup, recursive=True, ignore=['globals'])
        
    
    #analyze data
    print(paramDict.keys())
    paramVals = paramDict[selectedParam]
    
    areas, data = extractSignalFromAnalysisFile(data_filename,mode=mode,normTweezer=normTweezer,verbose=verbose)
    colors = {roi:c for roi,c in zip(areas,plt.cm.rainbow(np.linspace(0, 1, len(areas))))}
    
    n_tweezers = len(areas)
    
    no_func_params =  len(inspect.signature(fitType).parameters.keys())-1
    
    returnCoeffs = np.zeros((len(sequenceDict),n_tweezers,no_func_params))
    returnCoeffsErr =  np.copy(returnCoeffs)
    
    for sequenceIdx, sequenceKey in enumerate(sorted(sequenceDict.keys())):
        singleSeqParams = {param:values for param, values in sequenceDict[sequenceKey]['parameters'].items()\
                           if param!=selectedParam}
        
        singleSeqParam = list(singleSeqParams.keys())[0]
        singleSeqParamVals = singleSeqParams[singleSeqParam]
        
   
        if previewFits:
            fig, ax = plt.subplots()
        
        for aridx, area in enumerate(areas):
            scaledParamVals = scaleFactor*singleSeqParamVals
            
            if previewFits:
                try:
                    color=colors[area]
                except:
                    print('Colors not defined.')
                    color='C0'
            
            ax.scatter(scaledParamVals, data[sequenceIdx][area], color=color)
            try:
                
                if callable(fitInitialConditions):
                    seed_params = fitInitialConditions(scaledParamVals, 
                                          data[sequenceIdx][area].squeeze())
                else:
                    seed_params = fitInitialConditions
                
                coeffs, covar = curve_fit(fitType, scaledParamVals, 
                                          data[sequenceIdx][area].squeeze(), 
                                          p0=seed_params)
                
                errs = np.sqrt(np.diag(covar))

            except RuntimeError as e:
                print('Fit failed')
                print(e)
                coeffs=np.zeros(no_func_params)
                errs=np.zeros(no_func_params)
            
            
            returnCoeffs[sequenceIdx,aridx]=coeffs
            returnCoeffsErr[sequenceIdx,aridx]=errs
    
            if previewFits:
                fitX = np.linspace(min(scaledParamVals),max(scaledParamVals),200)
                ax.plot(fitX,fitType(fitX,*coeffs),
                        label = fitLabel.format(coeffs[fitCoefficient], errs[fitCoefficient]),
                        color=color)
         
        if previewFits:
            ax.grid()
            
            ax.set_xlabel(singleSeqParam)
            ax.set_ylabel('f4')
            
            ax.set_title(selectedParam+f'={paramVals[sequenceIdx]:.3f}')
            plt.show()
            
    return paramVals, returnCoeffs,returnCoeffsErr