import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import os
import h5py

from analysis.data import autolyze as az
from analysis.image import andorlyze as adl
from analysis.image.process import getSummedDataArray, extractROIDataSingleSequence, getParamArray
from analysis.image.roi import setROI, showROIPreview
from analysis.fit.functions import rabi_osc, quadratic,\
                                     cos_fit,rabi_osc_no_decay, rabi_osc_noDecayOffset
from analysis.graphic.export import dateString
from analysis.data.info import getTweezerAmplitudes
from analysis.data import h5lyze as hz


### Calibration functions ###

def extractCalibrationConstants(beta, amplitude, offset):
    '''
    Extract imaging calibration constants. This function is very model
    specific and is currently implementing the model from 08/01/2020.

    Parameters
    ----------
    beta : dict
        beta factors for each tweezer.
    amplitude : dict
        Amplitude of F=3 and F=4 rabi oscillations for each tweezer.
    offset : dict
        Amplitude of F=3 and F=4 rabi oscillations for each tweezer.


    Returns
    -------
    a_factor : dict
        a factors.
    b_factor : dict
        b factors.

    '''
    a_factor = {}
    b_factor = {}   
    alpha={}
    delta={}
    #Get final calibration
    for idx in beta:
        A4 = amplitude[4][idx]
        A3 = amplitude[3][idx]
        O3 = offset[3][idx]
        
        alpha[idx]=beta[idx]+A3/A4
        delta[idx]=(O3-A4*beta[idx])/(alpha[idx]*A4)
        
        a_factor[idx]=1/(alpha[idx]*(1+delta[idx]))
        
        b_factor[idx]=(1/(1+delta[idx]))*(beta[idx]/alpha[idx]+delta[idx])
        
        
    return a_factor, b_factor

def getImagingCalibration(directory,roiDict,
                          backgroundfp,
                          previewROIs, 
                          previewFit,
                          colors,
                          markers,
                          alphas,
                          fitType=rabi_osc_no_decay,
                          saveCalibrationConstants=False):
    '''
    Get imaging calibration from two runs, one with Rabi flopping and
    one with imaging 4,0 atoms when repumping after imaging.

    Parameters
    ----------
    directory : string
        Folder containing both runs.
    roiDict : dict
        roi dictionary.
    backgroundfp : string
        Background image path.
    previewROIs : dict
        roi dictionary.
    previewFit : bool
        
    colors : dict
        
    markers : dict
        
    alphas : dict
        
    fitType : TYPE, optional
        Fit function for the rabi oscillation. The default is rabi_osc_no_decay.
    saveCalibrationConstants : TYPE, optional
        Save a and b constatns to a .npy file. The default is False.

    Raises
    ------
    Exception
        Invalid sequence.

    Returns
    -------
    a_factor : dict
        
    b_factor : dict
        

    '''
    
    
    n_tweezers = len(roiDict[4].keys())
    
    #beta is the ratio N_3^R/M_4^R
    beta = {}
    
    offset = {}#O3 and O4 params
    amplitude={}#A3 and A4 params
    
    
    ###Fit parameters##3
    if fitType==rabi_osc:
        #this fit has a very weird offset values
        fitInitialConditions = {3: [300,0.025, 18, 3, 20, 0.5],
                                4: [300,0., 18, 3, 20, 0.5]}#A, t0, f, tau, offset, offset0
    if fitType == rabi_osc_no_decay:
        fitInitialConditions = {3:[300,0.025, 18, 12],
                                    4:[600,0., 18, 20]}#A, t0, f, offset
    if fitType == rabi_osc_noDecayOffset:
        fitInitialConditions = {3:[300,0.025, 18,3, 12],
                            4:[600,0., 18, 3,20]}#t, A,t0, f, tau, offset
    
    for folder in os.listdir(directory):        
        if not os.path.isdir(os.path.join(directory,folder)):
            continue
        
        print(f'Analyzing {folder}')
        ### ANALYSIS ##################################################################
        fp = os.path.join(directory, folder)
        
        imageArray = adl.getImageArray(az.getCompleteShots( az.getShots(fp)))
        
        # normalization
        norm_dict = {idx: [roiDict[4][idx], roiDict[3][idx]] for idx in roiDict[4].keys()}
        
        data, paramName, paramArray = extractROIDataSingleSequence(fp,roiDict,
                                                                    removeBackground=True,
                                                                    backgroundfp=backgroundfp,
                                                                    doNormalization=False,
                                                                    norm_dict=norm_dict)
            
        if previewROIs:  
            for manifold, rois in roiDict.items():
                previewROI = previewROIs[manifold]
    
                previewimg = np.mean(imageArray,axis=0)-np.load(backgroundfp)
                
                showROIPreview(previewimg,previewROIs[manifold],rois,title=f'F={manifold}',
                               roiLabel=f'F{manifold}',
                               filename=os.path.join(fp, f'{dateString()}_F={manifold}_preview.png'))
    
        if paramName == 'C_repeats':
            for idx in range(n_tweezers):
                beta[idx] = np.mean(data[3][idx])/np.mean(data[4][idx])
                
        elif paramName == 'MW1_Pulse1_Length':
            if previewFit:
                fig, ax = plt.subplots(figsize=(11,6))
            x = np.array(paramArray)*1e3
            
            for manifold, manifold_data in data.items():
                offset[manifold] = {}
                amplitude[manifold]={}            
                for idx, y in manifold_data.items(): 
                    
                    if previewFit:
                        c = colors[manifold][idx]
                        m = markers[manifold][idx]
                        a = alphas[manifold][idx]
                            
                        label = f'F{manifold}_{idx}'
                        
                        ax.scatter(x, y, color=c, label=label, marker=m, alpha=a)
    
                    coeffs, covar = curve_fit(fitType, x, y.squeeze(), p0=fitInitialConditions[manifold])
                    errs = np.sqrt(np.diag(covar))
                    
                    if fitType ==rabi_osc:
                        offset[manifold][idx]=coeffs[-2]
                    elif fitType==rabi_osc_no_decay or fitType==rabi_osc_noDecayOffset:
                        offset[manifold][idx]=coeffs[-1]
                        
                    amplitude[manifold][idx]=coeffs[0]
                    fitXvals = np.linspace(np.min(x),np.max(x),1000)
                    print(manifold)
                    print(coeffs)
                    
                    fitLabel = r'frequency =2$\pi$ {:.3f} +/-{:.3f} kHz '
                    
                    if previewFit:
                         ax.plot(fitXvals,
                            fitType(fitXvals,*coeffs),
                            color=c,
                            alpha=a,
                            label = fitLabel.format(coeffs[2], np.sqrt(np.diag(covar)[2])))
                
            if previewFit:
                ax.set_ylabel('Signal')
                ax.set_xlabel(paramName)
                ax.grid()
                ax.set_ylim(ymin = 0,ymax=700)
                
            plt.show()
    
        else:
            raise Exception('Invalid sequence for image calibration')
         
            
    a_factor, b_factor = extractCalibrationConstants(beta, amplitude, offset)
        
    
    if saveCalibrationConstants:
        save_array = np.stack([np.array(list(a_factor.values())),
                               np.array(list(b_factor.values()))])
        
        np.save(os.path.join(directory,'calibration.npy'),save_array)
        
    
    return a_factor, b_factor

# Takes in the new analysis file and adds it to the overall analysis file
def updateOffsetData(directory, total_file):   
    offset = {}
    offset_err ={} 
    print(f'Transferring fit data from {directory}')
    ### ANALYSIS ##################################################################
    fp = directory
    ### Check whether we analyzed the folder ###
    shots = az.getShots(fp)
    tw_amps = getTweezerAmplitudes(shots[0],'Spectrum6631', 0)
    TW_x_amplitude = az.getParam(shots[0], 'TW_x_amplitude')
    tw_amps = {str(k): v/TW_x_amplitude for k, v in tw_amps.items()}
    
    timestamp = az.getTimestamp(shots[0])
    # Editing the analysis h5 file
    with h5py.File(total_file, 'a') as f:
        
        sequences = f.require_group('sequences')
        if timestamp in sequences:
            print(f'Already analyzed folder. Skipping.')
    
        else:
            segmentGroup = sequences.require_group(timestamp)
            h5path = os.path.join(directory, 'analysis_results.h5')
            # Read data from the folder's analysis h5
            with h5py.File(h5path, mode='r') as f:
                data = hz.datasetsToDictionary(f['analysis'], recursive=True)
           # Loop over tweezers
            for key in data.keys():
                if 'err' in key:
                    offset_err[key[1]] = data[key][1]
                else:
                    offset[key[1]] = data[key][1]
                    
            hz.dictionaryToDatasets(segmentGroup.require_group('offset'), offset)
            hz.dictionaryToDatasets(segmentGroup.require_group('offset_err'), offset_err)
            hz.dictionaryToDatasets(segmentGroup.require_group('tw_amplitudes'), tw_amps)
            

def updateAmplitudeIterationData(directory,h5file,roiDict,
                          backgroundfp,
                          previewROIs, 
                          previewFit,
                          colors,
                          markers,
                          alphas):

    
    
    n_tweezers = len(roiDict[4].keys())
    
    
    phase = {}
    phase_err={} 

        
    print(f'Analyzing {directory}')
    ### ANALYSIS ##################################################################
    fp = directory
    ### Check whether we analyzed the folder ###
    shots = az.getShots(fp)
    tw_amps = getTweezerAmplitudes(shots[0],'Spectrum6631', 0)
    
    TW_x_amplitude = az.getParam(shots[0], 'TW_x_amplitude')
    
    tw_amps = {str(k): v/TW_x_amplitude for k, v in tw_amps.items()}
    
    timestamp = az.getTimestamp(shots[0])
    with h5py.File(h5file, 'a') as f:
        
        sequences = f.require_group('sequences')
        if timestamp in sequences:
            print(f'Already analyzed folder. Skipping.')
    
        else:
            segmentGroup = sequences.require_group(timestamp)
            
            imageArray = adl.getImageArray(shots)
            
            # normalization
            norm_dict = {idx: [roiDict[4][idx], roiDict[3][idx]] for idx in roiDict[4].keys()}
            
            data = extractROIDataSingleSequence(fp,roiDict, removeBackground=True, backgroundfp=backgroundfp, doNormalization=True, norm_dict=norm_dict)
            
            paramName, paramArray = getParamArray(shots, param_idx=0)
            
            if previewROIs:  
                for manifold, rois in roiDict.items():
                    previewROI = previewROIs[manifold]
        
                    previewimg = np.mean(imageArray,axis=0)-np.load(backgroundfp)
                    
                    showROIPreview(previewimg,previewROIs[manifold],rois,title=f'F={manifold}',
                                   roiLabel=f'F{manifold}',
                                   filename=os.path.join(fp, f'{dateString()}_F={manifold}_preview.png'))
        
        
            if previewFit:
                fig, ax = plt.subplots(figsize=(11,6))
            x = np.array(paramArray)
            
            manifold=4
            fitInitialConditions = {4:  [0,0.6,0.5] }
            fitBounds = ([-180, 0, -np.inf], [180, np.inf, np.inf])
            
            manifold_data=data[manifold]
            for idx, y in manifold_data.items(): 
                
                if previewFit:
                    c = colors[manifold][idx]
                    m = markers[manifold][idx]
                    a = alphas[manifold][idx]
                        
                    label = f'F{manifold}_{idx}'
                    
                    ax.scatter(x, y, color=c, label=label, marker=m, alpha=a)
    
                coeffs, covar = curve_fit(cos_fit, x, y.squeeze(), 
                                          p0=fitInitialConditions[manifold],
                                          bounds=fitBounds)
                errs = np.sqrt(np.diag(covar))
                

                    
                phase[str(idx)]=coeffs[0]
                phase_err[str(idx)]=errs[0]
                fitXvals = np.linspace(np.min(x),np.max(x),1000)
                
                fitLabel = r'offset = {:.1f}+/-{:.1f} degrees'
                
                if previewFit:
                     ax.plot(fitXvals,
                        cos_fit(fitXvals,*coeffs),
                        color=c,
                        alpha=a,
                        label = fitLabel.format(phase[str(idx)], phase_err[str(idx)]))
                
            if previewFit:
                ax.set_ylabel('Fraction')
                ax.set_xlabel(paramName)
                ax.grid()
                ax.set_ylim(ymin = 0,ymax=1.)
                ax.legend(loc='upper left',bbox_to_anchor=(1, 1),prop={'size': 8})
                
            plt.show()
            
            
            hz.dictionaryToDatasets(segmentGroup.require_group('phase'), phase)
            hz.dictionaryToDatasets(segmentGroup.require_group('phase_err'), phase_err)
            hz.dictionaryToDatasets(segmentGroup.require_group('tw_amplitudes'), tw_amps)

            

### Calibration dictionaries ##

class SignalCalibration:
    
    def __init__(self,target,root,
                 d = None, 
                 m = None, 
                 y = None,description=''):
        
        self.target=target
        self.directory = az.getDirectory(target,d = d, m = m, y = y)
        self.root=root
        
        self.description = description
        
        try:
            self.factors = np.load(os.path.join(self.directory,root,'calibration.npy'))
        except:
            raise Exception(f"Can't find calibration in {os.path.join(self.directory,root)}")
            
        self.noTweezers = self.factors.shape[1]
        self.a_factors = {idx:val for idx, val in enumerate(self.factors[0])}
        self.b_factors = {idx:val for idx, val in enumerate(self.factors[1])}
        
    def __getitem__(self,key):
        
        if key=='a':
            return self.a_factors
        elif key=='b':
            return self.b_factors
        else:
            raise IndexError('Invalid index. Index can be "a" or "b".')
            

### Dictionary ###


# calibrations = {
# '20201014':SignalCalibration(r'//RYD-EXPTCTRL/data_12_2018-03_2020/Experiments/rydberglab/RunExperiment_inTweezers','154500_imagingCalibration',\
#                                   d=14,m=10,y=2020,description='Exposure 100us, Andor Exposure 185 us, 5.5 MHz'),
# '20201102':SignalCalibration('RunExperiment_inTweezers','210000__importantNoiseCalibration',\
#                                   d=2,m=11,y=2020,description='Exposure 100us, Andor Exposure 285 us, 5.5 MHz'),
# '20201104_MW':SignalCalibration('RunExperiment_inTweezers','164500__noiseMeasurement_200rep_50usImagingWithMicrowave_calibration',\
#                                   d=4,m=11,y=2020,description='Exposure 50us, IM AM 0p45, with microwaves'),
# '20201104':SignalCalibration('RunExperiment_inTweezers','165000__noiseMeasurement_200rep_50usImagingWithRepump_calibration',\
#                                   d=4,m=11,y=2020,description='Exposure 50us, IM AM 0p45'),   
     
#  }
        
