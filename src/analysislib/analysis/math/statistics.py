# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:49:09 2020

@author: Jacob
"""

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def runningMean(x, N, step=1):
    """
    Function which performs a running mean on a 1D array x
    over a window of size N, stepping by step.
    """
    arrCumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (arrCumsum[N::step] - arrCumsum[:-N:step]) / float(N)

def calcFractionRegression(fractions, doRegressionTime=False, 
                           doRegressionSpace=False, spaceRegressionType='nearest_neighbor',
                           fractional=True):
    '''
    Take a dictionary f4 fraction values for different sequences and tweezers within
    each sequence and infer the variance of f4-f3.

    Parameters
    ----------
    fractions : dict
        The dictionary of different sequences, with sub dictionaries of different
        tweezers.
    doRegressionTime : bool, optional
         The default is False.
     doRegressionSpace: bool, optional
         The default is False.

    Returns
    -------
    variance : dict
        Dictionary of different ROIs with the array of variances of f4-f3 for
        each sequence.
    variance_error : dict
        Standard error of the variances.

    '''
    variance = dict()
    variance_error = dict()
    
    roi_keys = list(fractions[0].keys())
    roi_keys = np.array(roi_keys)
    
    sequence_keys = list(fractions.keys())
    
    n_tweezers = len(roi_keys)
    n_sequences = len(sequence_keys)
    
    modifiers = {r: {s: 1 for s in sequence_keys} for r in roi_keys}
    
    # Time regression
    if doRegressionTime:
        for roi_key in roi_keys:
            for sequence_key in sequence_keys:
                f4 = fractions[sequence_key][roi_key]
                
                f4 = np.diff(f4)
                modifiers[roi_key][sequence_key] *= 2
                
                fractions[sequence_key][roi_key] = f4
            
    # Space regression
    if doRegressionSpace:

        if spaceRegressionType == 'nearest_neighbor':            
            for tweezer_idx, roi_key in enumerate(roi_keys):
                for sequence_key in sequence_keys:
                    f4 = fractions[sequence_key][roi_key]
                    
                    neighbor_roi_key = roi_keys[(tweezer_idx+1)%n_tweezers]
                    f4_neighbor = fractions[sequence_key][neighbor_roi_key]
                        
                    # take spatial difference
                    f4 = f4 - f4_neighbor
                    modifiers[roi_key][sequence_key] *= 2
                    
                    fractions[sequence_key][roi_key] = f4
            
        elif spaceRegressionType == 'linear':
            # Join the arrays across tweezers                    
            fractions_joined = {sequence_key: np.array([f4 for f4 in d.values()]) for sequence_key, d in fractions.items()}
            
            # Prepare container
            regressed_fraction = {}
            
            for sequence_key, f4 in fractions_joined.items():
                # Calculate expected fraction based on linear regression
                prediction = expectedFractionRegression(f4, 'sklearn-lin')
                
                # Take difference between expected and actual fraction
                regressed_fraction[sequence_key] = f4 - prediction
                
            # Split the arrays across tweezers
            fractions = {sequence_key: {roi_key: f4 for roi_key, f4 in zip(roi_keys, arr)} for sequence_key, arr in regressed_fraction.items()}
            
        else:
            raise Exception(f'Invalid space regression type: {spaceRegressionType}')
                
    # Variance
    temp = np.zeros(n_sequences)
    
    for roi_key in roi_keys:
        # Prepare containers
        variance[roi_key] = np.copy(temp)
        variance_error[roi_key] = np.copy(temp)
        
        for sequence_key in sequence_keys:
            f4 = fractions[sequence_key][roi_key]
            modifier = modifiers[roi_key][sequence_key]
            
            if fractional:
                # multiplying by 4 to get variance of f4-f3 from variance of f4.
                #This is because f4-f3=2f4-1, so Var(f4-f3)=Var(2f4)=4Var(f4)
                modifier *= 1/4
                
            var = np.var(f4)/modifier
            
            n_shots = len(f4)
            variance[roi_key][sequence_key] = var
            variance_error[roi_key][sequence_key] = var*np.sqrt(2./(n_shots-1))
     
    return variance, variance_error

def expectedFractionRegression(data, method='cov',**kwargs):
    '''
    Implements Monika's expected variance regression. E.g. for tweezers, for each
    tweezer, it uses data from all other tweezers to estimate the expected variance.
    

    Parameters
    ----------
    data : 2d array
        Shape NxM, where N is the number of variables (e.g. tweezer fractions)\
        and M is the number of observations.

    Returns
    -------

    expected_fraction : array
        Array of length N returning the expected fraction based on the
        data from all other variables.

    '''
    no_variables = data.shape[0]
    prediction = np.zeros(data.shape)
    
    for tw_idx in range(no_variables):
        #slice to exclude the data from the current tweezer
        exclude_slice = np.ones(no_variables,dtype=bool)
        exclude_slice[tw_idx]=False
        
        X=data[exclude_slice]
        Z=data[tw_idx]
        
        ### Linear regression ###
        if method=='cov':
            weights, offset = linRegressionFromCovariance(X,Z)
            
        elif method=='sklearn-lin':
            #sklearn asks for arrays of shape (n_samples, n_features)
            reg = LinearRegression().fit(X.T, Z.T)
            weights, offset = reg.coef_, reg.intercept_
            
        elif method=='sklearn-pls':
            pls = PLSRegression(n_components=kwargs['n_components'])
            pls.fit(X.T, Z.T)
            weights = pls.coef_.squeeze()
            offset = np.mean(Z - weights@X)

        elif method=='sklearn-pca':
            pcr = make_pipeline(StandardScaler(), PCA(n_components=kwargs['n_components']),\
                                LinearRegression())
            pcr.fit(X.T, Z.T)
            pca = pcr.named_steps['pca']
        else:
            raise RuntimeError('Linear regression method unsupported.')
        
        #Prediction E(Z|data)
        if method=='sklearn-pca':
            prediction[tw_idx]=pcr.predict(X.T)
        else:
            prediction[tw_idx] = weights@X+offset

    return prediction

    
def expectedVarianceRegression(data, method='cov',**kwargs):
    '''
    Implements Monika's expected variance regression. E.g. for tweezers, for each
    tweezer, it uses data from all other tweezers to estimate the expected variance.
    

    Parameters
    ----------
    data : 2d array
        Shape NxM, where N is the number of variables (e.g. tweezer fractions)\
        and M is the number of observations.

    Returns
    -------

    expected_variance : array
        Array of length N returning the expected variance based on the
        data from all other variables.

    '''
    no_variables = data.shape[0]
    
    expected_variance = np.zeros(no_variables)
    prediction = expectedFractionRegression(data, method, **kwargs)
    
    for tw_idx in range(no_variables):
        Z=data[tw_idx]

        #first term is just variance of the current tweezer
        expected_variance[tw_idx] = np.var(Z - prediction[tw_idx])

    return prediction, expected_variance


def linRegressionFromCovariance(data, result_vec):
    '''
    Linear regression from Monika's notes.

    Parameters
    ----------
    data : 2d array
        Shape NxM, where N is the number of features (e.g. tweezer fractions)\
        and M is the number of observations.
    result_vec : 1d array
        Shape is (M) where M is the number of observations.

    Returns
    -------
    weights : array
        Linear regression weights (without the offset).
    offset : float
        Offset

    '''
    no_features = data.shape[0]

    #C in Monika's notes
    cov_mat = np.cov(data)
    
    #Covariance of each variable of the data and the result
    #X in Monika's notes
    cov_vec = np.zeros(no_features)
    
    for var_idx in range(no_features):
        cov_vec[var_idx] = np.cov(np.stack((result_vec,data[var_idx]),axis=0))[0][1]
        
    #weights, wihtout the offset
    weights = np.linalg.inv(cov_mat)@cov_vec
    
    #offset
    offset = np.mean(result_vec - weights@data)
    
    return weights, offset

def dataPCA(data,n_components=5,doPlot=False,**kwargs):
    
    n_features=data.shape[1]
    pca = PCA(n_components=n_components,**kwargs)
    pca.fit(data)

    # print(pca.explained_variance_ratio_)
    # print(pca.components_)
    
    if doPlot:
        fig, ax = plt.subplots()

        im = ax.imshow(pca.components_,cmap='coolwarm')
        
        ax.set_yticks(np.arange(0,n_components,2))
        ax.set_xticks(np.arange(0,n_features,2))
        ax.set_xlabel('Feature')
        ax.set_ylabel('PCA component')
        
        fig.colorbar(im)
        plt.show()       

    return pca












def analyzeStats(xdata, ydata, yerr, xlabel, xerr=None, fitType=None, p0=None,
                      normalized=False, scaleFactor=1, logplot=False,
                      xlim=[None,None], ylimLog=[None,None], ylimLin=[None,None],
                      savePlot=False, fitKeys=[],  title=None, fp=None, fileName = 'VarVsFraction.png',
                      showPlot=True, colors=None, fractional = True):
    
    coeffs = dict()
    errs = dict()
    if fitType:
        for key in fitKeys:

            coeffs[key], covar = curve_fit(fitType, xdata[key]*scaleFactor,
                                          ydata[key], p0=p0,sigma=yerr[key],
                                          bounds=(0,np.inf))
            errs[key] = np.sqrt(np.diag(covar))
    
    if showPlot:
        fig, ax = plt.subplots(figsize=(9,6))
        
        for key in xdata.keys():
            if colors:
                color=colors[key]
            else:
                color=None
            
            if xerr is not None:
                xe = xerr[key]
            else:
                xe = None
                
            ax.errorbar(xdata[key]*scaleFactor, ydata[key], yerr[key], xerr=xe, label=key, fmt='o',color=color)
            
            
            
        ax.grid()
        ax.legend(bbox_to_anchor = (1.25,1))
        ax.set_xlabel(xlabel)   
        
        if normalized:
            ax.set_ylabel('var($f_4-f_3$) / projection noise')
            ax.axhline(y=1, color='r', linestyle='--')
            ax.set_ylim(bottom=0)
        elif not fractional:
            plt.ylabel('var($F4-F3$)')
        else:
            plt.ylabel('var($f_4-f_3$)')
            
        if logplot:
            ax.set_yscale('log')
            ax.set_ylim(bottom=ylimLog[0], top=ylimLog[1])
        else:
            ax.set_ylim(bottom=ylimLin[0], top=ylimLin[1])
            
        ax.set_title(title,fontsize=20)
        
        ax.set_xlim(left=xlim[0], right=xlim[1])
        
        if fitType:
            for key in fitKeys:
                
                if colors:
                    color=colors[key]
                else:
                    color=None
                xvec = xdata[key]
                xvals = scaleFactor*np.linspace(np.min(xvec),np.max(xvec),1000)
                ax.plot(xvals, fitType(xvals,*coeffs[key]), label = key, color=color)
        
            # ax.plot(xvals, fitType(xvals, 250, 2.8e-4), color = 'r', linestyle = 'dashed')
                        
        if savePlot:
            plt.savefig(os.path.join(fp, fileName), dpi=300, bbox_inches='tight')
        
    return coeffs, errs, ax


