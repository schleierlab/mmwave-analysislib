# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:30:26 2020

@author: Jacob
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import ndimage
from uncertainties import ufloat
from uncertainties import unumpy as unp

from . import functions as ff


def applyFit(xvalues, yvalues, fitType, fitIC, bounds=(-np.inf, np.inf)):
    """
    Function to apply fit fitType to columns of yvalues. Independent variable 
    is given in an array xvalues (either a 1D array or an array the same shape
    as yvalues).
    """
    
    if len(yvalues.shape) == 1:
        yvalues = yvalues.reshape((yvalues.shape[0],1))
        ax1Length = 1
    else:
        ax1Length = yvalues.shape[1]
        
    if len(xvalues.shape) == 1:
        xvalues = np.transpose(np.tile(xvalues,(ax1Length,1)))

    coeffsAll = np.zeros((ax1Length, np.size(fitIC)))
    errAll = np.zeros((ax1Length, np.size(fitIC)))
      
    for idx in range(ax1Length):
        coeffs, covar = curve_fit(fitType, xvalues[:,idx],
                                   yvalues[:,idx], p0=fitIC,
                                   bounds=bounds)
        coeffsAll[idx] = coeffs
        errAll[idx] = np.sqrt(np.diag(covar))
    
    return coeffsAll, errAll


def generic_fit(model, xdata, ydata, guesses, hold=False,numpoints=100,meth='lm'):
    if hold:
        coefs = guesses
        covar = 0*guesses
    else:
        coefs, covar = curve_fit(model, xdata, ydata, guesses, method=meth)
        print("Fit Parameters\n")
        print(coefs) 
        print("Fit Errors\n")
        print(np.sqrt(np.diag(covar)))
    #print([[coefs[nn],np.sqrt(covar[nn,nn])] for nn in range(len(coefs))])
    x_fit = np.linspace(np.amin(xdata),np.amax(xdata),numpoints)
    y_fit = model(x_fit,*coefs)
    return coefs, np.sqrt(np.diag(covar)), x_fit, y_fit

def gaussianFitSeedGuess(x,y):
    '''
    

    Parameters
    ----------
    x : 1D array of length n.
    y :  1D array of length n.
    Returns
    -------
    x0 : float
        Guess for the gaussian center.
    A : float
        Amplitude guess.
    sigma : float
        Standard deviation guess.
    offset : float
        Offset guess.

    '''
    offset = np.min(y)
    yShifted = y-offset
    x0 = np.sum(yShifted*x)/np.sum(yShifted)
    A = np.max(y)-np.min(y)
    sigma = np.sqrt(np.sum(np.abs(yShifted)*(x-x0)**2)/np.sum(np.abs(yShifted)))
    
    return x0, A, sigma, offset

def getTransverseGaussianFit(imageArray, axisToSum=1):
    '''
    Get a transverse gaussian fit in an image array.

    Parameters
    ----------
    imageArray : ndarray
        Array of images.
    axisToSum : int, optional
        Axis of the image to sum over. The default is 1.

    Returns
    -------
    fitCoeffs :
        Fit coefficients: center, amplitude, waist, offset
    fitErrors :

    '''
    if len(imageArray.shape)==2:
        imageArray = imageArray[np.newaxis,...]
    
    fitCoeffs = np.zeros((imageArray.shape[0],4))
    fitErrors = np.zeros((imageArray.shape[0],4))
    
    for imidx, image in enumerate(imageArray):
        crossSectionSum = np.mean(image,axis=axisToSum)
        xValues = np.arange(np.size(crossSectionSum))

        fitInitialConditions =  list(gaussianFitSeedGuess(xValues,crossSectionSum)) #(x0, A, sigma,B)
        coeffs, covar = curve_fit(ff.gaussianOffset, xValues,crossSectionSum, p0=fitInitialConditions)   
        fitErrors[imidx] = np.sqrt(np.diag(covar))
        fitCoeffs[imidx]=coeffs
        
    return fitCoeffs, fitErrors

def temperatureFit(waists, times, waistErrors=None):
    '''
    

    Parameters
    ----------
    waists : numpy array
        Waists corresponding to times.
    times : numpy array
        Array of time values.
    waistErrors : array, optional
        Waist fit errors. The default is None.

    Returns
    -------
    temperature : ufloat
        Temperature in K.
    waist : ufloat
        Initial cloud waist.
    '''

    initialGuess = [1e-6, np.min(waists)]
    pfit, pcov = curve_fit(ff.cloudRadiusCesium, times,waists, sigma=waistErrors, p0=initialGuess)
    sig = np.sqrt(np.diag(pcov))
    temperature = ufloat(pfit[0],sig[0])
    waist = ufloat(pfit[1],sig[1])
    return temperature, waist


def imfit1DGaussian(imarray, sumAxis=0):
    coeffsArray = []
    for im in imarray:
        crossSectionSum = np.sum(im,axis=sumAxis)/np.shape(im)[sumAxis]
        xValues = np.arange(np.size(crossSectionSum))
        x0_guess = np.argmax(crossSectionSum)
        A_guess = np.max(crossSectionSum)
            
        fitInitialConditions = [x0_guess, A_guess, 20, 0] #(x0, A, sigma,B)            
        coeffs, covar = curve_fit(ff.gaussianOffset, xValues,
                                      crossSectionSum, p0=fitInitialConditions)
        coeffsArray.append(coeffs)
    
    return np.array(coeffsArray)


def fit_1Dgaussian(image, ic, view_fits=True):
    
    data = image.transpose().ravel()
    
    sizeX, sizeY = image.shape
    x = np.linspace(0, sizeX-1, sizeX)
    y = np.linspace(0, sizeY-1, sizeY)
    x, y = np.meshgrid(x, y)
             
    popt, pcov = curve_fit(ff.oneD_Gaussian_withRotation, (x, y), data, p0=ic)
    angle = popt[6]%np.pi
    data_fitted = ff.twoD_Gaussian_withRotation((x, y), *popt)
    
    errors = np.sqrt(np.diag(pcov))
    param_dict={'offset':popt[5], 'amp':popt[0],'x0':popt[1], 'xsig':abs(popt[3]), 'y0':popt[2], 'ysig':abs(popt[4]), 'rota':angle}
    error_dict={'offset':errors[5], 'amp':errors[0],'x0':errors[1], 'xsig':errors[3], 'y0':errors[2], 'ysig':errors[4], 'rota':angle}
    
    ## to view the fits
    if view_fits:
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data.reshape(sizeY, sizeX), cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(sizeY, sizeX), 6, colors='w')
        plt.show()
        plt.clf()
        print(param_dict)

    return param_dict, error_dict


def fit_2Dgaussian(image, rotatedFit=True): 
    
    data = image.transpose().ravel()

    sizeX, sizeY = image.shape
    x = np.linspace(0, sizeX-1, sizeX)
    y = np.linspace(0, sizeY-1, sizeY)
    x, y = np.meshgrid(x, y)

    maxImage = ndimage.measurements.maximum_position(image)
    guessX0 = maxImage[0]
    guessY0 = maxImage[1]
            
    if rotatedFit:    
        initial_guess = (100,guessX0,guessY0,100,100,10,0)
        popt, pcov = curve_fit(ff.twoD_Gaussian_withRotation, (x, y), data, p0=initial_guess)
        angle = popt[6]%np.pi
        data_fitted = ff.twoD_Gaussian_withRotation((x, y), *popt)
    else:
        initial_guess = (100,guessX0,guessY0,100,100,10)
        popt, pcov = curve_fit(ff.twoD_Gaussian_noRotation, (x, y), data, p0=initial_guess)
        angle = 0 #I need to find a way to send this angle into the twoD_Gaussian_noRotation function!!!
        data_fitted = ff.twoD_Gaussian_noRotation((x, y), *popt)
    
    param_dict={'offset':popt[5], 'amp':popt[0],'x0':popt[1], 'xsig':abs(popt[3]), 'y0':popt[2], 'ysig':abs(popt[4]), 'rota':angle}
    
    ## to view the fits
    fig, ax = plt.subplots(1, 1)
    # ax.hold(True)
    ax.imshow(data.reshape(sizeY, sizeX), cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(sizeY, sizeX), 6, colors='w')
    
    return param_dict