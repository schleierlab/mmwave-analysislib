# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:25:17 2016

@author: tracy
"""


from matplotlib.pyplot import imshow, ginput,  figure, suptitle, gca, close, colorbar, cm, plot, title
import numpy as np
from matplotlib.transforms import Affine2D


import pickle
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage

from . import andorlyze as andlz
from ..data import autolyze as az
from .roi import cropImageArray, setROI
from ..data import read as rf

from tweezerlyze.detection import DetectionBot


def getMean(fp, device = 'AndorIxon', imgidx = 0):
    filepaths, filenames=rf.open_all_HDF5_in_dir(fp)
        
    nfiles = len(filenames)*1.0 ###Needs to be float!
    nims = 0
    print("number of h5 files: " + str(len(filenames)))
    
    
    with h5py.File(filepaths[0]) as f:
        if device == 'AndorIxon':
            roi = ((0,1024),(0,1024)) #Ixon full        
            #roi = ((320,625),(290,610)) #Ixon abs imag beam
        elif device == 'AVTManta145':
            roi = ((0,1388),(0,1038)) # whole camera
        elif device == 'AVTManta223':
            roi = ((0,2048),(0,1088)) # whole camera
            #roi = ((650,1150),(320,780)) #zoomed in
                        
    meanIms = np.zeros((roi[1][1]-roi[1][0],roi[0][1]-roi[0][0]))
    for idx, item in enumerate(filepaths):
        if not(np.mod(idx,10)):
            print("Completed " + str(idx) + "/" + str(nfiles))
        
        with h5py.File(item) as f:
             if device == 'AndorIxon':
                 numFrames = f['devices']['AndorIxon']['EXPOSURES'].shape[0] 
                 nims += 1
                 name = 'andor'
             elif device == 'AVTManta145':
                numFrames = f['devices']['AVTManta145']['EXPOSURES'].shape[0] 
                nims += 1
                name = 'atoms'
             elif device == 'AVTManta223':
                numFrames = f['devices']['AVTManta223']['EXPOSURES'].shape[0] 
                nims += 1
                name = 'atoms'
             else:
                print('No Andor or Manta pictures in file')
                continue

        if imgidx > numFrames - 1:
            print('Requested image index exceeds number of frames')
            return
        if name == 'andor':
            data = andlz.getImage(item)
        else:
            name = name + str(imgidx)
            data=rf.getdata(item, name)
        imageROI = setROI(data,roi) 
        
        #Do a continuous sum:
        meanIms += imageROI / nfiles
#        print("Running difference between the mean and the image:")
#        print(np.sum(meanIms*nfiles-imageROI))
#        print((np.sum(imageROI)-nfiles*np.sum(meanIms)/(idx+1))/(1024*1024))
#        print(idx)
    print("number of images to average: " + str(nims))
            
    return meanIms

def getSummedData(image, roiBox, backgroundBox, background, remove_bkgnd = False):
    """
    Given an image and ROI boxes, returns a single value averaging over the
    entire ROI box
    """
    
    if remove_bkgnd:
        image = image - background
        
    imROI = setROI(image,roiBox)
    bgROI= setROI(image,backgroundBox)
    bgconst = np.float64(bgROI.sum())/(bgROI.shape[0]*bgROI.shape[1])
        
    summed_data = np.float64(imROI.sum())/(imROI.shape[0]*imROI.shape[1]) - bgconst
    
    return summed_data    


def getSummedDataArray(imageArray, roiBox, backgroundBox=None, nChunks=1,
                       removeBackground = False, backgroundfp = None,
                       doNormalization = False, normBoxes = None,
                       showROI = False):
    """
    Given an array of images and ROI boxes, returns an array of data vectors
    of length that vary depending on the value of nChunks:
        1 - return a single value averaging over the entire ROI box
        n - break into n averaged chunks
        None - average the cloud transversely but not longitudinally
    """
    
    summedData = []
    normData = []
    
    if removeBackground:
        background = np.load(backgroundfp)

    for image in imageArray:
        if removeBackground:
            image = image - background
            
        imROI = setROI(image,roiBox)
        
        if showROI:
            plt.imshow(imROI)
            plt.show()
         
        if backgroundBox:
            bgROI= setROI(image,backgroundBox)
            bgconst = np.float64(bgROI.sum())/(bgROI.shape[0]*bgROI.shape[1])
        else:
            bgconst = 0
        
        if not nChunks:
            imSummed = np.sum(imROI, axis = 0)
            summedData.append(imSummed/imROI.shape[0] - bgconst)
            
        else:
            imChunks = np.array_split(imROI, nChunks, axis = 1)
            shotChunks = []
            for chunk in imChunks:
                shotChunks.append(np.float64(chunk.sum())/(chunk.shape[0]*chunk.shape[1]) - bgconst)
            
            if doNormalization:
                normSum = 0
                for normROI in normBoxes:
                    chunk = setROI(image, normROI)
                    normSum = normSum + np.float64(chunk.sum())/(chunk.shape[0]*chunk.shape[1]) - bgconst
                
                shotChunks = shotChunks/normSum
                normData.append(normSum)
            
            summedData.append(shotChunks)
        
    return summedData, normData

def getParamArray(shots, param_idx):
    parameters = az.getParameterChanges(shots[0])
    parameters = list(parameters.keys())
    
    if not parameters:
        parameters = ['C_repeats']
    
    returnParameter=parameters[param_idx]
    if returnParameter == 'C_repeats':
        paramArray = np.arange(len(shots))
    else:
        paramArray = az.getParamArray(shots, returnParameter)
        paramArray = np.array(paramArray)
        
    return returnParameter, paramArray

def extractROIDataSingleSequence(fp,roiDict,removeBackground=False, backgroundfp='', doNormalization=False, norm_dict=None, param_idx=0):
    '''
    A function to extract roi data from a single sequence of shots. 
    roiDict is a dictionary of rois in the "style" of an(alyzeTweezers.py
    (3 or 4 manifold on the top level, and then specific tweezers on the bottom level).

    Parameters
    ----------
    fp : string
        Full file path to the shot folder.
    roiDict : dict
        DESCRIPTION.
    removeBackground : bool, optional
         The default is False.
    backgroundfp : string, optional
        Filepath to the background image. The default is ''.
    doNormalization : bool, optional
         The default is False.
    norm_dict : bool, optional
         The default is None.

    Returns
    -------
    data : dict
        Dictionary of summed roi data.
    parameters : list
        List of parameter names.
    paramArray : array
        Array of parameter values.

    '''
    shots = az.getShots(fp)
    shots = az.getCompleteShots(shots)
            
    imageArray = andlz.getImageArray(shots)

    data = {}

    for manifold, rois in roiDict.items(): 
        data[manifold]={}
        for idx, roi in rois.items():
            norm_list=None
            if doNormalization:
                norm_list = norm_dict[idx]
            
            roi_data, _ = getSummedDataArray(imageArray, 
                                               roi,
                                               removeBackground=removeBackground,
                                               backgroundfp=backgroundfp,
                                               doNormalization=doNormalization,
                                               normBoxes = norm_list)
    
            data[manifold][idx] = np.array(roi_data)
            
    return data

def getRoiArray(imageArray, roi, removeBackground=False, backgroundfp=None,
                roundBackground=False):
    
    dataArray = []
    
    if type(imageArray)==list:
        imageArray = np.array(imageArray)
    
    if removeBackground:
        background = np.load(backgroundfp)
        
        if roundBackground:
            background = np.round(background).astype('int16')
        
        imageArray = imageArray - background
        
    dataArray = cropImageArray(imageArray, roi)
    
    return dataArray


### Blob detection ###########################################################

def showBlobsImage(image,reference_blobs=None,**blob_kwargs):
    bot = DetectionBot()
    
    blob_locations = bot.set_blobs(image.T,**blob_kwargs)
    
    if reference_blobs is not None:
        true_mask, _ = compareBlobPositions(blob_locations,reference_blobs)
    else:
        true_mask=None
    
    fig, ax = bot.show_blobs(circles=True,cmap = 'viridis',text=True,true_mask=true_mask)

    return fig, ax

def extractBlobLocations(image,showBlobs=False,reference_blobs=None,**blob_kwargs):
    bot = DetectionBot()
    
    blob_locations = bot.set_blobs(image.T,**blob_kwargs)
    
    if reference_blobs is not None:
        true_mask, _ = compareBlobPositions(blob_locations,reference_blobs)
    else:
        true_mask= None
    
    if showBlobs:
        bot.show_blobs(circles=True,cmap = 'viridis',text=True,true_mask=true_mask)
            
    return blob_locations

def compareBlobPositions(blobs,blobs_reference,radius=3):
    blob_truth = np.zeros(len(blobs),dtype=bool)
    blob_ref_index = np.array([None for i in range(len(blobs))])
    
    for bidx, blob in enumerate(blobs):
        dist = np.linalg.norm(blobs_reference-blob,axis=1)
        blob_within_radius = dist<radius
        
        blob_truth[bidx] = np.any(blob_within_radius) 
        if np.sum(blob_within_radius)>1:
            raise Exception('Multiple blobs found close to the reference blobs.')
        
        if blob_truth[bidx]:
            blob_ref_index[bidx]=np.nonzero(blob_within_radius)[0][0]
        
    return blob_truth, blob_ref_index

def extractReferenceBlobs(image,roi_pad=0,**blob_kwargs):
    bot = DetectionBot()
        
    blob_locations = bot.set_blobs(image.T,**blob_kwargs)
    blob_rois=[]
    
    if 'min_sigma' in blob_kwargs:
        radius = blob_kwargs['min_sigma']
    elif 'max_sigma' in blob_kwargs:
        radius = blob_kwargs['max_sigma']
    else:
        radius=1

    if type(roi_pad)==tuple:
        pad_x = roi_pad[0]
        pad_y = roi_pad[1]
    else:
        pad_x = roi_pad
        pad_y = roi_pad
        
    r_x=radius+pad_x
    r_y=radius+pad_y

    for c in blob_locations.astype(int):
        blob_rois.append(((c[0]-r_x, c[0]+r_x), (c[1]-r_y, c[1]+r_y)))   
        
    return blob_locations, blob_rois

def extractDetectedBlobCounts(imArray,reference_blobs,showBlobs, **blob_kwargs):
    '''
    Get detected blobs in an array of images by using positions given in reference blobs.
    Returns the count of each reference blob, and a boolean array for each image
    denoting which blobs were detected.
    '''
  
    no_blobs=len(reference_blobs)
    #count blob array
    blob_count = np.zeros(no_blobs)    
        
    #blob found array
    blob_found = np.zeros((imArray.shape[0],no_blobs),dtype=bool)
    
    false_pos_locs = np.zeros((0,2))
    #Find blobs in each image
    for imidx, image in enumerate(imArray):
        blob_locations = extractBlobLocations(image,reference_blobs=reference_blobs\
                                              ,showBlobs=showBlobs,**blob_kwargs)
    
        blob_truth, blob_idx = compareBlobPositions(blob_locations,reference_blobs)
        
        blob_count[blob_idx[blob_truth].astype(int)]+=1
        blob_found[imidx][blob_idx[blob_truth].astype(int)]=True
        
        false_pos_locs = np.concatenate((false_pos_locs,\
                                         blob_locations[np.logical_not(blob_truth)]),axis=0)
       
    return blob_count, blob_found, false_pos_locs

def getTweezerEdgePairs(x_freqs,y_freqs):
    '''
    Parameters
    ----------
    x_freqs : np.array
        Tweezer x frequencies
    y_freqs : np.array
        Tweezer y frequencies

    Returns
    -------
    pairs : list
        A list of tuples of start and end points in frequency space.

    '''
    pairs = []
    
    if min(x_freqs)!=max(x_freqs):
        for y in y_freqs:
            pairs.append(((min(x_freqs),y),(max(x_freqs),y)))
    
    if min(y_freqs)!=max(y_freqs):
        for x in x_freqs:
            pairs.append(((x,min(y_freqs)),(x,max(y_freqs))))
            
    return pairs


def transformPairs(pairs,transform):
    '''
    Apply transform to pairs of coordinates
    
    Parameters
    ----------
    pairs : list of tuples
        Each pair of points ((x0,y0),(x1,y1))
    transform : matplotlib.transforms.Transform
    Returns
    -------
    pairs_trans : list of tuples
    '''
    pairs_trans = []
    for pair in pairs:
        pairs_trans.append((tuple(transform.transform(pair[0])),\
                            tuple(transform.transform(pair[1]))))
                           
    return pairs_trans

def getTweezerTransform(tweezerRef,pixRef,scale=(1,1),angles=(0,0)):
    '''
    Get a transformation from tweezer frequency space to image pixel space.
    
    Parameters
    ----------
    tweezerRef : tuple
        (x_freq,y_freq) coordinates of the reference point
    pixRef : tuple
        (x_pix,y_pix) coordinates of the reference point
    scale : tuple, optional
        DESCRIPTION. scale for (x,y) in px/MHz. The default is (1,1).
    angles : tuple, optional
        DESCRIPTION. Shear angles in x and y in degrees. The default is (0,0).

    Returns
    -------
    transformation: matplotlib.transforms.Transform
    '''
    
    #scale to apply after the shear transform to preserve distances along the axes
    rotated_scale = (np.cos(np.deg2rad(angles[0])),np.cos(np.deg2rad(angles[1])))

    trans=Affine2D().translate(*(-1*np.array(tweezerRef))).skew_deg(angles[1],angles[0]).scale(*rotated_scale)
    
    #transformation shifted to tweezerRef, transformed to frequency scale and then
    #applied the skew
    #transformation to the pixel space, with the start at pixRef
    trans=trans.scale(*scale).translate(*pixRef)
    
    #return a transformation from the frequency to the pixel space
    return trans
    

def getCOMdataArray(imageArray,removeBackground = False, backgroundfp = None):
    moments = []

    if removeBackground:
        background = np.load(backgroundfp)
    
    for image in imageArray:
        if removeBackground:
            image = image - background

        totalCounts = np.sum(image)*1.

        centerOfMass = ndimage.measurements.center_of_mass(image) # centerOfMass[0] is Y and centerOfMass[1] is X
        
        [mg1, mg2] = np.meshgrid(np.arange(int(imageArray.shape[0])), np.arange(int(imageArray.shape[1]))) #mg1 corresponds to X, mg2 corresponds to Y 
        mg1 = (mg1-centerOfMass[1])**2
        mg2 = (mg2-centerOfMass[0])**2
                
        stdDevX = np.sqrt(np.sum(image*mg1)/totalCounts)
        stdDevY = np.sqrt(np.sum(image*mg2)/totalCounts)

        moments.append([centerOfMass[1],centerOfMass[0],stdDevX,stdDevY])
        
    return np.array(moments)


# this function hasn't been verified
def subtractBackground(imageArray,backgroundfp):
    background = np.load(backgroundfp)
    for index, image in enumerate(imageArray):
        image = image - background
        imageArray[index] = image
    
    return imageArray


def getgMHz(Natoms, Delta, shift):
    return np.sqrt(shift*Delta/Natoms)/10**6
   

def choose_roi(data, **kwargs):
    figure()
    fig=gca()
    #imshow(data, interpolation="none", cmap = cm.gray, vmin=0, vmax=2*data.mean())
    imshow(data, interpolation="none", **kwargs)
    colorbar()
    pts=ginput(n=2, timeout=30, show_clicks=True)     
    xpts=[pts[0][0], pts[1][0]]
    ypts=[pts[0][1], pts[1][1]]
    xpts.sort()
    ypts.sort()
    close(fig.figure)
    return np.rint(xpts), np.rint(ypts)
    
def choose_trace_roi(x,y): #x is array of x values, y is array of y values
    figure()
    plot(x,y, 'o-')
    title("Choose ROI")
    pts=ginput(n=2, timeout=30, show_clicks=True)
    x1=pts[0][0]
    x2=pts[1][0]
    
    index1=min(range(len(x)), key=lambda i: abs(x[i]-x1)) #finds index of x value closest to the chosen ROI x value
    index2=min(range(len(x)), key=lambda i: abs(x[i]-x2))

    return index1, index2 
    
    
def useroi(dataarray, xpts, ypts):
    cutarray=[]
    for item in dataarray:
        cutpic=item[ypts[0]:ypts[1], xpts[0]:xpts[1]]
        cutarray.append(cutpic)
    return cutarray
    
def subim(arrayim, imfp):
    """subtracts the dark image located at filepath imfp from all images in arrayim"""
   
    bkgdim=rf.read_sis_file(imfp, log=1)
    subarray=[]
    c=1;
    print("subtracting")
    for item in arrayim:
        print("file no. ", c)
        im=np.double(item)-np.double(bkgdim)
        subarray.append(im)
        c=c+1;
    return subarray
    
def bgdsubtract(arrayim, setroi=1, fp=""):
    """subtracts out the background from logged ims"""
    
    if setroi==0:
       bgroi_file=open(fp,'rb')
       xptsbg=pickle.load(bgroi_file)
       yptsbg=pickle.load(bgroi_file)
       backgrounds=[]
       for item in arrayim:
           bgpic=item[yptsbg[0]:yptsbg[1], xptsbg[0]:xptsbg[1]]
           backgrounds.append(bgpic)
    else:
        backgrounds, xpts, ypts=cutimages(arrayim)
        if len(fp)!=0:
            roi_file=open(fp,'wb')
            pickle.dump(xpts, roi_file)
            pickle.dump(ypts, roi_file)
            roi_file.close()
    
    means=[]
    for item in backgrounds:
        avg=item.mean()
        print(avg)
        means.append(avg)
    
    del backgrounds
    loggedims=[]
#    
#    for index in range(len(arrayim)):
#        im=np.double(arrayim[index])-np.double(means[index])
#        loggedims.append(im)
#        print "subtracting background on im. ", index
    
    while len(arrayim)>0:
        im=np.double(arrayim[0])-np.double(means[0])
        loggedims.append(im)
       # print "images to go: ", len(arrayim)
        del arrayim[0]
        del means[0]
        

    del means    
    return loggedims
    


def imgcheck(imarray, xarray, cutoff):
    x_array=[]
    goodimgs=[]
    for index in range(len(imarray)):    
        picavg=imarray[index].mean()       
        if picavg>cutoff:    
            goodimgs.append(imarray[index])
            x_array.append(xarray[index])    
    return goodimgs, x_array


def cutimages_index(imarray, index=5, **kwargs):
    """
    Takes in an array of numpy images and applies the same ROI to all. Outputs cut images in another array
    """
    cutarray=[]
    
#    logpic=-np.log(np.abs(np.double(imarray[len(imarray)/2]))+0.00001) + np.log(2**16)
#    scaledlogpic=logpic*10000
#    rows, cols=np.where(scaledlogpic>50000)
#    scaledlogpic=scaledlogpic.astype(np.uint16)
#    slicedscaledlogpic=scaledlogpic<<4
    
   # xpts, ypts=choose_roi(imarray[len(imarray)/2], **kwargs)#choose_roi(imarray[-5])
    xpts, ypts=choose_roi(imarray[index], **kwargs)
    for item in imarray:
        cutim=item[ypts[0]:ypts[1], xpts[0]:xpts[1]]
        cutarray.append(cutim)
    return cutarray, xpts, ypts

    
def cutimages(imarray, **kwargs):
    """
    Takes in an array of numpy images and applies the same ROI to all. Outputs cut images in another array
    """
    cutarray=[]
    
#    logpic=-np.log(np.abs(np.double(imarray[len(imarray)/2]))+0.00001) + np.log(2**16)
#    scaledlogpic=logpic*10000
#    rows, cols=np.where(scaledlogpic>50000)
#    scaledlogpic=scaledlogpic.astype(np.uint16)
#    slicedscaledlogpic=scaledlogpic<<4
    
   # xpts, ypts=choose_roi(imarray[len(imarray)/2], **kwargs)#choose_roi(imarray[-5])
    xpts, ypts=choose_roi(imarray[5], **kwargs)
    for item in imarray:
        cutim=item[ypts[0]:ypts[1], xpts[0]:xpts[1]]
        cutarray.append(cutim)
    return cutarray, xpts, ypts

#def showlog(imarray, slicenum):
#    logimarray=[]
#    counter=1
#    for item in imarray:
#        logpic=-np.log(np.double(item)/2**16) 
#        scaledlogpic=logpic*10000
#        rows, cols=np.where(scaledlogpic>50000)
#        scaledlogpic[rows, cols]=1  
#        scaledlogpic=scaledlogpic.astype(np.uint16)
#        image=scaledlogpic<<slicenum
#        
#        logimarray.append(image)
#        print "log file no. ", counter
#        counter=counter+1
#    return logimarray

def showlog(imarray, slicenum):
    logimarray=[]
    counter=1
    for item in imarray:
        logpic=-np.log(item) 
        scaledlogpic=logpic*10000
        rows, cols=np.where(scaledlogpic>50000)
        scaledlogpic[rows, cols]=1  
        scaledlogpic=scaledlogpic.astype(np.uint16)
        image=scaledlogpic<<slicenum
        
        logimarray.append(image)
        print("log file no. ", counter)
        counter=counter+1
    return logimarray
    

def div(patharray): #makes divided image for absorption imaging analysis 
    absorption_images=[]
    for path in patharray:
        atoms=rf.getdata(path, 'atoms')
        flat=rf.getdata(path, 'flat')
        dark=rf.getdata(path, 'dark')
        #change data type to int16 so we can handle negative values in the subtraction
        numer=(atoms-dark).astype('int16') 
        denom=(flat-dark).astype('int16')
        #find the indices where the flat-dark is zero and change these elements to 1. This avoids nan values that result from division by zero
        rows, cols=np.where(denom==0)
        denom[rows, cols]=1
        #the absorption image is (atoms-dark)/(flat-dark). The posify function takes values of the array less than or equal to 0 and sets them to 1. This avoids -inf errors when taking the log later
        absim=rf.posify(np.double(numer)/denom)
#        rows, cols=np.where(absim>1)
#        absim[rows, cols]=1
        absorption_images.append(absim)
    return absorption_images

def takelog(absorption_images): #takes the -log of each image in an array of images for absorption imaging
    logims=[]
    for item in absorption_images:
        logim=-np.log(item)
        logims.append(logim)
    return logims
    
        
    
def posify(array):
        rows, cols=np.where(array<=0)
        array[rows, cols]=1
        return array      
    
def posify2(array):
        rows, cols=np.where(array<=0)
        array[rows, cols]=0
        return array

def div(patharray,dark): #makes divided image for absorption imaging analysis
    absorption_images=[]
    for path in patharray:
        atoms=rf.getdata(path, 'images/atoms0')
        flat=rf.getdata(path, 'images/atoms1')
        #change data type to int16 so we can handle negative values in the subtraction
        numer=(atoms-dark).astype('int16') 
        denom=(flat-dark).astype('int16')
        #find the indices where the flat-dark is zero and change these elements to 1. This avoids nan values that result from division by zero
        rows, cols=np.where(denom==0)
        denom[rows, cols]=1
        #the absorption image is (atoms-dark)/(flat-dark). The posify function takes values of the array less than or equal to 0 and sets them to 1. This avoids -inf errors when taking the log later
        absim=posify(np.double(numer)/denom)
        absorption_images.append(absim)
    return absorption_images

def normalizeAbsorptionImages(img1, img2):
    #when we get a stronger signal we need to average over a ROI that doesn't have 
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    img1 = img1*(mean2/mean1)
    
    return img1

def div_ROI(shots, dark, roi=None, h5folder='images/', normalizeImages=False): 
    #makes divided image for absorption imaging analysis  - taken from cavity lab's file imageFunctions.py
    absorption_images=[]
    numerators = []
    denominators = []
    
    if roi:
        dark = setROI(dark, roi)
        
    for shot in shots:
        atoms = rf.getdata(shot, h5folder + 'atoms0')
        flat = rf.getdata(shot, h5folder + 'atoms1')
        
        #change data type to int16 so we can handle negative values in the subtraction
        if roi:
            atoms = setROI(atoms, roi)
            flat = setROI(flat, roi)
        
        if normalizeImages:
            atoms = normalizeAbsorptionImages(atoms, flat)
        
        numer = (atoms-dark).astype('int16') 
        denom = (flat-dark).astype('int16')
        
        #find the indices where the flat-dark is zero and change these elements to 1. This avoids nan values that result from division by zero
        rows, cols = np.where(denom == 0)
        denom[rows, cols] = 1
        
        rows2, cols2 = np.where(numer == 0)
        numer[rows2, cols2] = 1
        
        # The absorption image is (atoms-dark)/(flat-dark). The posify function 
        # takes values of the array less than or equal to 0 and sets them to 1.
        # This avoids -inf errors when taking the log later

        absim = posify(np.double(numer)/np.double(denom))


        absorption_images.append(absim)
        numerators.append(numer)
        denominators.append(denom)
        
    return absorption_images, numerators, denominators


def div_ROI2(images,dark,roi = None,h5folder='images/', normalizeImages = False): 
    #makes divided image for absorption imaging analysis  - taken from cavity lab's file imageFunctions.py
    """
    same as div_ROI but for a single shot
    """
    if roi:
        dark = setROI(dark,roi)
        
    atoms=images[0]
    flat=images[1]

    if roi:
        atoms = setROI(atoms,roi)
        flat = setROI(flat,roi)
    
    if normalizeImages:
        atoms = normalizeAbsorptionImages(atoms,flat)
    
    numer=(atoms-dark).astype('int16') 
    denom=(flat-dark).astype('int16')

    rows, cols=np.where(denom==0)
    denom[rows, cols]=1
    rows2, cols2=np.where(numer==0)
    numer[rows2, cols2]=1

    absim=posify(np.double(numer)/np.double(denom))

    return absim, numer, denom


def div_meanImage(meanAtoms, meanFlat): #makes divided image for absorption imaging analysis when using mean images
    #change data type to int16 so we can handle negative values in the subtraction
    numer = meanAtoms.astype('int16') 
    denom = meanFlat.astype('int16')
    #find the indices where the flat-dark is zero and change these elements to 1. This avoids nan values that result from division by zero
    rows, cols=np.where(denom==0)
    denom[rows, cols]=1
    rows2, cols2=np.where(numer==0)
    numer[rows2, cols2]=1
    #The posify function takes values of the array less than or equal to 0 and sets them to 1. This avoids -inf errors when taking the log later
    absim=posify(np.double(numer)/np.double(denom))
    return absim