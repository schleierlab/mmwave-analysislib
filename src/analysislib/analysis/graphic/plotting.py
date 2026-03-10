# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:50:35 2020

@author: Jacob
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit, minimize
from analysis.fit.functions import line_pi2, twoD_Gaussian_noRotation, twoD_Quadratic

import re


def scatterWithErrors(data1, data2, errs, colors, fig = None, ax = None, param_dict = dict(),
                      figsize=(9,6)):
    '''
    Function to unify data scatter and errorbars. If fig and ax are not provided
    new ones are created and returned. If there are additional parameters to be
    passed to ax.scatter, you can use param_dict. Ie. custom label can be passed
    as param_dict = {'label': 'Custom label'}.
    '''

    if not (fig and ax):
        fig, ax =plt.subplots(figsize=figsize)

    ax.scatter(data1,data2, color = colors, **param_dict)
    ax.errorbar(data1,data2, errs,\
             ecolor = colors, fmt='none')

    return fig, ax

def scatterWithErrorsMarkers(data1, data2, errs, colors, markers='o', fig = None, ax = None, param_dict = dict(),
                      figsize=(9,6)):
    '''
    Function to unify data scatter and errorbars. If fig and ax are not provided
    new ones are created and returned. If there are additional parameters to be
    passed to ax.scatter, you can use param_dict. Ie. custom label can be passed
    as param_dict = {'label': 'Custom label'}.
    '''

    if not (fig and ax):
        fig, ax =plt.subplots(figsize=figsize)

    ax.scatter(data1,data2, s=50, color = colors, marker=markers, **param_dict)
    ax.errorbar(data1,data2, errs,\
             ecolor = colors, fmt='none')

    return fig, ax

def colorsForUniqueLabels(labels, labeltxt, cmap = plt.cm.rainbow):
    '''
    A function which returns a set of colors and indices array for those color
    for an array of labels.
    '''
    uniqueLabels = np.unique(labels)

    colors = cmap(np.linspace(0.1,0.9,len(uniqueLabels)))
    colorsAllLabels = np.array([])
    for label in labels:
        colorsAllLabels = np.append(colorsAllLabels,np.argwhere(uniqueLabels==label))
    colorsAllLabels = colorsAllLabels.astype(int)

    return colors, colorsAllLabels


def tightColorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, **kwargs)
    return cbar

def drawGrid(ax,pairs,**kwargs):
    '''
    Draw a grid on ax based on pairs of points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    pairs : list of tuples
        Each pair of points ((x0,y0),(x1,y1))
    **kwargs :
        Additional ax.plot arguments

    Returns
    -------
    ax : matplotlib.axes.Axes

    '''
    for pair in pairs:
        ((x0,y0),(x1,y1))=pair
        ax.plot((x0,x1),(y0,y1),**kwargs)

    return ax


def adjust_lightness(color, amount=.8):
    '''
    Adjust the lightness of color.
    Taken from tack overflow: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    '''
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_saturation(color, amount=.8):
    '''
    Adjust the saturation of color.
    Taken from tack overflow: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    '''
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1],  max(0, min(1, amount * c[2])))


def setAxParams(ax,axpar):
    '''
    Set matplotlib axes parameters based on a dictionary of values axpar.
    '''

    if 'yscale' in axpar:
        kwargs = {'value': axpar['yscale']}

        if axpar['yscale'] == 'log':
            if 'basey' in axpar:
                kwargs['base'] = axpar['basey']
            else:
                kwargs['base'] = 10

        ax.set_yscale(**kwargs)

    if 'xscale' in axpar:
        kwargs = {'value': axpar['xscale']}

        if axpar['xscale'] == 'log':
            if 'basex' in axpar:
                kwargs['base'] = axpar['basex']
            else:
                kwargs['base'] = 10

        ax.set_xscale(**kwargs)

    if 'fsize' in axpar:
        fsize = axpar['fsize']
    else:
        fsize=None

    ax.tick_params(axis='x', labelsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)

    if 'title' in axpar:
            ax.set_title(axpar['title'],fontsize=fsize)
    if 'xlabel' in axpar:
            ax.set_xlabel(axpar['xlabel'],fontsize=fsize)
    if 'ylabel' in axpar:
            ax.set_ylabel(axpar['ylabel'],fontsize=fsize)
    if 'xlim' in axpar:
            ax.set_xlim(axpar['xlim'])
    if 'ylim' in axpar:
            ax.set_ylim(axpar['ylim'])
    if 'aspect' in axpar:
        ax.set_aspect(axpar['aspect'])

    return ax


def transparent_edge_plot(ax, x, y, yerr = None , xerr = None, marker = 'o', ms = 12, **kwargs):
    if yerr is not None and xerr is not None:
        base,_,_ = ax.errorbar(x, y, yerr, xerr, ms = ms, ls ="none", alpha=0.6, markeredgewidth = 2, **kwargs) # marker = marker,
    elif yerr is not None and xerr is None:
        base,_,_ = ax.errorbar(x, y, yerr, ms = ms, ls ="none", alpha=0.6, markeredgewidth = 2, **kwargs) # marker = marker,
    else:
        base, = ax.plot(x, y, ms = ms, ls ="none", alpha =0.5, markeredgewidth = 2, **kwargs) # marker = marker,
    ax.plot(x, y, ms =ms, marker =marker, linestyle ="None", markeredgecolor = base.get_color(), markerfacecolor ="None", markeredgewidth =2)
    return ax

def lighterFillScatter(ax, x, y, yerr = None , marker = 'o', ms = 12, **kwargs):
    '''
    Make a data scatter where color defines the marker edge color, while
    the fill color is lighter.
    '''
    if 'color' in kwargs:
        color = kwargs['color']
        del kwargs['color']
    elif 'c' in kwargs:
        color = kwargs['c']
        del kwargs['c']
    else:
        color = 'C0'

    color_lighter = adjust_lightness(color,1.2)

    if yerr is not None:
        base,_,_ = ax.errorbar(x, y, yerr, ms = ms, marker = marker, linestyle ="None", markeredgewidth = 2, color=color_lighter,markeredgecolor = color,barsabove=True,ecolor=color,**kwargs)
    else:
        base, = ax.plot(x, y, ms = ms, marker = marker, linestyle ="None", markeredgewidth = 2, color=color_lighter,markeredgecolor = color,**kwargs)

    return ax

def getTwIdxFromLabel(twLabel):
    match = re.match(r"([a-z]+)(_*)([0-9]+)", twLabel, re.I)
    if match:
        items = match.groups()
        twIdx=int(items[-1])
    else:
        raise ValueError('Invalid tweezer label.')

    return twIdx

def twIdxListFromLabelList(twLabelList):
    twIdxList = [0]*len(twLabelList)
    for labelIdx, twLabel in enumerate(twLabelList):
        twIdxList[labelIdx]=getTwIdxFromLabel(twLabel)

    return twIdxList

def generateTweezerSubplots(xData, yData, yErrs,
                            regionToPlot,
                            colorSet,
                            colorIndices, markerSet, xlabel, ylabel, yType,
                            legendlabels, labelFormat, axisLimit = None,
                            linestyle='None', markersize=8, pslice=slice(None),yscale='linear'):

    '''
    xData is typically varyParams
    yData is whatever you want to plot
    yErrs is the error on yData
    regionToPlot is typically regionIdx
    colorSet is the list of colors you want to use, and colorIndices are the indices you want to index to. Both produced by colorsForUniqueLabels.
    markerSet is a list of markers for each plot
    xlabel and ylabel are just the labels for the axes
    yType is in the set {'phase', 'norm', and 'totalCounts'}, and is used to set axis limits if none are given.
    legendlabels are the unique labels for each run
    labelformat is labeltxt
    axisLimit is a 1x2 list of the min and max ylim, default is None
    linestyle is default None
    markersize is default None
    pslice
    '''

    if pslice is None:
        pslice = slice(None)

    numTweezers=len(yData)
    if numTweezers>5:
        nrows=2

    else:
        nrows=1

    ncols=int(np.ceil(numTweezers/nrows))
    axIdxToSetLabel=[0,int(np.ceil(numTweezers/nrows))]

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(8*(ncols),6*(nrows)))
    ax=ax.flatten()

    minMax = [100,-100]
    legend_elements = {}
    for plotIdx, twLabel in enumerate(yData):
        roiIndex = getTwIdxFromLabel(twLabel)

        try:
            colorForDifferentLabels = colorSet[twLabel]
        except IndexError:
            colorForDifferentLabels = colorSet

        legend_elements[roiIndex] = []
        for idx, uniqueLabel in enumerate(legendlabels):
            legend_elements[roiIndex].append(Line2D([0], [0], color=colorForDifferentLabels[idx],\
                                                    label=labelFormat.format(uniqueLabel),\
                                                    marker = markerSet[roiIndex], markersize=markersize,linestyle=linestyle))

        scatterWithErrorsMarkers(xData[pslice],yData[twLabel][regionToPlot[twLabel][0]][pslice],\
                            yErrs[twLabel][regionToPlot[twLabel][0]][pslice],colorForDifferentLabels[colorIndices][pslice], markerSet[roiIndex], fig, ax[plotIdx])

        if axisLimit == None:
            if yType == 'phase':
                minMax = [min(minMax[0], np.min(yData[twLabel][regionToPlot[twLabel][0]])), max(minMax[1], np.max(yData[twLabel][regionToPlot[twLabel][0]]))]
                axlim = [a + b for a, b in zip(minMax, [-0.5, 0.5])]
            elif yType == 'norm':
                axlim = [0, 2]
            elif yType == 'totalCounts':
                axlim = [0, np.max(yData[twLabel][regionToPlot[twLabel][0]])+20]
            else:
                axlim = [-np.pi,np.pi]
        else:
            axlim = axisLimit
        ax[plotIdx].set_title(twLabel)
        ax[plotIdx].legend(handles=legend_elements[roiIndex], prop={'size': 8},bbox_to_anchor=(1., 1.0))

        ax[plotIdx].set_yscale(yscale)

    for plotIdx in range(numTweezers):
        ax[plotIdx].set_ylim(axlim)
        ax[plotIdx].grid()
        ax[plotIdx].set_xlabel(xlabel)
        if plotIdx in axIdxToSetLabel:
            ax[plotIdx].set_ylabel(ylabel)
        else:
            # ax[0].get_shared_y_axes().join(ax[0], ax[ax_idx])
            ax[plotIdx].set_yticklabels([])


    fig.tight_layout(pad=1.5)
    #plt.subplots_adjust(wspace=0, hspace=0)

    return fig, ax


def generateTweezerSubplots2D(xData, yData, yErrs,
                                regionToPlot,
                                colorSet,
                                colorIndices, markerSet, xlabel, ylabel, yType,
                                legendlabels, labelFormat, plotParam, labelParam, axisLimit = None,
                                linestyle='None', markersize=8, pslice=slice(None),yscale='linear', fit = None):
    '''
    xData is typically varyParams
    yData is whatever you want to plot
    yErrs is the error on yData
    regionToPlot is typically regionIdx
    colorSet is the list of colors you want to use, and colorIndices are the indices you want to index to. Both produced by colorsForUniqueLabels.
    markerSet is a list of markers for each plot
    xlabel and ylabel are just the labels for the axes
    yType is in the set {'phase', 'norm', and 'totalCounts'}, and is used to set axis limits if none are given.
    legendlabels are the unique labels for each run
    labelformat is labeltxt
    axisLimit is a 1x2 list of the min and max ylim, default is None
    linestyle is default None
    markersize is default None
    pslice
    '''

    if pslice is None:
        pslice = slice(None)

    numTweezers=len(yData)
    if numTweezers>5:
        nrows=2

    else:
        nrows=1

    ncols=int(np.ceil(numTweezers/nrows))
    axIdxToSetLabel=[0,int(np.ceil(numTweezers/nrows))]

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(8*(ncols),6*(nrows)))
    ax=ax.flatten()

    if yType == 'phase':
        vmin = -0.50*np.pi
        vmax = 0.25*np.pi
        label = 'phase (rad)'
    elif yType == 'counts':
        vmin = 0
        vmax = 350
        label = 'total counts'

    minMax = [100,-100]
    legend_elements = {}
    relevantIdxVals = {}
    for plotIdx, twLabel in enumerate(yData):
        roiIndex = getTwIdxFromLabel(twLabel)

        try:
            colorForDifferentLabels = colorSet[twLabel]
        except IndexError:
            colorForDifferentLabels = colorSet

        relevantIdxs = {}
        relevantIdxVals[twLabel] = {}
        allX = list(np.unique(xData))
        allY = list(np.unique(legendlabels))
        # allY = list(np.unique(allY))

        legend_elements[roiIndex] = []
        for idx, uniqueLabel in enumerate(legendlabels):
            legend_elements[roiIndex].append(Line2D([0], [0], color=colorForDifferentLabels[idx],\
                                                    label=labelFormat.format(uniqueLabel),\
                                                    marker = markerSet[roiIndex], markersize=markersize,linestyle=linestyle))
            relevantIdx = np.flatnonzero(colorIndices == idx)
            relevantIdxs[legendlabels[idx]] = relevantIdx

            for ii, relevantidx in enumerate(relevantIdx):
                xInd = allX.index(xData[pslice][relevantidx])
                yInd = allY.index(legendlabels[idx])
                relevantIdxVals[twLabel][(yInd,xInd)] = yData[twLabel][regionToPlot[twLabel][0]][pslice][relevantidx]
        # print(relevantIdxVals[twLabel])

        i,j = zip(*relevantIdxVals[twLabel].keys())
        vals2d = np.zeros((len(legendlabels), len(xData)//len(legendlabels)))
        np.add.at(vals2d, tuple((i,j)), tuple(relevantIdxVals[twLabel].values()))

        cmap2d = ax[plotIdx].imshow(vals2d, vmin = vmin, vmax = vmax, extent = (min(allX), max(allX), min(allY), max(allY)), origin = 'lower')
        cb2d = fig.colorbar(cmap2d, ax=ax[plotIdx], anchor=(0, 0.0), shrink=0.5)
        cb2d.set_label(label, rotation=270, labelpad=18.0)
        # ax[plotIdx].set_xticks(range(len(allX)))
        # ax[plotIdx].set_yticks(range(len(allY)))
        ax[plotIdx].set_xlabel(labelParam)
        ax[plotIdx].set_ylabel(plotParam)
        ax[plotIdx].tick_params(labelsize = 12)
        ax[plotIdx].set_title(twLabel)

        nX = 5
        nY = 2
        [l.set_visible(False) for (i,l) in enumerate(ax[plotIdx].xaxis.get_ticklabels()) if i % nX != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax[plotIdx].yaxis.get_ticklabels()) if i % nY != 0]

        if fit:
            allX, allY = np.meshgrid(allX, allY)

            init_guess = [max(vals2d.ravel()) - min(vals2d.ravel()), max(vals2d.ravel()) - min(vals2d.ravel()), 0, 50, min(vals2d.ravel())]
            coeff, covar = curve_fit(twoD_Quadratic, (allX, allY),
                                                  vals2d.ravel(),
                                                  p0= init_guess)
            # ((max(vals2d.ravel()) - min(vals2d.ravel())),0, 57, 100, 50, min(vals2d.ravel()))
            data_fitted = twoD_Quadratic((allX, allY), *coeff)

            print(coeff)
            data_fitted = np.reshape(data_fitted, (21, 17))

            ax[plotIdx].contour(allX, allY, data_fitted, 1, colors = 'w')

            # if twLabel == 'T4_5':
            #     ax[plotIdx].scatter(-17, 50, marker = '*', color = 'w', s = 20)
            #     ax[plotIdx].scatter(coeff[2], coeff[3], marker = '*', color = 'indigo', s = 20)

    fig.subplots_adjust(hspace = -0.20, wspace = 0.25)

    return fig, ax

def plotTimeFraction(fractionArrays,seqidx=0,showKeys=['T0','T1','T2','T3','T4'],colors=None):
    fig, ax =plt.subplots(figsize=(9,6))
    for twKey in showKeys:
        x = np.arange(len(fractionArrays[seqidx][twKey]))
        if colors:
            color=colors[twKey]
        else:
            color=None
        ax.scatter(x,fractionArrays[seqidx][twKey],label=twKey,color=color)

    ax.legend(bbox_to_anchor = (1.2,1))
    ax.grid()
    plt.show()

def plotTwoRegionFraction(fractionArrays,seqidx=0,showKeys=['T0','T1'],fp=None,fileName='',title=''):
    fig, ax =plt.subplots(figsize=(8,6))



    ax.plot(fractionArrays[seqidx][showKeys[0]],
               fractionArrays[seqidx][showKeys[1]],
                linestyle='none', marker='o',markersize=12,
               label=f'{showKeys[0]} vs. {showKeys[1]}',color='C0',markeredgecolor ='k')

    ax.set_xlabel(f'f4 fraction for {showKeys[0]}')
    ax.set_ylabel(f'f4 fraction for {showKeys[1]}')
    ax.set_title(title)
    ax.legend()
    ax.grid()
    if fp:
        plt.savefig(os.path.join(fp,fileName),dpi=250)
    plt.show()

def scatterManyTweezersFraction(fractionArrays,seqidx=0,showKeys=['T0','T1'],
                               fp=None,fileName='',title=''):
    '''
    Scatter atom fraction in a grid for different tweezers. for a single run of repeats

    Parameters
    ----------
    fractionArrays : dict
        Dictionary of fraction arrays for multiple sequences.
    seqidx : int, optional
        Sequence index to plot. The default is 0.
    showKeys : list, optional
        List of tweezer labels to plot. The default is ['T0','T1'].
    fp : string, optional
        File path of the directory to save the plot. The default is None.
    fileName : string, optional
        Plot filename to be saved. The default is ''.
    title : string, optional
        Plot title. The default is ''.

    '''
    noTweezers = len(showKeys)

    fractions = fractionArrays[seqidx].copy()

    for twIdx, twKey in enumerate(showKeys):
        fractions[twKey] = fractions[twKey]-np.mean(fractions[twKey])

    fig, ax =plt.subplots(noTweezers,noTweezers,figsize=(14,14),
                      sharex='col', sharey='row')
                        #gridspec_kw={'hspace': 0, 'wspace': 0})

    for twIdx1, twKey1 in enumerate(showKeys):
        for twIdx2, twKey2 in enumerate(showKeys):
            ax[twIdx1][twIdx2].plot(fractions[twKey2],
                       fractions[twKey1],
                        linestyle='none', marker='o',markersize=8,
                        color='C0',markeredgecolor ='k')

            if twIdx2==0:
                ax[twIdx1][twIdx2].set_ylabel(twKey1,rotation=0)

            if twIdx1==noTweezers-1:
                ax[twIdx1][twIdx2].set_xlabel(twKey2)



            ax[twIdx1][twIdx2].grid()

            ax[twIdx1][twIdx2].set_xlim([-0.22,0.22])
            ax[twIdx1][twIdx2].set_ylim([-0.22,0.22])

            ax[twIdx1][twIdx2].set_yticks([-0.2,0,0.2])
            ax[twIdx1][twIdx2].set_xticks([-0.2,0,0.2])

            ax[twIdx1][twIdx2].set_aspect(1)

            ax[twIdx1][twIdx2].tick_params(axis='x', labelsize=12)
            ax[twIdx1][twIdx2].tick_params(axis='y', labelsize=12)


    fig.suptitle(title)
    fig.subplots_adjust(top=1.2 )
    fig.set_tight_layout(True)
    if fp:
        plt.savefig(os.path.join(fp,fileName),dpi=250)
    plt.show()


def plotGeneralTweezerData(xdata, ydata, yerr, xlabel,ylabel,
                      xlim=[None,None], ylim=[None,None],
                      savePlot=False,  title=None, fp=None, fileName = 'SignalVsFraction.png',
                      colors=None,fig=None,ax=None,legend=True):
    '''


    Parameters
    ----------
    xdata : dict
        Dictionary of x data. Each key is the tweezer key.
    ydata : dict
        Dictionary of y data. Each key is the tweezer key.
    yerr : dict
       Dictionary of yerr data. Each key is the tweezer key. Can be None.
    xlabel : string

    ylabel : string

    xlim : list, optional
         The default is [None,None].
    ylim : list, optional
         The default is [None,None].
    savePlot : bool, optional
         The default is False.
    title : string, optional
        Plot title. The default is None.
    fp : string, optional
        File path for the image to be saved. The default is None.
    fileName : string, optional
        The default is 'SignalVsFraction.png'.
    colors : dcit, optional
        Dictionary of colors. Each key is the tweezer key. The default is None.
    fig : obj
        mpl fig. The default is None.
    ax : obj
        mpl ax. The default is None.


    Returns
    -------
    fig : obj
        mpl fig.
    ax : obj
        mpl ax.

    '''
    if not (fig and ax):
        fig, ax = plt.subplots(figsize=(9,6))

    for key in xdata.keys():
        if colors:
            color=colors[key]
        else:
            color=None
        if yerr:
            ax.errorbar(xdata[key], ydata[key], yerr[key], label=key, ls='none',color=color) # fmt='o' GM changed to ls='none' 07/15/2022
        else:
            ax.scatter(xdata[key], ydata[key], label=key,color=color)


    ax.grid()
    if legend:
        ax.legend(bbox_to_anchor = (1.15,1))
    ax.set_xlabel(xlabel)

    plt.ylabel(ylabel)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_title(title,fontsize=20)

    if savePlot:
        plt.savefig(os.path.join(fp, fileName), dpi=300, bbox_inches='tight')

    return fig, ax

def plotTweezerBars(ydata, yerr, xlabel,ylabel,
                       title=None, fp=None, fileName = 'SignalVsTweezer.png',
                      colors=None):

    ### plot the atom number fits vs tweezer ###
    fig, ax = plt.subplots(figsize=(11,6))

    for twKey in ydata:
        twIdx = getTwIdxFromLabel(twKey)
        ax.bar(twIdx, ydata[twKey], yerr=yerr[twKey], color = colors[twKey])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=18)
    ax.set_xticks(twIdxListFromLabelList(ydata.keys()))
    ax.grid()
    if fp:
        fig.savefig(os.path.join(fp, fileName), dpi=300, bbox_inches='tight')

    return fig, ax