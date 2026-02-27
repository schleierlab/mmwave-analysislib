# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:28:19 2018

@author: Quantum Engineer
"""

import os
import h5py
from decimal import Decimal
import time
import datetime
from numpy import arange, linspace, array, logspace
import numpy as np
import matplotlib.pyplot as plt

from analysis.image.process import getSummedDataArray
import analysis.data.h5lyze as hz
import math

### FOLDER READING

def getDirectory(target, d = None , m = None, y = None):
    """
    Target is either a directory or a labscript filename. If target is a directory,
    returns target. Otherwise target is interpreted as a labscript filename
    and it returns a directory string to data for the experiment 'labscript_filename'
    for d-m-y, or the present day if d-m-y is not provided
    """

    if os.path.isdir(target) and os.path.isabs(target):
        directory = target
    else:
        labscript_filename = target
        computer =  os.environ['COMPUTERNAME']

        if computer == 'RYD-EXPTCTRL':
            prefix = r'Z:\Experiments\rydberglab'
        elif computer == 'RYD-ANALYSIS1':
            prefix = r'Z:\Experiments\rydberglab'
        else:
            prefix = r'Z:\Experiments\rydberglab'

        directory = os.path.join(prefix,labscript_filename)

        d_today = datetime.date.today().timetuple()
        d_passed = [y, m, d]
        datestr = r''
        for index in range(3):
            if d_passed[index]:
                item = format(d_passed[index],'02')
            else:
                item = format(d_today[index],'02')
            datestr = os.path.join(datestr,item)

        directory = os.path.join(directory,datestr)

    return directory


def autoFolder(target):
    """
    Given a filepath fp, returns the most recently modified folder in fp
    """

    fp = getDirectory(target)

    folders = [d for d in os.listdir(fp) if \
               os.path.isdir(os.path.join(fp,d))]

    def ftime(f):
        try:
            return int(f[0:4])
        except:
            return False

    # remove nonstandard names
    folders = [f for f in folders if ftime(f)]

    # get times
    indices = [ftime(f) for f in folders]

    newest_folder = folders[np.argmax(indices)]
    directory = os.path.join(fp, newest_folder)

    print('autoFolder found: ' + directory)

    return directory


#### SHOT READING

def getShots(directory, keyword = '', badword = None , nshots = None):
    """
    Given a directory, returns an array of nshots h5 filepaths containing the
    keyword given keyword
    """

    files = os.listdir(directory)
    files = [f for f in files if (keyword in f)]

    if badword:
        files = [f for f in files if not (badword in f)]

    h5filepaths=[]

    for index, item in enumerate(files):
        if item.endswith(".h5"):
            h5filepaths.append(os.path.join(directory,item))

    h5filepaths = h5filepaths[:nshots]

    return h5filepaths


def getCompleteShots(shots, ignoreTime=5):
    """
    Given a list of shots, this returns all shots that have completed
    """

    completedShots = []

    for shot in shots:
        with h5py.File(shot,'r') as f:
            subgroupDict = getSubgroupDict(f)
        if ('data' or 'images' or 'AndorIxonImages') in subgroupDict:
            #ignore shots modified within the last ignoreTime seconds:
            if time.time() - os.path.getmtime(shot) > ignoreTime:
                completedShots.append(shot)

    return completedShots


def getIncompleteShots(shots):
    """
    Given a list of shots, this returns all shots that have not begun running
    Warning: a shot in the middle of running will not be returned
    """

    incompleteShots = []

    for shot in shots: #only keep shots that have not run
        with h5py.File(shot,'r') as f:
            attrDict = getAttributeDict(f)
        if not 'run time' in attrDict:
            incompleteShots.append(shot)

    return incompleteShots


def getRunningShot(shots):
    """
    Given a list of shots, returns the shot that is currently running or None
    if no shot is running
    """

    runningShot = None

    for shot in shots:
        with h5py.File(shot,'r') as f:
            attrDict = getAttributeDict(f)
            subgroupDict = getSubgroupDict(f)

        if ('run time' in attrDict):
            if not ('data' or 'images' or 'AndorIxonImages') in subgroupDict:
                runningShot = shot

    return runningShot


def getLastModifiedShot(shots):
    """
    From a list of shots, this function returns the one that has been modified
    most recently. It could be the one currently running, or the last one queued
    depending on if shots are in the process of being generated
    """

    latest_shot = None
    latest_mtime = None

    for shot in shots:
        mtime = os.path.getmtime(shot)

        if (not latest_shot) or (mtime > latest_mtime):
            latest_shot = shot
            latest_mtime = mtime

    return latest_shot


def getAttributeDict(group):
    """
    Returns a dictionary of attributes for <group> in an h5 file. Usage is:

        with h5py.File(shot, 'r') as f:
            attrDict = getAttributeDict(f['some group'])
    """

    if type(group) == str: #if we are passed a filepath
        group = h5py.File(group,'r')

    items = group.attrs.items()
    attributeDict = dict()

    for item in items:
        name = item[0]
        attributeDict[name] = item[1]

    return attributeDict


def getSubgroupDict(group):
    """
    Returns a dictionary of items for <group> in an h5 file. Usage is:

        with h5py.File(shot, 'r') as f:
            attrDict = getAttributeDict(f['some group'])
    """

    items = group.items()
    subgroupDict = dict()

    for item in items:
        name = item[0]
        subgroupDict[name] = item[1]

    return subgroupDict


def getFilename(shot):
    """
    Given directory/<>.h5, returns <>.h5
    """

    index1 = shot.rfind('\\')
    name = shot[index1 + 1:]
    return name


def getSequenceID(shot):
    """
    Given a shot, returns the ID (i.e. timestamp_experiment) for the associated sequence
    """

    with h5py.File(shot,'r') as f:
        attributeDict = getAttributeDict(f)

    sequenceID = attributeDict['sequence_id']

    return sequenceID


def getRunTime(shot):
    """
    Returns the time at which shot began running, or None if it has not run
    """

    with h5py.File(shot,'r') as f:
        attributeDict = getAttributeDict(f)

    if 'run time' in attributeDict:
        runtime = attributeDict['run time']
        index = runtime.find('T')
        runtime = runtime [index+1:]
    else:
        runtime = None

    return runtime


def getTimestamp(shot):
    """
    Given a shot, returns abbreviated timestamp in the form HHMMSS that it was queued
    """

    with h5py.File(shot,'r') as f:
        attributeDict = getAttributeDict(f)

    sequence_id = attributeDict['sequence_id']
    index1 = sequence_id.find('T')
    timestamp = sequence_id[index1+1 : index1 + 7]

    return timestamp

def getSequenceDictionary(shots):
    """
    Given a list of shots, this function sorts them into sequences partitioned
    by sequenceID and stores in them in a dictionary with keys given by
    sequence prefixes, i.e. <key>_shotNumber.h5
    """

    sequenceDict = dict()

    for shot in shots:
        with h5py.File(shot,'r') as f:
            attributeDict = getAttributeDict(f)
        sequenceID = attributeDict['sequence_id']

        if (attributeDict['n_runs'] == 1):
            sequenceID += '_repeats'

        if ('run repeat' in attributeDict) and not (attributeDict['n_runs'] == 1):
            repeatNo = attributeDict['run repeat']
            sequenceID += '_repeats_'+str(repeatNo)

        if sequenceID in sequenceDict:
            sequenceDict[sequenceID].append(shot)
        else:
            sequenceDict[sequenceID] = [shot]

    return sequenceDict

def getParameterChangesFromFile(f):
    '''
    Given an open h5 file, return parameter changes.


    Parameters
    ----------
    f : TYPE
        DESCRIPTION.

    Returns
    -------
    parameters : TYPE
        DESCRIPTION.

    '''
    parameters = dict()
    subgroupDict = getSubgroupDict(f['globals'])

    for name, subgroup in subgroupDict.items():
        itemList = f['globals'][name].attrs.items()
        for param, strvalue in itemList:
            if ('arange' in strvalue) or ('[' in strvalue) or ('linspace' in strvalue):
                try:
                    parameters[param] = eval(strvalue)
                except:
                    # there is a variable referenced in the string
                    g = hz.attributesToDictionary(f['globals'])
                    for group in g.values():
                        for k, v in group.items():
                            if k in strvalue:
                                strvalue = strvalue.replace(k, str(v))

                    parameters[param] = eval(strvalue)

    parameters = dict(sorted(parameters.items()))

    return parameters

def getParameterChanges(shot):
    """
    Given one shot, returns all parameter names with values of type list
    """

    parameters = dict()

    with h5py.File(shot,'r') as f:
        subgroupDict = getSubgroupDict(f['globals'])

        for name, subgroup in subgroupDict.items():
            itemList = f['globals'][name].attrs.items()
            for param, strvalue in itemList:
                if ('arange' in strvalue) or ('[' in strvalue) or ('linspace' in strvalue) or ('logspace' in strvalue):
                    try:
                        parameters[param] = eval(strvalue)
                    except:
                        # there is a variable referenced in the string
                        g = hz.attributesToDictionary(f['globals'])
                        for group in g.values():
                            for k, v in group.items():
                                if k in strvalue:
                                    strvalue = strvalue.replace(k, str(v))

                        parameters[param] = eval(strvalue)

    return parameters


def getParameterChanges2(shot0, shot1):
    """
    Given two shots, returns all parameters names with values that are different
    between two filepaths and their values. fp1 and fp2 are complete paths
    including filename.h5
    """

    parameterDict = dict()

    with h5py.File(shot0,'r') as f0, h5py.File(shot1,'r') as f1:

        params0 = getAttributeDict(f0['globals'])
        params1 = getAttributeDict(f1['globals'])

        for key in params0:
            if key in params1:
                if not np.array_equal(params0[key],params1[key]):
                    parameterDict[key] = np.array([params0[key],params1[key]])

    parameterDict = dict(sorted(parameterDict.items()))


    return parameterDict

def getParameterChanges3(target, folder=None, nshots=2,day=None,month=None,year=None):
    """
    Given a target (and optionally a subfolder) with at least two shots in it,
    returns the parameter changes between the shots with getParameterChanges2
    """

    if type(target) == list:
        shots = target
    else:
        directory = getDirectory(target,day,month,year)

        if folder:
            directory = os.path.join(directory,folder)

        shots = getShots(directory)


    referenceShot = shots[0]
    testShots = shots[1:nshots]
    parameters = []

    #find names of parameters changing between referenceShot and testShots
    for shot in testShots:
        parameterDict = getParameterChanges2(referenceShot, shot)
        parameters = list(set().union(parameterDict.keys(), parameters))


    parameterDict = dict()

    #find parameter values for each parameter
    for param in parameters:
        vals = []

        for shot in shots:
            myval = getParam(shot, param)

            if type(myval)==bytes:
                myval = myval.decode("utf-8")

            vals.append(myval)

        parameterDict[param] = np.array(vals)

    #not sure why this is getting sorted, but it shouldn't change the parameter
    #value order
    parameterDict = dict(sorted(parameterDict.items()))


    return parameterDict



def getParameterDifferences(shot0,shot1):
    """
    This function returns all variable names that are present in one file
    but not present in the other. fp1 and fp2 are complete paths including filename.h5
    """
    parameters = [] #variables not present between files

    with h5py.File(shot0,'r') as f0, h5py.File(shot1,'r') as f1:

        params0 = getAttributeDict(f0['globals'])
        params1 = getAttributeDict(f1['globals'])

        for key in params0:
            if not key in params1:
                parameters.append(key)

        for key in params1:
            if not key in params0:
                parameters.append(key)

    return parameters


def getParam(shot,parameter):
    with h5py.File(shot,'r') as f:
            items = f['globals'].attrs.items()
            for item in list(items):
                if parameter == item[0]:
                    paramVal = item[1]
                    if type(paramVal)==bytes:
                        paramVal = paramVal.decode("utf-8")
                    return paramVal

    return None

def getParamArray(shots, parameter):
    """
    Given an array of shot filepaths and a paremter name, returns an array of
    values that the parameter takes
    """

    param_array = []

    for shot in shots:
        param_value = getParam(shot,parameter)
        param_array.append(param_value)

    return param_array


def delimitSubfolders(fp):
    """
    Given a filepath, returns a dictionary of shots
    """

    delimDict = dict()

    directory = getDirectory(fp)

    folders = [d for d in os.listdir(directory) if \
           os.path.isdir(os.path.join(directory,d))]

    for folder in folders:
        shots = getShots(os.path.join(fp,folder))
        delimDict[folder] = shots

    return delimDict


def delimitShots(shots, delimParameter):
    """
    Given a set of completed shots and a parameter to delineate between, this
    function returns a dictionary with shots sorted by keys taking values
    of the delimParameter
    """

    delimDict = dict()

    for fp in shots:
        if delimParameter == 'sequence_id':
            value = getTimestamp(fp)
        else:
            with h5py.File(fp,'r') as f:
                value = f['globals'].attrs[delimParameter]

        if not (value in delimDict):
            delimDict[value] = [fp]
        else:
            delimDict[value].append(fp)

    return delimDict

def delimitShotsManyParameters(shots, parameters):
    """
    Given a set of completed shots and parameters to delineate between, this
    function returns a dictionary with shots sorted by keys taking values
    of the delimParameters
    """

    delimDict = dict()
    paramLists = []
    folderList = []

    for fp in shots:
        with h5py.File(fp,'r') as f:
            paramList = []
            mystr = ''
            for param in parameters:
                paramVal = f['globals'].attrs[param]
                if type(paramVal)==bytes:
                    paramVal = paramVal.decode("utf-8")
                if isinstance(paramVal,float):
                    paramVal = np.round(paramVal,10)
                paramList.append(paramVal)

                if len(mystr)>0:
                    mystr += '_'
                mystr += param + '_' + str(paramVal)

        if not paramList in paramLists:
            paramLists.append(paramList)
            folderList.append(mystr)

        index = paramLists.index(paramList)
        key = folderList[index]

        if not (key in delimDict):
            delimDict[key] = [fp]
        else:
            delimDict[key].append(fp)

    return delimDict #, paramLists, folderList


def getSequencesParameter(folders):
    '''
    Return parameters that change across sequences given by folders; ignore those that change within.
    '''

    units = {}
    testShots = []
    for folder in folders:
         shots = getShots(folder, nshots=1)
         testShots.append(shots[0])

    paramDict = getParameterChanges3(testShots, nshots = len(testShots))

    shotsSingleFolder = getShots(folders[0])
    singleSeqParameters = getParameterChanges3(shotsSingleFolder, nshots=len(shotsSingleFolder))

    for param in paramDict:
        units[param] = getUnit(shots[0], param)

    return paramDict, units, singleSeqParameters



def getUnit(shot, parameter, scaleFactor = None):
    """
    Given a shot and a parameter key, this function returns the unit associated
    with the parameter value
    """

    units = None

    with h5py.File(shot,'r') as f:
        subgroupDict = getSubgroupDict(f['globals'])

        for name, subgroup in subgroupDict.items():
            unitDict = getAttributeDict(f['globals'][name]['units'])
            if parameter in unitDict:
                units = unitDict[parameter]

    if scaleFactor and (scaleFactor != 1):
        units += ' / {:.2e}'.format(Decimal(scaleFactor))

    return units


### DIRECTORY CLEANING AND FILE MANAGEMENT

def deleteDuds(target):
    """
    Deletes dud folders, which happen when runmanager tries to queue a shot but
    it fails before the h5 file can be populated, so it only has a 'globals'
    group return
    """

    directory = getDirectory(target)

    shots = getShots(directory)

    for shot in shots:
        if len(h5py.File(shot,'r').items()) == 1:
            os.remove(shot)


def deleteCompleteShots(target):
    """
    Deletes all shots which have completed
    """
    directory = getDirectory(target)


    shots = getShots(directory)
    shots = getCompleteShots(shots)

    for shot in shots:
        os.remove(shot)


def deleteIncompleteShots(target):
    """
    Delets all shots that have not completed. It only considers shots that
    were queued before the current sequence.
    """
    directory = getDirectory(target)

    shots = getShots(directory)
    complete_shots = getCompleteShots(shots)

    if complete_shots:
        latest_shot = getLastModifiedShot(complete_shots)
    else:
        latest_shot = getLastModifiedShot(shots)

    latest_timestamp = getTimestamp(latest_shot)

    for shot in shots:
        timestamp = getTimestamp(shot)
        if (timestamp < latest_timestamp) and not (shot in complete_shots):
            os.remove(shot)


def deleteRepeatShots(target):
    """
    Deletes all completed repetitions
    """
    directory = getDirectory(target)

    shots = getShots(directory)
    shots = getCompleteShots(shots)

    for shot in shots:
        with h5py.File(shot,'r') as f:
            attributeDict = getAttributeDict(f)

        if 'run repeat' in attributeDict:
            os.remove(shot)

    return

def delete_trash_shots(target):
    """
    Delete all shots with the description "trash"
    This method is intended for use with the control server which queues such
    shots as an analysis buffer between actual measurements.

    Parameters
    ----------
    target : string
        Path to directory in which to remove trash shots.

    Returns
    -------
    None.

    """
    # Get the shots
    directory = getDirectory(target)

    shots = getShots(directory)
    shots = getCompleteShots(shots)

    for shot in shots:
        with h5py.File(shot,'r') as f:
            attributeDict = hz.attributesToDictionary(f['globals'])
            description = attributeDict['Constants']['C_sequenceDescription']
            try:
                description = description[1:-1]
            except:
                pass

        if description=='trash':
            print("TRYING TO REMOVE TRASH SHOT")
            # If the description is "trash", get rid of the shot
            os.remove(shot)



def deleteSingleShotSequences(target):
    """
    Deletes all sequences that only have one shot (i.e. no variables swept)
    """
    directory = getDirectory(target)
    shots = getShots(directory)
    shots = getCompleteShots(shots)
    sequenceDict = getSequenceDictionary(shots)

    for sequence, shots in sequenceDict.items():
        with h5py.File(shots[0],'r') as f:
            attributeDict = getAttributeDict(f)
        if attributeDict['n_runs'] == 1:
            for shot in shots:
                os.remove(shot)
    return

def moveCompletedFiles(target, lastFolder = False, deleteRepeats = False,
                       ignoreRepeats = False, deleteSingles = False, appendDescription = True,
                       appendRepeatNumber = False,
                       singleFolder = False, d = None, m = None, y = None,
                       ignoreTime=5, delete_trash=False):
    """
    This function moves all complted files in a directory into folders named
    by the sequence and the variable (if something was scanned) or repeats.
    Directory can either be a full filepath or the name of a labscript file
    """
    directory = getDirectory(target,d,m,y)
    # deleteDuds(directory)

    if deleteRepeats:
        deleteRepeatShots(directory)

    if deleteSingles:
        deleteSingleShotSequences(directory)

    if delete_trash:
        delete_trash_shots(directory)

    shots = getShots(directory)

    if ignoreRepeats:
        shots = [s for s in shots if not '_rep' in s]

    shots = getCompleteShots(shots, ignoreTime)

    sequenceDict = getSequenceDictionary(shots)

    target_directory = None
    directories = []

    for sequence, shots in sequenceDict.items():

        folders = [f for f in os.listdir(directory) if \
               os.path.isdir(os.path.join(directory,f))]

        target_directory = None

        if lastFolder:
            if len(folders)>0:
                target_directory = os.path.join(directory,folders[-1])
            else:
                raise Exception('There is no lastFolder')

        else:
            foldername = getTimestamp(shots[0])
            parameterNames = getParameterChanges(shots[0])

            for variable in parameterNames:
                foldername += '__' + variable

            sequenceNameSplit = sequence.split('_')

            if (not parameterNames) or 'repeats' in sequenceNameSplit or singleFolder:
                foldername += '__repeats'

                if appendRepeatNumber:
                    if not (sequenceNameSplit[-1] == 'repeats'):
                        foldername += '_'+sequenceNameSplit[-1]


            if appendDescription:
                description = getParam(shots[0], 'C_sequenceDescription')
                if (description is not None) and (len(description) > 0):
                    foldername += ('__' + description)


            for folder in folders:
                if foldername in folder:
                    target_directory = os.path.join(directory, folder)
                    break

            if not target_directory:
                target_directory = os.path.join(directory, foldername)
                os.mkdir(target_directory)

        directories.append(target_directory)
        for oldfp in shots:
            filename = getFilename(oldfp)
            newfp = os.path.join(target_directory, filename)

            os.rename(oldfp, newfp)

    return directories

def padFolderNamesWithZeros(target, folder=None, d = None , m = None, y = None):

    directory = getDirectory(target, d = d , m = m, y = y)

    negNum = False

    if folder:
        directory = os.path.join(directory,folder)

    subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
    subfolderNum = []

    maxPadLength = 0
    for subfolder in subfolders:
        splitSubfolder = str.split(subfolder, '_')
        try:
            checkFloat = float(splitSubfolder[-1])
            subfolderNum.append(float(splitSubfolder[-1]))
        except:
            raise Exception('The last part of (at least) one of the folder names is not numeric!')

        numLength = len(splitSubfolder[-1])

        if float(splitSubfolder[-1]) < 0:
            negNum = True

        if numLength > maxPadLength:
            maxPadLength = numLength

    newSubfolders = []

    ### needed to sort negative and positive folder names
    subfolderNum = np.array(subfolderNum)
    subfoldersNeg = np.array(subfolders)

    if negNum == True:
        idx = [x for _, x in sorted(zip(subfolderNum, range(len(subfolderNum))))]
        subfoldersNeg = subfoldersNeg[idx]
        subfolderNum = subfolderNum[idx]
        folderCount = 0

        for subfolder in subfoldersNeg:
            subdir = os.path.join(directory, subfolder)

            splitSubfolder = str.split(subfolder, '_')
            if '.' not in splitSubfolder[-1]:
                splitSubfolder[-1] = str(folderCount).zfill(maxPadLength)
                newSubfolder = '_'.join(splitSubfolder)
            elif '.' in splitSubfolder[-1] and 'e' not in splitSubfolder[-1]:
                splitSubfolder[-1] = folderCount.ljust(maxPadLength, '0')
                newSubfolder = '_'.join(splitSubfolder)
            else:
                newSubfolder = subfolder
            newSubfolders.append(newSubfolder)
            newSubdir = os.path.join(directory, newSubfolder)
            folderCount += 1

            os.rename(subdir, newSubdir)

    else:
        for subfolder in subfolders:
            subdir = os.path.join(directory, subfolder)

            splitSubfolder = str.split(subfolder, '_')
            if '.' not in splitSubfolder[-1]:
                splitSubfolder[-1] = str(splitSubfolder[-1]).zfill(maxPadLength)
                newSubfolder = '_'.join(splitSubfolder)
            elif '.' in splitSubfolder[-1] and 'e' not in splitSubfolder[-1]:
                splitSubfolder[-1] = splitSubfolder[-1].ljust(maxPadLength, '0')
                newSubfolder = '_'.join(splitSubfolder)
            else:
                newSubfolder = subfolder
            newSubfolders.append(newSubfolder)
            newSubdir = os.path.join(directory, newSubfolder)

            os.rename(subdir, newSubdir)


def sortExpansionSequence(target, sequenceParameters=[], parameters=[],
                          deleteRepeats=False, deleteSingles=False,
                          multipleSequences=False,folder=None, d = None, m = None, y = None):
    """
    This function sorts a folder of h5 files into subfolders based on a list of
    parameters that change within any single subfolder. If parameters is left
    empty, the function will find them automatically (but this is slow if multiple
    sequences are used). If you want to sort the data as it comes in you will
    either need to supply 'parameters' or run everything as a single expansion
    sequence. If 'parameters' is supplied, 'sequenceParameters' is not necessary.
    """

    #load directory
    directory = getDirectory(target,d,m,y)
    if folder:
        directory = os.path.join(directory,folder)


    folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]

    #clean directory
    deleteDuds(directory)

    if deleteRepeats:
        deleteRepeatShots(directory)

    if deleteSingles:
        deleteSingleShotSequences(directory)

    shots = getShots(directory)
    shots = getCompleteShots(shots)

    # get parameter changes if not provided
    if len(parameters)==0:
        if multipleSequences:
            parameters = list(getParameterChanges3(shots, nshots=len(shots)).keys())
        else:
            parameters = list(getParameterChanges(shots[0]))

    # remove parameters that should be changing within each folder
    for param in sequenceParameters:
        if param in parameters:
            parameters.remove(param)

    # sort shots by parameters that change across folders
    delimDict = delimitShotsManyParameters(shots, parameters)
    # print(delimDict)

    # move sorted shots into appropriate folders
    for folder in delimDict:
        shots = delimDict[folder]

        targetdirectory = os.path.join(directory,folder)

        if not folder in folders:
            os.mkdir(targetdirectory)

        for oldfp in shots:
            filename = getFilename(oldfp)
            newfp = os.path.join(targetdirectory,filename)
            os.rename(oldfp,newfp)
    return

def sort_expansion_sequence(directory, sequenceParameters=[], parameters=[],
                          deleteRepeats=False, deleteSingles=False,
                          multipleSequences=False, d = None, m = None, y = None):
    """
    This function sorts a folder of h5 files into subfolders based on a list of
    parameters that change within any single subfolder. If parameters is left
    empty, the function will find them automatically (but this is slow if multiple
    sequences are used). If you want to sort the data as it comes in you will
    either need to supply 'parameters' or run everything as a single expansion
    sequence. If 'parameters' is supplied, 'sequenceParameters' is not necessary.
    """

    directories = []

    folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]

    #clean directory
    deleteDuds(directory)

    if deleteRepeats:
        deleteRepeatShots(directory)

    if deleteSingles:
        deleteSingleShotSequences(directory)

    shots = getShots(directory)
    shots = getCompleteShots(shots)

    # get parameter changes if not provided
    if len(parameters)==0:
        if multipleSequences:
            parameters = list(getParameterChanges3(shots, nshots=len(shots)).keys())
        else:
            parameters = list(getParameterChanges(shots[0]))

    # remove parameters that should be changing within each folder
    for param in sequenceParameters:
        if param in parameters:
            parameters.remove(param)

    # sort shots by parameters that change across folders
    delimDict = delimitShotsManyParameters(shots, parameters)
    # print(delimDict)

    # move sorted shots into appropriate folders
    for folder in delimDict:
        shots = delimDict[folder]

        targetdirectory = os.path.join(directory,folder)
        directories.append(targetdirectory)

        if not folder in folders:
            os.mkdir(targetdirectory)

        for oldfp in shots:
            filename = getFilename(oldfp)
            newfp = os.path.join(targetdirectory,filename)
            os.rename(oldfp,newfp)
    return directories

def separate100Shots(target, sequenceParameters=[], parameters=[],
                          deleteRepeats=False, deleteSingles=False,
                          multipleSequences=False,folder=None, d = None, m = None, y = None):
    '''
    This function sorts the shots in target folder into subfolders with 100 shots in
    it each. Frankenstein-ed from sortExpansionSequence above and moveByPattern below.
    '''

    n = 100

    #load directory
    directory = getDirectory(target,d,m,y)
    if folder:
        directory = os.path.join(directory,folder)

    # folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]

    if deleteRepeats:
        deleteRepeatShots(directory)

    if deleteSingles:
        deleteSingleShotSequences(directory)

    shots = getShots(directory)
    shots = getCompleteShots(shots)

    shots_count = 0
    folder_count = 0

    # for shot in shots:
    for shot in shots:
        if folder_count < len(shots)/n:
            folder = str('shots__' + str(folder_count))
            targetdirectory = os.path.join(directory,folder)
            if not os.path.isdir(targetdirectory):
                os.mkdir(targetdirectory)

            filename = getFilename(shot)
            newfp = os.path.join(targetdirectory,filename)
            os.rename(shot,newfp)

            shots_count += 1
            if shots_count == n:
                folder_count += 1
                shots_count = 0

    padFolderNamesWithZeros(directory)

    return

def moveByPattern(target, pattern, folder):
    """
    Moves all h5 files with filenames containing the given pattern to destination,
    which is either the name of a folder or a full path
    """

    #load directory
    directory = getDirectory(target)

    # get matching shots
    shots = getShots(directory)
    shots = [s for s in shots if pattern in s]

    # create desination if it doesn't exist
    dst = os.path.join(directory, folder)
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # move shots
    for oldfp in shots:
        filename = getFilename(oldfp)
        newfp = os.path.join(dst,filename)
        os.rename(oldfp,newfp)

def waitForShot(shot, sleep_time=1, timeout=None, verbose=False):
    """
    Sleep until a shot is finished
    """
    t_start = time.time()

    while True:
        try:
            with h5py.File(shot, 'r') as f:
                try:
                    f.attrs['run time']
                    if 'data' in f:
                        if verbose:
                            print('shot complete')
                        break
                    else:
                        if verbose:
                            print('shot started but not complete')
                except:
                    if verbose:
                        print('shot created but not started')
        except:
            if verbose:
                print('shot not created yet')
        time.sleep(sleep_time)

        if timeout is not None:
            if time.time() - t_start > timeout:
                raise Exception(f'Timeout ({timeout} s) while waiting for shot {shot}')

### AUTOMATED ANALYSIS FUNCTIONS
def optimize_sequence(fp, roi, backgroundfp=None, rejectEndpoints=True,
                      plot=False):

    shots = getShots(fp)
    sequence = getCompleteShots(shots)

    parameters = getParameterChanges(sequence[0])
    parameters = list(parameters.keys())

    if not parameters:
        parameters = ['C_repeats']

    plotParameter = parameters[0]
    paramArray = getParamArray(sequence, plotParameter)
    paramArray = np.array(paramArray)

    imageArray = adl.getImageArray(sequence)
    summedData, normData = getSummedDataArray(imageArray, roi)

    max_idx = np.argmax(summedData)
    max_signal = summedData[max_idx]
    max_param = paramArray[max_idx]

    if rejectEndpoints:
        if max_idx == 0:
            raise Exception('Optimal shot is first in sequence. Recommended to add more points below scan.')
        if max_idx == len(shots)-1:
            raise Exception('Optimal shot is last in sequence. Recommended to add more points beyond scan.')

    if plot:
        fig, ax = plt.subplots()
        plt.scatter(paramArray, summedData)
        plt.scatter([max_param], [max_signal], color='r')
        plt.xlabel(plotParameter)
        plt.ylabel('counts')

        plt.savefig(os.path.join(fp, 'optimization.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    return max_param


### Lyse-dependent analysis functions
def shotHasAtoms(shot, roi_name, threshold):
    with h5py.File(shot,'r') as f:
        roi_data = hz.datasetsToDictionary(f['analysis']['regions'][roi_name])

        signal = roi_data['count']['3'] + roi_data['count']['4']

        if signal > threshold:
            return True
        else:
            return False

def shotsWithAtoms(shots, roi_name, threshold_multiplier=0.5):
    signals = {}

    for shot in shots:
        roi_data = hz.datasetsToDictionary(f['analysis']['regions'][roi_name])
        signals[shot] = roi_data['count']['3'] + roi_data['count']['4']

    threshold = np.median(list(signals.values()))

    shots_with_atoms = [s for s in shots if signals[s] > threshold*threshold_multiplier]


def shots_without_atoms(shots, threshold_multiplier=0.5, absolute_threshold=100):
    """
    Parameters
    ----------
    shots : list of strings
        list of paths to the h5 files you want to check for atoms.
    threshold_multiplier : float, optional
        Multiplies the average number of counts in the provided list of files.
        The default is 0.5.
    absolute_threshold : int or float, optional
        A lower bound below which all shots will be considered without atoms.
        This is meant to catch cases where most of the shots don't have atoms
        in which case the median counts will also be low. The default is 100.

    Returns
    -------
    shots_without_atoms : list of strings
        A list with all the atoms this silly function thinks don't have atoms.

    """

    signals = {}
    images_array = adl.getImageArray(shots, excludeBlanks=True)

    # Open shots and calculate total counts
    for shot in shots:
        with h5py.File(shot,'r') as f:
            counts = hz.datasetsToDictionary(f["analysis"]["regions"]["T0"]["count"])
            signals[shot] = counts['3'][0] + counts['4'][0]

    # Check which shots have atoms
    threshold = np.median(list(signals.values()))
    shots_without_atoms = [s for s in shots if ((signals[s] < threshold*threshold_multiplier) or (signals[s]<absolute_threshold))]

    return shots_without_atoms

if __name__ == '__main__':
#    directory = r'X:\rydberglab\dipoleTrap\2019\04\24\000000_testing'\
#    directory = r'C:\Users\Jacob\Stanford Drive\SchleierLab\Jacob\Temporary Analysis\sort_testing'

#    parameters = ['MW1_Pulse1_Length', 'C_sequenceDescription']
#    sequenceParameters = ['C_repeats']
#
#    sortExpansionSequence(directory, parameters=parameters)
    pass