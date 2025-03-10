# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:11:12 2023

@author: sslab
"""
import sys
root_path = r"C:\Users\sslab\labscript-suite\userlib\analysislib"

if root_path not in sys.path:
    sys.path.append(root_path)

try:
    lyse
except:
    import lyse

from analysis.data import h5lyze as hz
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt



# Is this script being run from within an interactive lyse session?
if lyse.spinning_top:
    # If so, use the filepath of the current h5_path
    h5_path = lyse.path
else:
    # If not, get the filepath of the last h5_path of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

with h5py.File(h5_path, mode='r+') as f:
    g = hz.attributesToDictionary(f['globals'])
    info_dict = hz.getAttributeDict(f)
    images = hz.datasetsToDictionary(f['manta419b_mot_images'], recursive=True)


image_types = list(images.keys())
print(images[image_types[0]])
test_sum = {'sum': sum(sum(images[image_types[0]]))}
# print(test_sum)

exposure_time = 1e-5
print(np.round(1e+6*exposure_time))


px = 5.5
magnification = 0.4
pixels = images[image_types[0]].shape[0]
fov = pixels*px*magnification

plt.figure("Raw")
plt.imshow(images[image_types[0]], extent = [0, fov, 0, fov])
plt.xlabel("x [um]")
plt.ylabel("y [um]")

plt.figure("BKG")
plt.imshow(images[image_types[1]], extent = [0, fov, 0, fov])
plt.xlabel("x [um]")
plt.ylabel("y [um]")

plt.figure("SUB")
plt.imshow(images[image_types[0]] - images[image_types[1]], extent = [0, fov, 0, fov])
plt.xlabel("x [um]")
plt.ylabel("y [um]")
# plt.imshow(images[image_types[1]])

with h5py.File(h5_path, mode='a') as f:

    analysis_group = f.require_group('analysis')
    region_group = analysis_group.require_group('test')

    hz.dictionaryToDatasets(region_group, test_sum, recursive=True)