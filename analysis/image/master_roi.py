# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:45:08 2021

@author: Quantum Engineer

"""

import numpy as np

fast_kinetics_shift = 204
rx = 5
ry = 5


# previewROIBox = ((575, 800), (230, 290))
previewROIBox = ((450, 880), (250, 350))

preview_rois = {4: previewROIBox,
                3: (previewROIBox[0], (previewROIBox[1][0] + 2*fast_kinetics_shift, previewROIBox[1][1] + 2*fast_kinetics_shift))}

# preview_rois = {4: (previewROIBox[0], (previewROIBox[1][0] + fast_kinetics_shift, previewROIBox[1][1] + fast_kinetics_shift)),
#                3: (previewROIBox[0], (previewROIBox[1][0] + 2*fast_kinetics_shift, previewROIBox[1][1] + 2*fast_kinetics_shift))}

# center_offset = np.array([0, -12])
# center_offset = np.array([-1, -8])
center_offset = np.array([-1, 0])

# PROTIP: are the centers outdated? find them again with tweezerBlobDetection.py

# Updated 10/11/2021
# These are values before moving the tweezers due to optimizing the first AOD efficiency on 03/09/2022
# centers_4 = np.array([[770., 948.],
#         [710., 949.],
#         [697., 949.],
#         [685., 949.],
#         [673., 950.],
#         [660., 950.],
#         [648., 950.],
#         [635., 950.],
#         [623., 951.],
#         [611., 951.]]).astype('int') + center_offset - np.array([0, 2*fast_kinetics_shift])

# Values after move
# centers_4 = np.array([[707, 248],
#         [646, 249],
#         [634, 249],
#         [621, 250],
#         [609, 250],
#         [596, 250],
#         [583, 250],
#         [571, 251],
#         [558, 251],
#         [546, 251]])

# values after new frequencies
# centers_4 = np.array([[721, 247],
#         [709, 247],
#         [697, 248],
#         [684, 248],
#         [672, 248],
#         [660, 249],
#         [647, 249],
#         [635, 249],
#         [622, 249],
#         [610, 249]])

# centers_3 = np.array([[608, 932],
#         [633, 931],
#         [658, 931],
#         [682, 930],
#         [707, 929],
#         [731, 928],
#         [755, 928],
#         [780, 927],
#         [804, 927],
#         [828, 926]]) + center_offset 

# centers_3 = np.array([[606, 712],
#        [631, 711],
#        [655, 711],
#        [680, 710],
#        [704, 709],
#        [728, 709],
#        [753, 708],
#        [777, 707],
#        [801, 707],
#        [825, 706]]) + center_offset


# centers_3 = np.array([[679, 712],
#        [654, 713],
#        [630, 713],
#        [605, 714],
#        [703, 711],
#        [727, 710],
#        [752, 709],
#        [776, 709],
#        [800, 708],
#        [824, 708]]) + center_offset

# 7/15
centers_3 = np.array([[608, 715],
       [632, 714],
       [657, 714],
       [681, 713],
       [705, 712],
       [729, 711],
       [754, 711],
       [778, 710],
       [802, 710],
       [826, 709]]) + center_offset

# centers_3 = np.array([[596, 931],
#        [546, 932],
#        [571, 931],
#        [621, 930],
#        [719, 928],
#        [670, 930],
#        [695, 929],
#        [646, 930],
#        [743, 928],
#        [768, 928]]) 

# # Test 20 tweezers :
# centers_3 =    np.array([[534, 931],
#         [583, 931],
#         [609, 931],
#         [596, 931],
#         [546, 931],
#         [571, 931],
#         [521, 931],
#         [558, 931],
#         [509, 931],
#         [621, 930],
#         [646, 930],
#         [484, 932],
#         [633, 930],
#         [497, 931],
#         [658, 930],
#         [670, 929],
#         [683, 929],
#         [695, 929],
#         [707, 929],
#         [719, 929]])


# centers_3 = centers_4 + np.array([0, 2*fast_kinetics_shift])

centers_4 = centers_3 - np.array([0, 2*fast_kinetics_shift])

rois_4 = [((x-rx, x+rx+1),(y-ry, y+ry+1)) for x,y in centers_4]
rois_3 = [((x-rx, x+rx+1),(y-ry, y+ry+1)) for x,y in centers_3]

n_tweezers = len(centers_4)

idxs = [idx for idx in range(n_tweezers)]

# group based on manifold
roiDict = {4: {roi_name: roi for roi_name, roi in zip(idxs, rois_4)},
        3: {roi_name: roi for roi_name, roi in zip(idxs, rois_3)}}
