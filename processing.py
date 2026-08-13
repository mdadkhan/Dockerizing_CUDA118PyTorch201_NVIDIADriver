# AI-CAC Project Code
# Creator: Raffi Hagopian MD
# Adapted Agatston Calculation functions from github msingh9/cs230-Coronary-Calcium-Scoring-/code/my_lib.py

import pandas as pd
import numpy as np
import pydicom
import os
import re 
import torch
from scipy import ndimage

def load_pydicom_slices_by_axial_cord(tuples): # tuples consist of list of tuples (dicom slice file path, dicom slice axial position) for a given study 
    tuples.sort(key=lambda x: x[1], reverse = False)   #sort list of tuples inplace, by axial coordinate, true axial cord order is similar to reverse file order
    slices = [pydicom.read_file(tup[0]) for tup in tuples]
    return slices

# Make sure CT voxel values are in HU -- could have different intercepts/slopes
def get_pixels_hu(slices):
    #print('Number of slices: %s' % len(slices))
    image = np.stack([s.pixel_array for s in slices], axis=2)    
    # Convert to int16 (from sometimes int16)
    image = image.astype(np.int16)
    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16), list(slices[0].PixelSpacing) + [slices[0].SliceThickness]

def get_object_agatston(calc_object, calc_pixel_count):
    object_max = np.max(calc_object)
    object_agatston = 0
    if 130 <= object_max < 200:
        object_agatston = calc_pixel_count * 1
    elif 200 <= object_max < 300:
        object_agatston = calc_pixel_count * 2
    elif 300 <= object_max < 400:
        object_agatston = calc_pixel_count * 3
    elif object_max >= 400:
        object_agatston = calc_pixel_count * 4
    return object_agatston

#input volume already must be in Hounsfeild Units -- #for_slice
def compute_agatston_for_vol(input_vol_hu, mask, voxel_dims, min_calc_object_pixels=3):
    if np.sum(mask) == 0:
        return 0
    # Divide Voxel_vol by 3 to normalize to standard CAC 3mm slice thickness - inputs may have variable slice thickness. Agatston formula is based on area with 3mm slices.
    voxel_vol = voxel_dims[0] * voxel_dims[1] * voxel_dims[2] / 3 
    agatston_score = 0
    labeled_mask, num_labels = ndimage.label(mask>0) 
    for calc_idx in range(1, num_labels + 1):
        label = np.zeros(mask.shape)
        label[labeled_mask == calc_idx] = 1
        calc_object = input_vol_hu * label
        calc_voxel_count = np.sum(label)
        # Remove small calcified objects.
        if calc_voxel_count <= min_calc_object_pixels:
            continue
        calc_volume = calc_voxel_count * voxel_vol
        object_agatston = round(get_object_agatston(calc_object, calc_volume))
        agatston_score += object_agatston
    return int(agatston_score)

def compute_agatston_for_batch(batch_vol_hu, batch_mask_vol, batch_voxel_dims):
    scores = []
    for i in range(0,batch_vol_hu.shape[0]):
        vol_hu = batch_vol_hu[i].squeeze().detach().numpy()
        mask_vol = batch_mask_vol[i].squeeze().detach().numpy()
        voxel_dims = batch_voxel_dims[i].numpy()
        score = compute_agatston_for_vol(vol_hu, mask_vol, voxel_dims, 1)
        scores.append(score)
    return scores

