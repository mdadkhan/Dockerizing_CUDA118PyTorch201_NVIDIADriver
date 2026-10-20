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

# AI-CAC was trained and validated on 2.5-5 mm axial slices. Series from 0.5 mm up to
# 2.5 mm are reconstructed to 3 mm before inference; anything outside 0.5-5 mm is excluded.
VALIDATED_SLICE_THICKNESS_MM = (2.5, 5.0)
MIN_ACCEPTED_SLICE_THICKNESS_MM = 0.5
RECONSTRUCTED_SLICE_THICKNESS_MM = 3.0

def is_validated_slice_thickness(thickness_mm):
    low, high = VALIDATED_SLICE_THICKNESS_MM
    return low <= thickness_mm <= high

def reconstruct_thin_slices(volume, voxel_resolution):
    # Thin slices (< 2.5 mm) are interpolated along z to 3 mm (cubic), and the z voxel size
    # becomes 3 mm so Agatston scoring uses the reconstructed thickness.
    current_thickness = float(voxel_resolution[2])
    if current_thickness >= VALIDATED_SLICE_THICKNESS_MM[0]:
        return volume, voxel_resolution
    z_zoom_factor = current_thickness / RECONSTRUCTED_SLICE_THICKNESS_MM
    volume = ndimage.zoom(volume, (1, 1, z_zoom_factor), order=3)
    voxel_resolution = [float(voxel_resolution[0]), float(voxel_resolution[1]), RECONSTRUCTED_SLICE_THICKNESS_MM]
    return volume, voxel_resolution

def slice_thickness_report(original_thickness_mm):
    """Fields describing how a series' slice thickness was handled, for logs and the API response."""
    original = float(original_thickness_mm)
    if is_validated_slice_thickness(original):
        return {
            'original_slice_thickness_mm': original,
            'reconstructed_to_mm': None,
            'within_validated_range': True,
            'warning': None,
        }
    low, high = VALIDATED_SLICE_THICKNESS_MM
    warning = (
        f'AI-CAC was validated on {low:g}\u2013{high:g} mm slice thickness. '
        f'This series ({original:g} mm) was reconstructed to {RECONSTRUCTED_SLICE_THICKNESS_MM:g} mm. '
        'Consider further validation.'
    )
    return {
        'original_slice_thickness_mm': original,
        'reconstructed_to_mm': RECONSTRUCTED_SLICE_THICKNESS_MM,
        'within_validated_range': False,
        'warning': warning,
    }
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
def compute_agatston_for_vol(input_vol_hu, mask, voxel_dims, min_calc_object_pixels=1): # default 1 matches compute_agatston_for_batch and the values used in the paper
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

