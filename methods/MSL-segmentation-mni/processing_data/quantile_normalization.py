import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
    image2 is the reference
"""
def quantile_normalization(image1, image2):
    n1 = len(image1)
    n2 = len(image2)
    r1 = stats.rankdata(image1, method='average')
    r2 = r1*n2/n1
    sorted_image2 = np.sort(image2)
    image1_normalized = np.interp(r2, 1+np.arange(n2), sorted_image2)

    return image1_normalized

"""
    image: numpy array
    brain_mask: brain mask for image
    ref_image: FLAIR template
    ref_brain_mask: brain mask for ref_image
"""
def quantile_normalization_3D_image(image, ref_image, brain_mask, ref_brain_mask, plot_name):

    image_flat = image.ravel()
    ref_image_flat = ref_image.ravel()
    brain_mask_flat = brain_mask.ravel()
    ref_brain_mask_flat = ref_brain_mask.ravel()

    I = np.where(brain_mask_flat == 1 )
    val = image_flat[I]

    Ir = np.where(ref_brain_mask_flat == 1)
    val_ref = ref_image_flat[Ir]

    val_norm = quantile_normalization(val, val_ref)
    image_norm_flat = np.zeros(len(image_flat))
    image_norm_flat[I] = val_norm
    image_norm = image_norm_flat.reshape(image.shape)

    return image_norm

