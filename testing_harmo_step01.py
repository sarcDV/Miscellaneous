import os, glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import cv2
import torch

from skimage.io import imread
from skimage.color import rgb2gray

from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from skimage.util.shape import view_as_blocks
from skimage.metrics import structural_similarity as ssim
from numpy.lib import stride_tricks
# from get_rolling_window import rolling_window

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
import os.path
from dipy.viz import regtools
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

import torchio as tio
### ------------------------------------------
### ------------------------------------------
### ------------------------------------------
import warnings
warnings.filterwarnings("ignore")
## ---------------------------------
def registration_SyN(ref, invol): #, mask):
  ref = nib.Nifti1Image(ref, np.eye(4))
  invol = nib.Nifti1Image(invol, np.eye(4))
  ## mask = nib.Nifti1Image(mask, np.eye(4))  
  ## input as nifti not numpy arrays
  static = invol.get_fdata().squeeze()
  static_grid2world = invol.affine
  moving = ref.get_fdata()
  moving_grid2world = ref.affine
  ## ----------
  identity = np.eye(4)
  affine_map = AffineMap(identity,static.shape, static_grid2world,moving.shape, moving_grid2world)
  resampled = affine_map.transform(moving)
  ## ----------
  c_of_mass = transform_centers_of_mass(static, static_grid2world,moving, moving_grid2world)
  transformed = c_of_mass.transform(moving)
  ## ----------
  nbins = 32
  sampling_prop = None
  metric = MutualInformationMetric(nbins, sampling_prop)
  level_iters = [10000, 1000, 100]
  sigmas = [3.0, 1.0, 0.0]
  factors = [4, 2, 1]
  affreg = AffineRegistration(metric=metric,level_iters=level_iters,sigmas=sigmas,factors=factors)
  transform = TranslationTransform3D()
  params0 = None
  starting_affine = c_of_mass.affine
  translation = affreg.optimize(static, moving, transform, params0,static_grid2world, moving_grid2world,starting_affine=starting_affine)
  transformed = translation.transform(moving)
  ## -----------
  transform = RigidTransform3D()
  params0 = None
  starting_affine = translation.affine
  rigid = affreg.optimize(static, moving, transform, params0,static_grid2world, moving_grid2world,starting_affine=starting_affine)
  transformed = rigid.transform(moving)
  ## -----------
  transform = AffineTransform3D()
  params0 = None
  starting_affine = rigid.affine
  affine = affreg.optimize(static, moving, transform, params0,static_grid2world, moving_grid2world,starting_affine=starting_affine)
  transformed = affine.transform(moving)
  ## -----------
  metric = CCMetric(3)
  level_iters = [10, 10, 5]
  sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
  pre_align = affine.affine
  mapping = sdr.optimize(static, moving, static_grid2world, moving_grid2world, pre_align)
  warped_moving = mapping.transform(moving)
  # warped_mask = mapping.transform(mask)

  return warped_moving #, warped_mask


def pad3D(invol, max1d, max2d, max3d):
	aa = np.pad(invol, 
			(((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2),
			 ((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2),
			 ((max3d-invol.shape[2])//2, (max3d-invol.shape[2])//2)), 
			'constant')

	if aa.shape[0] == 255:
		aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')
	if aa.shape[1] == 255:
		aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')

	return aa

def hist_match(source, template):

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape)

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def minMAXnorm(invol):
	return (invol-invol.min())/(invol.max()-invol.min())

### ------------------------------------------
### ------------------------------------------
### ------------------------------------------
aa = nib.load('./T1_template_Repro/ay42_2330_sag_1mm.nii.gz')
bb = nib.load('./original_volumes/IXI002/IXI002-Guys-0828-T1.nii.gz')

aabet = nib.load('./T1_template_Repro/ay42_2330_sag_1mm.nii.gz').get_fdata()
bbbet = nib.load('./original_volumes/IXI002/IXI002-Guys-0828-T1_bet.nii.gz').get_fdata()

# print(aa.header['dim'][3], bb.header['dim'][3])
find_max_1d = np.array([aa.header['dim'][1], bb.header['dim'][1]]).max()
find_max_2d = np.array([aa.header['dim'][2], bb.header['dim'][2]]).max()
find_max_3d = np.array([aa.header['dim'][3], bb.header['dim'][3]]).max()

aa = aa.get_fdata()
bb = bb.get_fdata()

aa = minMAXnorm(pad3D(aa, 256, 256, find_max_3d))
bb = minMAXnorm(pad3D(bb, 256, 256, find_max_3d))

aabet = minMAXnorm(pad3D(aabet, 256, 256, find_max_3d))
bbbet = minMAXnorm(pad3D(bbbet, 256, 256, find_max_3d))

## full head warping:
# warped_= registration_SyN(ref=aa, invol=bb)
# img_t1w = nib.Nifti1Image(warped_, np.eye(4))
# nib.save(img_t1w, 'test_warped_.nii.gz')

## brain warping:
# warpedbet_= registration_SyN(ref=aabet, invol=bbbet)
# img_t1w = nib.Nifti1Image(warpedbet_, np.eye(4))
# nib.save(img_t1w, 'test_warpedbet_.nii.gz')

warped_ = nib.load('test_warped_.nii.gz').get_fdata()
matched_ = hist_match(source=bb, template=warped_)

warpedbet_ = nib.load('test_warpedbet_.nii.gz').get_fdata()
matchedbet_ = minMAXnorm(hist_match(source=bbbet, template=warpedbet_))

maskbin_ = (bbbet>0)*1.0
invmaskbin_ = 1-maskbin_
# matchedbrain_ = hist_match(source=bb*maskbin_, template=warped_*maskbin_)
# matchedskull_ = hist_match(source=bb*invmaskbin_, template=warped_*invmaskbin_)

compositematched_ = (maskbin_*matched_) + (matchedbet_) +(invmaskbin_*bb)
# compositematched_ = matchedskull_ + (matchedbrain_)
# compositematched_ = (compositematched_>0.05)*compositematched_

plt.figure(figsize=(8,8))

plt.subplot(231)
plt.imshow(np.rot90(aa[:,:,137]), cmap='gray')
plt.axis(False)
plt.title('Target Repro 3D vol')

plt.subplot(232)
plt.imshow(np.rot90(bb[:,:,bb.shape[2]//2]), cmap='gray')
plt.axis(False)
plt.title('Source image IXI')

plt.subplot(233)
plt.imshow(np.rot90(matched_[:,:,bb.shape[2]//2]), cmap='gray')
plt.axis(False)
plt.title('Naive histogram matching')

plt.subplot(234)
plt.imshow(np.rot90(bbbet[:,:,aa.shape[2]//2]), cmap='gray')
plt.axis(False)
plt.title('Mask source image IXI')

plt.subplot(235)
plt.imshow(np.rot90(warped_[:,:,bb.shape[2]//2]), cmap='gray')
plt.axis(False)
plt.title('Warped image to source')

plt.subplot(236)
# plt.imshow(np.rot90(matchedbet_[:,:,bb.shape[2]//2]), cmap='gray')
# plt.imshow(np.rot90(warped_[:,:,229]), cmap='gray')
plt.imshow(np.rot90(compositematched_[:,:,bb.shape[2]//2]), cmap='gray')
plt.axis(False)
plt.title('Composite \nharmonization process')

plt.tight_layout()
plt.show()


"""
import pyrtools as pt
import pywt
# haarLo = pt.named_filter('haar')
# haarHi = pt.pyramids.WaveletPyramid._modulate_flip(haarLo)

# # Load image
original = np.rot90(bb[:,:,bb.shape[2]//2])

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    # ax.set_xticks([])
    # ax.set_yticks([])

fig.tight_layout()
# plt.show()

coeffs = pywt.dwt2(np.rot90(bb[:,:,bb.shape[2]//2]), 'haar')
cA, (cH, cV, cD) = coeffs

plt.figure(figsize=(12, 3))
plt.subplot(141)
plt.imshow(cA, cmap='gray')
plt.subplot(142)
plt.imshow(cH, cmap='gray')
plt.subplot(143)
plt.imshow(cV, cmap='gray')
plt.subplot(144)
plt.imshow(cD, cmap='gray')
plt.tight_layout()
# plt.show()


plt.figure(figsize=(12, 3))
plt.subplot(141)
plt.imshow(cA, cmap='gray')
plt.subplot(142)
plt.imshow(cH*cA, cmap='gray')
plt.subplot(143)
plt.imshow(cV*cA, cmap='gray')
plt.subplot(144)
plt.imshow(cD*cA, cmap='gray')
plt.tight_layout()
plt.show()

"""