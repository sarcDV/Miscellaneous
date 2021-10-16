from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from skimage.util.shape import view_as_blocks
from skimage.metrics import structural_similarity as ssim
from numpy.lib import stride_tricks
from get_rolling_window import rolling_window

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

import warnings
warnings.filterwarnings("ignore")
## ---------------------------------
def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6

def seg_2D_kmeans(invol, n_clusters):
  """slice-wise segmentation"""
  outvol = np.zeros((invol.shape))
  for ii in range(0, invol.shape[2]):
    if np.sum(invol[:,:,ii])>0.001:
      img = invol[:,:,ii]
      img = np.stack((img,)*3, axis=-1)
      image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D)
      clustered = kmeans.cluster_centers_[kmeans.labels_]
      clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
      outvol[:,:,ii] = clustered_3D[:,:,0]
    else:
      outvol[:,:,ii] = invol[:,:,ii]
    
  return outvol

def seg_3D_kmeans(invol, n_clusters): 
  vol_1d = invol.reshape(vol.shape[0]*vol.shape[1]*vol.shape[2], 1)
  vol_1d = np.stack((vol_1d,)*3, axis=1)
  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vol_1d.squeeze())
  clustered = kmeans.cluster_centers_[kmeans.labels_]
  clustered_3D = clustered.reshape(vol.shape[0], vol.shape[1], vol.shape[2], 3)
  clustered_3D = clustered_3D[:,:,:,0]

  return clustered_3D

def seg_3D_GMM(invol, n_clusters):
  vol_1d = invol.reshape(vol.shape[0]*vol.shape[1]*vol.shape[2], 1)
  vol_1d = np.stack((vol_1d,)*3, axis=1)
  gmm_model = GMM(n_components=n_clusters, covariance_type='full').fit(vol_1d.squeeze())  #tied works better than full
  gmm_labels = gmm_model.predict(vol_1d.squeeze())
  segmented = np.uint16(gmm_labels.reshape(vol.shape[0], vol.shape[1], vol.shape[2]))

  return segmented

def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X

def extract_patches_3D(volume, patch_shape, extraction_step,datype='float32'):
  patch_h, patch_w, patch_d = patch_shape[0], patch_shape[1], patch_shape[2]
  stride_h, stride_w, stride_d = extraction_step[0], extraction_step[1], extraction_step[2]
  img_h, img_w, img_d = volume.shape[0],volume.shape[1],volume.shape[2]
  N_patches_h = (img_h-patch_h)//stride_h+1
  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img = N_patches_h * N_patches_w * N_patches_d
  raw_patch_martrix = np.zeros((N_patches_img,patch_h,patch_w,patch_d),dtype=datype)
  k=0
  list_patches = []
  #iterator over all the patches
  for h in range((img_h-patch_h)//stride_h+1):
    for w in range((img_w-patch_w)//stride_w+1):
      for d in range((img_d-patch_d)//stride_d+1):
        # print(k,h*stride_h,(h*stride_h)+patch_h,\
        #         w*stride_w,(w*stride_w)+patch_w,\
        #         d*stride_d,(d*stride_d)+patch_d)
        list_patches.append([k,h*stride_h,(h*stride_h)+patch_h,\
                            w*stride_w,(w*stride_w)+patch_w,\
                            d*stride_d,(d*stride_d)+patch_d])
        raw_patch_martrix[k]=volume[h*stride_h:(h*stride_h)+patch_h,\
                                  w*stride_w:(w*stride_w)+patch_w,\
                                      d*stride_d:(d*stride_d)+patch_d]
        k+=1
  assert(k==N_patches_img)
  list_patches = np.asanyarray(list_patches)
  ## print(list_patches)

  return raw_patch_martrix, list_patches 


def find_and_crop(invol, atlasref):
  invol = invol/invol.max()
  ref = atlasref/atlasref.max()
  ## ----- extract - patches ---------
  ## patches_, list_patches = extract_patches_3D(invol, [64,96,48], [32,48,24])
  patches_, list_patches = extract_patches_3D(invol, [96,96,48], [48,48,24])
  patches_ = np.swapaxes(patches_, 0,3)
  patches_ = np.swapaxes(patches_, 0,2)
  patches_ = np.swapaxes(patches_, 0,1)
  list_val = np.zeros((patches_.shape[-1]))
  for ii in range(0,patches_.shape[-1]):
    tmpvol_ = patches_[:,:,:,ii]
    tmpvol_ = tmpvol_/tmpvol_.max()
    list_val[ii] = ssim(ref, tmpvol_, data_range=1)

  result = np.where(list_val == np.nanmax(list_val))  
  return patches_[:,:,:,result].squeeze(), result, list_patches

def registration_SyN(ref, invol, mask):
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
  warped_mask = mapping.transform(mask)

  return warped_moving, warped_mask

def patching_back(invol, selected_,index_, list_patches):
  empty_ = np.zeros((invol.shape))
  subvol = list_patches[index_[0]]
  empty_[subvol[0][1]:subvol[0][2], subvol[0][3]:subvol[0][4], subvol[0][5]:subvol[0][6] ] = selected_
  return empty_

## ----------------------------------------------------------
## ----------------------------------------------------------
## ----------------------------------------------------------
# vol  = nib.load('./nii-files/IXI002-Guys-0828-T1_struc_brain.nii.gz').get_fdata()
# patches = extract_patches_3D(vol, [64,96,48], [32,48,24])

# vol = nib.load('./atlases_tmp/mni_icbm152/mni_icbm152_pd_tal_nlin_sym_09c.nii.gz').get_fdata()
# ##vol = nib.load('./atlases_tmp/Masks3Dbin-new.nii.gz').get_fdata()
# patches, list_tmp = extract_patches_3D(vol, [96,96,64], [24,24,12])
# print(patches.shape)
# patches_ = np.swapaxes(patches, 0,3)
# patches_ = np.swapaxes(patches_, 0,2)
# patches_ = np.swapaxes(patches_, 0,1)
# patches_ = patches_[:,:,:,169]
# imgm = nib.Nifti1Image(patches_, np.eye(4))
# nib.save(imgm, 'patches_pd_969664_169.nii.gz')
"""
t1ref = nib.load('patch52at1atlas_.nii.gz')## .get_fdata()
t1sel = nib.load('patch_auto_selected.nii.gz')##.get_fdata()
## print(t1sel.shape)

static = t1sel.get_fdata().squeeze()
static_grid2world = t1sel.affine

moving = t1ref.get_fdata()
moving_grid2world = t1ref.affine

## ----------------------------------------------------------
identity = np.eye(4)
affine_map = AffineMap(identity,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
resampled = affine_map.transform(moving)

# regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
# regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
# regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")

c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)

transformed = c_of_mass.transform(moving)
# regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_com_0.png")
# regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_com_1.png")
# regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_com_2.png")
### -----------------------------------------------------
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

transform = TranslationTransform3D()
params0 = None
starting_affine = c_of_mass.affine
translation = affreg.optimize(static, moving, transform, params0,
                              static_grid2world, moving_grid2world,
                              starting_affine=starting_affine)
### -----------------------------------------------------
# transformed = translation.transform(moving)
# regtools.overlay_slices(static, transformed, None, 0,"Static", "Transformed", "transformed_trans_0.png")
# regtools.overlay_slices(static, transformed, None, 1,"Static", "Transformed", "transformed_trans_1.png")
# regtools.overlay_slices(static, transformed, None, 2,"Static", "Transformed", "transformed_trans_2.png")


transform = RigidTransform3D()
params0 = None
starting_affine = translation.affine
rigid = affreg.optimize(static, moving, transform, params0,
                        static_grid2world, moving_grid2world,
                        starting_affine=starting_affine)

transformed = rigid.transform(moving)
# regtools.overlay_slices(static, transformed, None, 0,"Static", "Transformed", "transformed_rigid_0.png")
# regtools.overlay_slices(static, transformed, None, 1,"Static", "Transformed", "transformed_rigid_1.png")
# regtools.overlay_slices(static, transformed, None, 2,"Static", "Transformed", "transformed_rigid_2.png")
### -----------------------------------------------------
transform = AffineTransform3D()
params0 = None
starting_affine = rigid.affine

affine = affreg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=starting_affine)
transformed = affine.transform(moving)
# regtools.overlay_slices(static, transformed, None, 0,"Static", "Transformed", "transformed_affine_0.png")
# regtools.overlay_slices(static, transformed, None, 1,"Static", "Transformed", "transformed_affine_1.png")
# regtools.overlay_slices(static, transformed, None, 2,"Static", "Transformed", "transformed_affine_2.png")

## ----------------------------------------------------------
metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
static_affine = static_grid2world
moving_affine = moving_grid2world
pre_align = affine.affine
mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)
warped_moving = mapping.transform(moving)
# regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving.png')
# warped_static = mapping.transform_inverse(static)
# regtools.overlay_slices(warped_static, moving, None, 1, 'Warped static', 'Moving', 'warped_static.png')

# mask_ = nib.load('patch52aMatlas_.nii.gz').get_fdata()
# warped_mask = mapping.transform(mask_.squeeze())

# imgm = nib.Nifti1Image(warped_mask, np.eye(4))
# nib.save(imgm, 'patch_Matlas_warped.nii.gz')
"""
# img = nib.Nifti1Image(warped_moving, np.eye(4))
# nib.save(img, 'patch_atlas_warped.nii.gz')
### -----------------------------------------------------
# selected_ = find_and_crop(vol, t1ref)
# img = nib.Nifti1Image(selected_, np.eye(4))
# nib.save(img, 'patch_auto_selected.nii.gz')
## run with python3 image-segmentation-v01.py
