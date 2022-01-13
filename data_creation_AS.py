import os
import random
from glob import glob
import numpy as np
import nibabel as nib
import math
import multiprocessing.dummy as multiprocessing
import random
from collections import defaultdict
from typing import List
import SimpleITK as sitk
import torch
import torchio as tio
from torchio.data.io import read_image
from scipy.ndimage import affine_transform
from torchio.transforms import Motion, RandomMotion
from torchio.transforms.interpolation import Interpolation

def create_subjectlist(path):
    files = glob(path+"/**/*.nii", recursive=True) + glob(path+"/**/*.nii.gz", recursive=True)
    subjects = []
    for file in files:
        subjects.append(os.path.basename(file))

    return subjects


class RealityMotion():
    def __init__(self, n_threads = 4, mu = 0, sigma = 0.1, random_sigma=True):
        self.n_threads = n_threads
        self.mu = mu
        self.sigma = sigma
        self.sigma_limit = sigma
        self.random_sigma = random_sigma

    def __perform_singlePE(self, idx):
        rot_x = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        rot_y = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        rot_z = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        tran_x = 0 # int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))
        tran_y = 0 # int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))
        tran_z = 0 # int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))
        ## print(rot_x, rot_y, rot_z, tran_x, tran_y, tran_z)
        temp_vol = self.__rot_tran_3d(self.in_vol, rot_x, rot_y, rot_z, tran_x, tran_y, tran_z)
        temp_k = np.fft.fftn(temp_vol)
        for slc in range(self.in_vol.shape[2]):
            self.out_k[idx,:,slc]=temp_k[idx,:,slc] 

    def __call__(self, vol):
        if self.random_sigma:
            self.sigma = random.uniform(0, self.sigma_limit)
        shape = vol.shape
        device = vol.device
        self.in_vol = vol.squeeze().cpu().numpy()
        self.in_vol = self.in_vol/self.in_vol.max()
        self.out_k = np.zeros((self.in_vol.shape)) + 0j
        if self.n_threads > 0:
        	pool = multiprocessing.Pool(self.n_threads)
        	pool.map(self.__perform_singlePE, range(self.in_vol.shape[0]))
        else:
            for idx in range(self.in_vol.shape[0]):
            	self.__perform_singlePE(idx)
        vol = np.abs(np.fft.ifftn(self.out_k)) 
        vol = torch.from_numpy(vol).view(shape).to(device)
        del self.in_vol, self.out_k
        return vol

    def __x_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[1, 0, 0],
                        [0, cos_t, -sin_t],
                        [0, sin_t, cos_t]])

    def __y_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[cos_t, 0, sin_t],
                        [0, 1, 0],
                        [-sin_t, 0, cos_t]])

    def __z_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[cos_t, -sin_t, 0],
                        [sin_t, cos_t, 0],
                        [0, 0, 1]])

    def __rot_tran_3d(self, J, rot_x, rot_y, rot_z, tran_x, tran_y, tran_z):  
        M = self.__x_rotmat(rot_x) * self.__y_rotmat(rot_y) * self.__z_rotmat(rot_z) 
        translation = ([tran_x, tran_y, tran_z])
        K = affine_transform(J, M, translation, order=1)
        return K/(K.max()+1e-16)

class MotionCorrupter():
    def __init__(self,  norm_mode=0, noise_dir=2, mu=0, sigma=0.1, random_sigma=False, n_threads=4):
        
        self.norm_mode = norm_mode #0: No Norm, 1: Divide by Max, 2: MinMax
        self.noise_dir = noise_dir #0, 1 or 2 - which direction the motion is generated, only for custom random
        self.mu = mu #Only for Reality Motion
        self.sigma = sigma #Only for Reality Motion
        self.random_sigma = random_sigma  #Only for Reality Motion - to randomise the sigma value, treating the provided sigma as upper limit and 0 as lower
        self.n_threads = n_threads #Only for Reality Motion - to apply motion for each thread encoding line parallel, max thread controlled by this. Set to 0 to perform serially.

        self.corrupter = RealityMotion(n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=random_sigma)

    def perform(self, vol):
        vol = vol.float()
        transformed = self.corrupter(vol)
        if self.norm_mode==1:
            vol = vol/vol.max()
            transformed = transformed/transformed.max()
        elif self.norm_mode==2:
            vol = (vol-vol.min())/(vol.max()-vol.min())
            transformed = (transformed-transformed.min())/(transformed.max()-transformed.min())
        return torch.cat([vol,transformed], 0)



if __name__ == '__main__':
    # path_ = "./data-toy-for-debugging/train/output"
    path_ = "./data-toy-for-debugging/val/output"
    # path_ = "./output/train"
    n_threads = 48
    # myfile = open('train-toy.txt', 'w')
    myfile = open('val-toy.txt', 'w')
    subjects_dataset = create_subjectlist(path_)
    ## numsub_ = len(subjects_dataset)
    for ii in range(0,6):
        for s in subjects_dataset:
            name_ = s[0:len(s)-7]
            file = path_+"/"+s
            mu = 0.0      ## 0.0 - 1.0
            sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))  ## 0.05 - 0.1 np.random.uniform(low=0.01, high=0.2, size=(50,))
            subject_proc= tio.Subject(im=tio.ScalarImage(file), filename=os.path.basename(file))
            moco = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=False)
            transforms = [tio.Lambda(moco.perform, p = 1)]
            transform = tio.Compose(transforms)
            ## print('...corrupting ', str(ii+1), "... ", s, "  thread in use ", n_threads, " sigma =  ", str(sigma) )
            gt, inp = transform(subject_proc['im'][tio.DATA])
            data_ =  np.float32(np.abs(inp.cpu().numpy().squeeze()))
            # nib.save(nib.Nifti1Image(data_, None), "./temp/"+str(name_)+"-"+"{:03d}".format(ii+1)+'.nii.gz')
            # nib.save(nib.Nifti1Image(data_, None), "./input/train/"+str(name_)+"-"+"{:03d}".format(ii+1)+'.nii.gz')
            # nib.save(nib.Nifti1Image(data_, None), "./data-toy-for-debugging/train/input/"+str(name_)+"-"+"{:03d}".format(ii+1)+'.nii.gz')
            nib.save(nib.Nifti1Image(data_, None), "./data-toy-for-debugging/val/input/"+str(name_)+"-"+"{:03d}".format(ii+1)+'.nii.gz')
            print('corruption ', str(ii+1), " done for: ", s, "  thread in use ", n_threads, " sigma =  ", str(sigma))
            ##print('...corruption ', str(ii+1), " done for ... ", s)
            myfile.write(str(name_)+"-"+"{:03d}".format(ii+1)+'   '+ str(sigma[0])+'\n')
