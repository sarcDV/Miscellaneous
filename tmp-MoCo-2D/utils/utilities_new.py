import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import nibabel as nib
import random
from numpy.lib.arraypad import pad
import torch
from statistics import median
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from skimage import exposure
from progressbar import ProgressBar
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)
import torchvision.utils as vutils
import torchvision
import torchio
import time
import multiprocessing.dummy as multiprocessing
from tqdm import tqdm

#class torchio.transforms.RandomMotion(degrees: float = 10, translation: float = 10, num_transforms: int = 2, image_interpolation: str = 'linear', **kwargs)
#class torchio.transforms.RandomGhosting(num_ghosts: Union[int, Tuple[int, int]] = (4, 10), axes: Union[int, Tuple[int, ...]] = (0, 1, 2), intensity: Union[float, Tuple[float, float]] = (0.5, 1), restore: float = 0.02, **kwargs)
###########################################################################
##### Auxiliaries  ########################################################
###########################################################################

def pad3D(invol, max1d, max2d, max3d):
    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2),
             ((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2),
             ((max3d-invol.shape[2])//2, (max3d-invol.shape[2])//2)), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')
    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')
    if aa.shape[2] == (int(max3d)-1):
        aa = np.pad(aa, ((0,0),(0,0),(1,0)), 'constant')

    return aa

def pad2D(invol, max1d, max2d):
    if (invol.shape[0] % 2) != 0:  # if shape[0] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((1,invol.shape[1]))), axis=0)
        
    if (invol.shape[1] % 2) != 0:  # if shape[1] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((invol.shape[0],1))), axis=1)

    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2),
             ((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2)), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')
    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')

    return aa

def pad2DD1(invol, max1d):
    if (invol.shape[0] % 2) != 0:  # if shape[0] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((1,invol.shape[1]))), axis=0)

    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2 ) ), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')

    return aa

def pad2DD2(invol, max2d):
    if (invol.shape[1] % 2) != 0:  # if shape[1] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((invol.shape[0],1))), axis=1)

    aa = np.pad(invol, 
            (((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2)), 
            'constant')

    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')

    return aa

def randomCrop(img, cor, width, height): 
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    cor = cor[y:y+height, x:x+width]
    return img, cor

def randomCropInput(img, width, height): 
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    
    return img

def CropCentralCore(img, width, height):
    x = img.shape[1]//2
    y = img.shape[0]//2
    img = img[y-height//2:y+height//2, x-width//2:x+width//2]
    return img

def PadAndResize(img, width, height):
    dsize = (width, height)
    if img.shape[0]==img.shape[1]:
        img  = cv2.resize(img, dsize)
    elif img.shape[0]>img.shape[1]:
        img = pad2DD2(img, img.shape[0])
        img = cv2.resize(img, dsize)
    elif img.shape[0]<img.shape[1]:
        img = pad2DD1(img, img.shape[1])
        img = cv2.resize(img, dsize)
    return img

###########################################################################
##### Contrast Augmentation  ##############################################
###########################################################################

def randomContrastAug(img):
    expo_selection = np.random.randint(0,5,1)
    if expo_selection[0] == 0:
        adjimg = exposure.adjust_gamma(img, np.random.uniform(0.75, 1.75, 1)[0])
    elif expo_selection[0] == 1:
        adjimg = exposure.equalize_adapthist(img, 
											kernel_size=int(np.random.randint(25, high=100, size=(1))[0]), #21, 
											clip_limit=0.01, 
											nbins=512)
    elif expo_selection[0] == 2:
        adjimg = exposure.adjust_sigmoid(img, 
	 								   cutoff=np.random.uniform(0.01, 0.75, 1)[0], #0.5, 
	  								   gain=int(np.random.randint(1, high=4, size=(1))[0]), #10, 
	  								   inv=False)
    elif expo_selection[0] == 3:
        adjimg = np.abs(exposure.adjust_log(img, np.random.uniform(-0.5, 0.5, 1)[0]))
    else:
        adjimg = img

    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()-adjimg.min())

    return adjimg #, expo_selection[0]

###########################################################################
##### Cut noise level  ####################################################
###########################################################################

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()-adjimg.min())

    return adjimg 

###########################################################################
##### Motion  #############################################################
###########################################################################

class Motion2DOld():
    def __init__(self, sigma_range=(2.0, 3.0), n_threads=10):
        self.sigma_range = sigma_range
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        rot = self.sigma*random.randint(-1,1)
        img_aux = ndimage.rotate(self.img, rot, reshape=False)
        # rot = np.random.uniform(self.mu, self.sigma, 1)*random.randint(-1,1)
        # rot = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        # img_aux = ndimage.rotate(self.img, rot[0], reshape=False)
        img_h = np.fft.fft2(img_aux)
        if self.axis_selection == 0:
            self.aux[:,idx]=img_h[:,idx]
        else:
            self.aux[idx,:]=img_h[idx,:]

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]
        self.mu=0
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]))
        else:
            for idx in range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]):
                self.__perform_singlePE(idx)
        cor =np.abs(np.fft.ifft2(self.aux)) 
        del self.img, self.aux, self.axis_selection, self.mu, self.sigma
        return cor/(cor.max()+1e-16)

class Motion2D():
    def __init__(self, sigma_range=(0.10, 2.5), restore_original=5e-2, n_threads=20):
        self.sigma_range = sigma_range
        self.restore_original = restore_original
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        img_aux = ndimage.rotate(self.img, self.random_rots[idx], reshape=False)
        img_h = np.fft.fft2(img_aux)            
        if self.axis_selection == 0:
            self.aux[:,self.portion[idx]]=img_h[:,self.portion[idx]]  
        else:
            self.aux[self.portion[idx],:]=img_h[self.portion[idx],:]  

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]

        if self.axis_selection == 0:
            dim = 1
        else:
            dim = 0

        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[dim]//n_), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[dim]//n_), 
                                                        high=int(img.shape[dim]), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int))) 
            self.portion = np.concatenate((portiona, portionb))  
        else:
            self.portion = np.sort(np.unique(np.random.randint(low=int(img.shape[dim]//2)-int(img.shape[dim]//n_+1), 
                                                     high=int(img.shape[dim]//2)+int(img.shape[dim]//n_+1), 
                                                     size=int(img.shape[dim]//n_+1), dtype=int)))
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        self.random_rots = self.sigma * np.random.randint(-1,1,len(self.portion))
        #  self.random_rots = np.random.randint(-4,4,len(self.portion))

        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(len(self.portion)-1))
        else:
            for idx in range(len(self.portion)-1):
                self.__perform_singlePE(idx)     
        cor =np.abs(np.fft.ifft2(self.aux)) # + self.restore_original *img

        del self.img, self.aux, self.axis_selection, self.portion, self.random_rots
        return cor/(cor.max()+1e-16)
###########################################################################
##### slice selection  ####################################################
###########################################################################

def select_slice_orientation(test, orientation):
    if orientation == 3:
        if test.shape[2] > (test.shape[0]//2):
            rnd_orient = np.random.randint(0,3,1)[0]
            # print(rnd_orient)
            if rnd_orient == 0:
                rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)
                
                img = (test[:,rndslice_[0],:])
            elif rnd_orient == 1:
                rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
                                        
                img = np.rot90(test[:,:,rndslice_[0]])    
            else:
                rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
                img = np.flipud(test[rndslice_[0],:,:])
        else:
            rnd_orient = 1
            rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)                           
            img = np.rot90(test[:,:,rndslice_[0]]) 
    
    elif orientation == 0:
        rnd_orient = 0 
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
    elif orientation == 1:
        rnd_orient = 1
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)       
        img = (test[:,rndslice_[0],:])
    elif orientation == 2: 
        rnd_orient = 2
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
        img = np.flipud(test[rndslice_[0],:,:])

    img = (img-img.min())/(img.max()-img.min())
            
    return img, rndslice_, rnd_orient

def select_slice_orientation_both(test, cor,orientation):

    if orientation == 3:
        if test.shape[2] > (test.shape[0]//2):
            rnd_orient = np.random.randint(0,3,1)[0]
            # print(rnd_orient)
            if rnd_orient == 0:
                rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)
                
                img = (test[:,rndslice_[0],:])
                imgcor = (cor[:,rndslice_[0],:])
            elif rnd_orient == 1:
                rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
                                        
                img = np.rot90(test[:,:,rndslice_[0]]) 
                imgcor = np.rot90(cor[:,:,rndslice_[0]])   
            else:
                rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
                img = np.flipud(test[rndslice_[0],:,:])
                imgcor = np.flipud(cor[rndslice_[0],:,:])
        else:
            rnd_orient = 1
            rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)                           
            img = np.rot90(test[:,:,rndslice_[0]]) 
            imgcor = np.rot90(cor[:,:,rndslice_[0]]) 
    
    elif orientation == 0:
        rnd_orient = 0 
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
        imgcor = np.rot90(cor[:,:,rndslice_[0]]) 
    elif orientation == 1:
        rnd_orient = 1
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)       
        img = (test[:,rndslice_[0],:])
        imgcor = (cor[:,rndslice_[0],:])
    elif orientation == 2: 
        rnd_orient = 2
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
        img = np.flipud(test[rndslice_[0],:,:])
        imgcor = np.flipud(cor[rndslice_[0],:,:])

    img = (img-img.min())/(img.max()-img.min())
    imgcor = (imgcor-imgcor.min())/(imgcor.max()-imgcor.min())       
    return img, imgcor, rndslice_, rnd_orient

###########################################################################
##### k-space   ###########################################################
###########################################################################

def kspace_cropcenter(intensor, coresize_):
    inpksp = torch.zeros(1,2,coresize_, coresize_)
    tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    ## normalize:
    tmpinp = tmpinp/torch.abs(tmpinp).max()
    crop_ = torchvision.transforms.CenterCrop(coresize_)
    tmpinp = crop_(tmpinp)
    inpksp[:,0,:,:]=torch.real(tmpinp)
    inpksp[:,1,:,:]=torch.imag(tmpinp)
    return inpksp.float()

def kspace_clearcore(intensor, sizeimg_, coresize_):
    inpksp = torch.zeros(1,2, sizeimg_, sizeimg_)
    tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    ## normalize:
    tmpinp = tmpinp/torch.abs(tmpinp).max()
    inpksp[:,0,:,:] = torch.real(tmpinp)
    inpksp[:,1,:,:] = torch.imag(tmpinp)
    inpksp[:,0,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=0
    inpksp[:,1,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=0
    return inpksp.float()

def from_kspacecore_toImg(intensor, sizeimg, coresize):
    ## pad tensor
    m = torch.nn.ZeroPad2d(sizeimg//2-coresize//2)
    padded_ = m(intensor)
    realt = padded_[:,0,:,:].float()
    imagt = padded_[:,1,:,:].float()
    z = torch.complex(realt,imagt)
    img = torch.abs(torch.fft.ifft2(torch.squeeze(z)))
    img = img/torch.abs(img).max()
    return torch.unsqueeze(torch.unsqueeze(img,0),0).float()

def from_kspace_toImg(intensor, complex_=False):
    if complex_ == True:
        img = torch.abs(torch.fft.ifft2(torch.squeeze(intensor)))   
    else:
        realt = intensor[:,0,:,:].float()
        imagt = intensor[:,1,:,:].float()
        z = torch.complex(realt,imagt)
        img = torch.abs(torch.fft.ifft2(torch.squeeze(z)))
        
    img = img/torch.abs(img).max()
    return torch.unsqueeze(torch.unsqueeze(img,0),0).float()

def split_dual_inner_outer(intensor, sizeimg_, coresize_):
    imgdual = torch.zeros(1,2,sizeimg_, sizeimg_)
    tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    ## inner core to zero:
    tmpinp[sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=torch.zeros(coresize_,coresize_,dtype=torch.cfloat)
    img = torch.abs(torch.fft.ifft2(torch.squeeze(tmpinp)))
    img = img/torch.abs(img).max()
    imgdual[0,0,...]=img
    ## outer area to zero:
    tmpinpo = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    crop_ = torchvision.transforms.CenterCrop(coresize_)
    tmpinpo = crop_(tmpinpo)
    m = torch.nn.ZeroPad2d(sizeimg_//2-coresize_//2)
    padded_ = m(tmpinpo)
    img_ = torch.abs(torch.fft.ifft2(torch.squeeze(padded_)))
    img_ = img_/torch.abs(img_).max()
    imgdual[0,1,...]=img_

    return imgdual.float()

def split_three_ch(intensor, sizeimg_, coresize_):
    imgdual = torch.zeros(1,3,sizeimg_, sizeimg_)
    tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    ## inner core to zero:
    tmpinp[sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=torch.zeros(coresize_,coresize_,dtype=torch.cfloat)
    img = torch.abs(torch.fft.ifft2(torch.squeeze(tmpinp)))
    img = img/torch.abs(img).max()
    imgdual[0,0,...]=img
    ## outer area to zero:
    tmpinpo = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    crop_ = torchvision.transforms.CenterCrop(coresize_)
    tmpinpo = crop_(tmpinpo)
    m = torch.nn.ZeroPad2d(sizeimg_//2-coresize_//2)
    padded_ = m(tmpinpo)
    img_ = torch.abs(torch.fft.ifft2(torch.squeeze(padded_)))
    img_ = img_/torch.abs(img_).max()
    imgdual[0,1,...]=img_
    imgdual[0,2,...]=intensor

    return imgdual.float()

def from_Img_to_kspace(intensor, sizeimg_, coresize_, inner=True, outer=False, fullspc=False):
    if inner == True:
        tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
        ## normalize:
        tmpinp = tmpinp/torch.abs(tmpinp).max()
        crop_ = torchvision.transforms.CenterCrop(coresize_)
        tmpinp = crop_(tmpinp)
    elif outer == True:
        tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
        ## normalize:
        tmpinp = tmpinp/torch.abs(tmpinp).max()
        tmpinp[:,:,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=torch.zeros(coresize_,coresize_,dtype=torch.cfloat)
    
    elif fullspc == True:
        tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
        ## normalize:
        tmpinp = tmpinp/torch.abs(tmpinp).max()

    return torch.unsqueeze(torch.unsqueeze(tmpinp,0),0)

def FourChannelComplexUncombined(intensor, sizeimg_, coresize_):
    tmpinp = torch.fft.fftshift(torch.fft.fft2(torch.squeeze(intensor)))
    ## normalize:
    tmpinp = tmpinp/torch.abs(tmpinp).max()

    tmpinit = torch.zeros(4, sizeimg_, sizeimg_)
    ## core
    tmpinit[0,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2] = torch.real(tmpinp[sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
                                                                                   sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2])
    tmpinit[1,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2] = torch.imag(tmpinp[sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
                                                                                   sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2])

    ## outer
    tmpinp_ = tmpinp
    tmpinp_[sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2] = 0.0+0.0j
    tmpinit[2,...]=torch.real(tmpinp_)# -torch.tensor(tmpinit[0,...])
    tmpinit[3,...]=torch.imag(tmpinp_)# -torch.tensor(tmpinit[1,...])
    # tmpinit[2:3,
    #         sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
    #         sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=0.0

    return torch.unsqueeze(tmpinit, 0)

def Recombine4Ch(coreksp, outksp, sizeimg_, coresize_):
    # print(coreksp.size(), outksp.size(), sizeimg_, coresize_)
    tmpinit = torch.zeros(1,4, sizeimg_, sizeimg_)
    tmpinit[0,0,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2] = coreksp[0,0,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
                                                                        sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2].float()
    tmpinit[0,1,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
            sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2] = coreksp[0,1,sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
                                                                        sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2].float() 
    
    tmpinit[0,2,...] = outksp[0,0,...].float()
    tmpinit[0,3,...] = outksp[0,1,...].float()
    # tmpinit[0,2:,
    #         sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2,
    #         sizeimg_//2-coresize_//2:sizeimg_//2+coresize_//2]=0.0
    realt = (tmpinit[0,0,...]+tmpinit[0,2,...]).float()
    imagt = (tmpinit[0,1,...]+tmpinit[0,3,...]).float()
    z = torch.complex(torch.squeeze(realt),torch.squeeze(imagt))
    img = torch.abs(torch.fft.ifft2(torch.squeeze(z)))
    img = img/torch.abs(img).max()                                                            

    return torch.unsqueeze(torch.unsqueeze(img,0),0).float()

###########################################################################
##### Motion Corruption Class  ############################################
###########################################################################
class MoCoDataset():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=0,
                       sigma_range=(1.5, 2.5),
                       level_noise=0.07,
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.transform = transform
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.cter = Motion2DOld(n_threads=20)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))

        for ii in range(0, self.patches):
            img, slice, orient = select_slice_orientation(image_in_, self.orientation)
            img = randomContrastAug(cutNoise(img, self.level_noise))
            cor = self.cter(img)
            if (img.shape[0]<=size_) and (img.shape[1]<=size_):
                img = pad2D(img, size_, size_)
                cor = pad2D(cor, size_, size_)       
            elif (img.shape[0]>size_) and (img.shape[1]>size_):
                img, cor = randomCrop(img, cor, width=size_, height=size_) 
                # img = randomCropInput(img, width=size_, height=size_)
            elif (img.shape[0]<=size_) and (img.shape[1]>size_):
                img = pad2DD1(img, size_) 
                cor = pad2DD1(cor, size_)
                img, cor = randomCrop(img, cor, width=size_, height=size_) 
                # img = pad2DD1(img, size_) 
                # img = randomCropInput(img,  width=size_, height=size_)
            elif (img.shape[0]>size_) and (img.shape[1]<=size_):
                img = pad2DD2(img, size_) 
                cor = pad2DD2(cor, size_) 
                img, cor = randomCrop(img, cor, width=size_, height=size_)
                # img = pad2DD2(img, size_) 
                # img = randomCropInput(img,  width=size_, height=size_)
            else:
                print(img.shape, 'ciao')
            ### ------- 
            # img = randomContrastAug(img)
            # cor = generate_motion_2d(img)
            # cor = generate_motion_2d_old(img)
            # cter = Motion2DOld(n_threads=20)
            # cor = self.cter(img)
            ssimtmp = structural_similarity(img, cor, data_range=1)

            stackcor[ii,:,:]= cor
            stackimg[ii,:,:]= img
            stackssim[ii,0] = ssimtmp
        
        if self.transform:
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            # img = self.transform(img)
            # cor = self.transform(cor)
            # ssimtmp = self.transform(ssimtmp)
        
        return stackcor, stackimg, stackssim
        # return cor, img, ssimtmp

class MoCoDataset2DNoPatchesBsOne():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                orientation=3,
                sigma_range=(0.01, 2.5),
                level_noise=0.07,
                transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.orientation = orientation
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.transform = transform
        self.cter = Motion2DOld(n_threads=10)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        img, slice, orient = select_slice_orientation(image_in_, self.orientation)
        img = randomContrastAug(cutNoise(img, self.level_noise))
        cor = self.cter(img)
        ssimtmp = structural_similarity(img, cor, data_range=1)
        ### ------        
        if self.transform:
            img = self.transform(img)
            cor = self.transform(cor)
            ssimtmp = self.transform(ssimtmp)
        
        return cor, img, ssimtmp

class MoCoDatasetRegression():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=3,
                       sigma_range=(2.0, 3.0),
                       level_noise=0.025,
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.transform = transform
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.cter = Motion2DOld(n_threads=10, sigma_range=self.sigma_range)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))

        for ii in range(0, self.patches):
            img, slice, orient = select_slice_orientation(image_in_, self.orientation)
            img = randomContrastAug(cutNoise(img, self.level_noise))
            
            if (img.shape[0]<=size_) and (img.shape[1]<=size_):
                img = pad2D(img, size_, size_)       
            elif (img.shape[0]>size_) and (img.shape[1]>size_):
                img = PadAndResize(img, width=size_, height=size_)
            elif (img.shape[0]<=size_) and (img.shape[1]>size_):
                img = PadAndResize(img, width=size_, height=size_)
            elif (img.shape[0]>size_) and (img.shape[1]<=size_):
                img = PadAndResize(img, width=size_, height=size_)
            else:
                print(img.shape, 'ciao')
            ### ------- 
            cor = self.cter(img)
            ssimtmp = structural_similarity(img, cor, data_range=1)

            stackcor[ii,:,:]= cor
            stackimg[ii,:,:]= img
            stackssim[ii,0] = ssimtmp
        
        if self.transform:
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            
        return stackcor, stackimg, stackssim

class MoCoDatasetTorchIO():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=3,
                       level_noise=0.025,
                       num_ghosts=5,
                       axes=2,
                       intensity=0.75,
                       restore=0.02,
                       degrees=10,
                       translation=10,
                       num_transforms=10,
                       image_interpolation='linear',
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.level_noise = level_noise
        self.transform = transform
        self.num_ghosts = num_ghosts
        self.axes = axes
        self.intensity = intensity
        self.restore = restore
        self.degrees = degrees
        self.translation = translation
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))
        trans = torchio.transforms.RandomGhosting(num_ghosts=int(np.random.randint(low=3,high=self.num_ghosts, size=1)[0]),#5,
                                         axes=np.random.randint(self.axes, size=1)[0],
                                         intensity=np.random.uniform(0.05, self.intensity, 1)[0],# 1.75,
                                         restore=np.random.uniform(0.01, self.restore, 1)[0])# 0.02)
        transMot = torchio.transforms.RandomMotion(degrees=np.random.uniform(0.01, self.degrees, 1)[0],# 10,
                                         translation=np.random.uniform(0.01, self.translation, 1)[0],# 10,
                                         num_transforms=int(np.random.randint(low=2,high=self.num_transforms, size=1)[0]),#5,
                                         image_interpolation='linear')
        ## augmentation 3D:
        img = randomContrastAug(cutNoise(image_in_, self.level_noise))
        ## corrupt with torchio:
        testtens = torch.unsqueeze(torch.tensor(img),0)
        d_ = transMot(testtens)
        d_ = trans(d_)
        cor = d_.detach().cpu().numpy().squeeze()
        ### -------
        for ii in range(0, self.patches):
            ## select slice:
            imgin, imgcor, slice, orient = select_slice_orientation_both(img, cor, self.orientation)
            
            if (imgin.shape[0]<=size_) and (imgin.shape[1]<=size_):
                imgin = pad2D(imgin, size_, size_)
                imgcor = pad2D(imgcor, size_, size_)       
            elif (imgin.shape[0]>size_) and (imgin.shape[1]>size_):
                imgin = PadAndResize(imgin, width=size_, height=size_)
                imgcor = PadAndResize(imgcor, width=size_, height=size_)
            elif (imgin.shape[0]<=size_) and (imgin.shape[1]>size_):
                imgin = PadAndResize(imgin, width=size_, height=size_)
                imgcor = PadAndResize(imgcor, width=size_, height=size_)
            elif (imgin.shape[0]>size_) and (imgin.shape[1]<=size_):
                imgin = PadAndResize(imgin, width=size_, height=size_)
                imgcor = PadAndResize(imgcor, width=size_, height=size_)
            else:
                print(imgin.shape, 'ciao')
            ### ------- 
            
            ssimtmp = structural_similarity(imgin, imgcor, data_range=1)

            stackcor[ii,:,:]= imgcor
            stackimg[ii,:,:]= imgin
            stackssim[ii,0] = ssimtmp
        
        if self.transform:
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            
        return stackcor, stackimg, stackssim
###########################################################################
###########################################################################
###########################################################################
def tensorboard_regression(writer, inputs, outputs, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def tensorboard_correction(writer, inputs, outputs, targets, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                        vutils.make_grid(targets[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def tensorboard_correction_kspace(writer, inputsFS, inputs, outputs, targets, targetsFS, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputsFS is not None:
        writer.add_image('{}/inputFS'.format(section),
                        vutils.make_grid(inputsFS[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                        vutils.make_grid(targets[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targetsFS is not None:
        writer.add_image('{}/targetFS'.format(section),
                        vutils.make_grid(targetsFS[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def getSSIM(gt, out, gt_flag, data_range=1):
    vals = []
    for i in range(gt.shape[0]):
        if not gt_flag[i]:
            continue
        for j in range(gt.shape[1]):
            vals.append(structural_similarity(gt[i,j,...], out[i,j,...], data_range=data_range))
    return median(vals)
