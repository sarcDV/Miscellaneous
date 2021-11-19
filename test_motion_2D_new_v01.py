import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import random
import torch
import glob, os, sys
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from skimage import exposure
from progressbar import ProgressBar
from skimage.metrics import structural_similarity as ssim
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

    return adjimg, expo_selection[0]

###########################################################################
##### Motion  #############################################################
###########################################################################

def generate_motion_2d(img):
    aux = np.zeros([img.shape[0],img.shape[1]]) + 1j*np.zeros([img.shape[0],img.shape[1]])
    axis_selection = np.random.randint(0,2,1)
    
    if axis_selection[0] == 0:
        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[1]//n_), 
                                                        size=int(img.shape[1]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[1]//n_), 
                                                        high=int(img.shape[1]), 
                                                        size=int(img.shape[1]//2*n_), dtype=int))) 
            portion = np.concatenate((portiona, portionb))  
        else:
            portion = np.sort(np.unique(np.random.randint(low=int(img.shape[1]//2)-int(img.shape[1]//n_+1), 
                                                     high=int(img.shape[1]//2)+int(img.shape[1]//n_+1), 
                                                     size=int(img.shape[1]//n_+1), dtype=int)))
        random_rots = np.random.randint(-4,4,len(portion))
        
        for kk in range(0, len(portion)-1):
            img_aux = ndimage.rotate(img, random_rots[kk], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[:,portion[kk]]=img_h[:,portion[kk]]        
        cor =np.abs(np.fft.ifft2(aux)) + 0.2 *img
    else:
        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[0]//n_), 
                                                        size=int(img.shape[0]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[0]//n_), 
                                                        high=int(img.shape[0]), 
                                                        size=int(img.shape[0]//2*n_), dtype=int))) 
            portion = np.concatenate((portiona, portionb))
        else:
            portion = np.sort(np.unique(np.random.randint(low=int(img.shape[0]//2)-int(img.shape[0]//n_), 
                                                     high=int(img.shape[0]//2)+int(img.shape[0]//n_), 
                                                     size=int(img.shape[0]//n_), dtype=int)))
        random_rots =  np.random.randint(-4,4,len(portion))
        for kk in range(0, len(portion)):
            img_aux = ndimage.rotate(img, random_rots[kk], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[portion[kk],:]=img_h[portion[kk],:]        
        cor =np.abs(np.fft.ifft2(aux)) +  0.2 *img 
    return cor/(cor.max()+1e-16)


def corrupt_recursively(test):
    rnd_orient = np.random.randint(0,3,1)[0]
    # print(rnd_orient)
    if rnd_orient == 0:
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                    high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                    size=1)
        
        img = (test[:,rndslice_[0],:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    elif rnd_orient == 1:
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                    high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                    size=1)
                                   
        img = np.rot90(test[:,:,rndslice_[0]]) 
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    else:
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                    high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                    size=1)
                                    
        img = np.flipud(test[rndslice_[0],:,:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)

    return img, cor, rndslice_, rnd_orient

def select_slice_orientation(test):
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

    img = (img-img.min())/(img.max()-img.min())
            
    return img, rndslice_, rnd_orient
###########################################################################
##### Motion Corruption Class  ############################################
###########################################################################
class MoCoDataset():
    """Motion Correction Dataset"""
    def __init__(self, input_list, transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.transform = transform

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        # print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        img, slice, orient = select_slice_orientation(test)
        if (img.shape[0]<=256) and (img.shape[1]<=256):
            img = pad2D(img, 256, 256)       
        elif (img.shape[0]>256) and (img.shape[1]>256):
            img = randomCropInput(img, width=256, height=256)
        elif (img.shape[0]<=256) and (img.shape[1]>256):
            img = pad2DD1(img, 256) 
            img = randomCropInput(img,  width=256, height=256)
        elif (img.shape[0]>256) and (img.shape[1]<=256):
            img = pad2DD2(img, 256) 
            img = randomCropInput(img,  width=256, height=256)
        else:
            print(img.shape, 'ciao')
        ### -------     
        img = randomContrastAug(img)
        cor = generate_motion_2d(img)
        ssimtmp = ssim(img, cor, data_range=1)

        if self.transform:
            img = self.transform(img)
            cor = self.transform(cor)
            ssimtmp = self.transform(ssimtmp)
        
        return cor, img, ssimtmp


###########################################################################
###########################################################################
###########################################################################

# #defining path for required files
# target_train_ = glob.glob('./selected_1mm/train/*')
# #input_train_  = glob.glob('./selected_112mm/train/*')
# input_train_ = []
# for string in target_train_:
#     tmp_= string.replace("1mm", "112mm")
#     input_train_.append(tmp_)
# # -.-.-.-.-.-.-.-.-.-.-.-.-..-.-.-.-.-.--.-.-.-.-.-.-.- # 

### ------------------------------------------------------------
### ------------------------------------------------------------
### ------------------------------------------------------------
### 1 - folder containing N nii files
### 2 - pick S random volumes
### 3 - create data loader with [batch_size, channel, width, height]

#test = nib.load('./niifiles/T1_sag_1_o.nii.gz').get_fdata()
test = nib.load('./niifiles/IXI002-Guys-0828-T1.nii.gz').get_fdata()
# test = nib.load('./niifiles/IXI527-HH-2376-PD.nii.gz').get_fdata()
# test = nib.load('./niifiles/IXI662-Guys-1120-T2.nii.gz').get_fdata()
# test = nib.load('./niifiles/au70_025_ON.nii.gz').get_fdata()
# test = nib.load('./niifiles/au70_T1_ON_enhanced.nii.gz').get_fdata()
# img, cor, slice, orient = corrupt_recursively(test)


# mask = np.zeros((256,256))
ssimarray_ = np.zeros((100))
niiinit_ = np.zeros((512,256,len(ssimarray_)))
## for ii in range(0, len(ssimarray_)):
ii = 0
while ii < len(ssimarray_):
    img, slice, orient = select_slice_orientation(test)
    if (img.shape[0]<=256) and (img.shape[1]<=256):
        img = pad2D(img, 256, 256)       
    elif (img.shape[0]>256) and (img.shape[1]>256):
        img = randomCropInput(img, width=256, height=256)
    elif (img.shape[0]<=256) and (img.shape[1]>256):
        img = pad2DD1(img, 256) 
        img = randomCropInput(img,  width=256, height=256)
    elif (img.shape[0]>256) and (img.shape[1]<=256):
        img = pad2DD2(img, 256) 
        img = randomCropInput(img,  width=256, height=256)
    else:
        print(img.shape, 'ciao')

    img, expo = randomContrastAug(img)
    cor = generate_motion_2d(img)
    ssimtmp = ssim(img, cor, data_range=1)
    if ssimtmp > 0.01: # 0.65 and ssimtmp < 0.85:
        print("Iteration# "+str(ii+1)+", SSIM="+str(ssim(img, cor, data_range=1))+\
               " slice: "+str(slice)+", orientation: " +str(orient)+", exposure adjs: "+str(expo))
        comb = np.concatenate((np.rot90(img), np.rot90(cor)), axis=0)
        ssimarray_[ii] = ssim(img, cor, data_range=1)
        niiinit_[:,:,ii] = comb 
        ii += 1
    else:
        ii = ii
    # img, cor, slice, orient = corrupt_recursively(test)
    # ssimtmp = ssim(img, cor, data_range=1)
    # if ssimtmp > 0.65 and ssimtmp < 0.85:
    #     print("Iteration# "+str(ii+1)+", SSIM="+str(ssim(img, cor, data_range=1))+\
    #         " slice: "+str(slice)+", orientation: " +str(orient))
    #     ssimarray_[ii] = ssim(img, cor, data_range=1)
    #     if (img.shape[0]<=256) and (img.shape[1]<=256):
    #         img = pad2D(img, 256, 256) 
    #         cor = pad2D(cor, 256, 256)
    #     elif (img.shape[0]>256) and (img.shape[1]>256):
    #         img, cor = randomCrop(img, cor, width=256, height=256)
    #     elif (img.shape[0]<=256) and (img.shape[1]>256):
    #         img = pad2DD1(img, 256) 
    #         cor = pad2DD1(cor, 256)
    #         img, cor = randomCrop(img, cor, width=256, height=256)
    #     elif (img.shape[0]>256) and (img.shape[1]<=256):
    #         img = pad2DD2(img, 256) 
    #         cor = pad2DD2(cor, 256)
    #         img, cor = randomCrop(img, cor, width=256, height=256)
    #     else:
    #         print(img.shape, 'ciao')
    #     # img = pad2D(img, 256, 256)
    #     # cor = pad2D(cor, 256, 256)
    #     comb = np.concatenate((np.rot90(img), np.rot90(cor)), axis=0)
    #     niiinit_[:,:,ii] = comb 
    #     ii += 1
    # else:
    #     ii = ii

c_= nib.Nifti1Image(np.abs(niiinit_), None)
nib.save(c_, 'test_new_2D_v14.nii.gz')
# plt.figure()
# plt.plot(ssimarray_,'*-')
# plt.title('SSIM')
# plt.show()