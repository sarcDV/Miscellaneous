import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import random
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from progressbar import ProgressBar
from skimage.metrics import structural_similarity as ssim
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
        
        for kk in range(0, len(portion)):
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
    if rnd_orient == 0:
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//3), 
                                    high=int(test.shape[1]//2)+int(test.shape[1]//3)-4, 
                                    size=1)
        img = (test[:,rndslice_[0],:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    elif rnd_orient == 1:
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//3), 
                                    high=int(test.shape[2]//2)+int(test.shape[2]//3)-4, 
                                    size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    else:
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//3), 
                                    high=int(test.shape[0]//2)+int(test.shape[0]//3)-4, 
                                    size=1)
        img = np.flipud(test[rndslice_[0],:,:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)

    return img, cor, rndslice_, rnd_orient

test = nib.load('T1_sag_1_o.nii.gz').get_fdata()

ssimarray_ = np.zeros((100))

for ii in range(0, len(ssimarray_)):
    img, cor, slice, orient = corrupt_recursively(test)
    print("Iteration# "+str(ii+1)+", SSIM="+str(ssim(img, cor, data_range=1))+\
          " slice: "+str(slice)+", orientation: " +str(orient))
    ssimarray_[ii] = ssim(img, cor, data_range=1)

plt.figure()
plt.plot(ssimarray_)
plt.show()

# c_= nib.Nifti1Image(np.abs(test_m), None)
# nib.save(c_, 'test_m3D_gaussian.nii.gz')
