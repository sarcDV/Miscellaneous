#!/usr/bin/python
import numpy as np
import sys
import nibabel as nib
import os, subprocess
import cv2
from PIL import Image
from skimage import io
from scipy import interpolate

def main():
    binary_mask_filling(sys.argv[1])
    return 

def binary_mask_filling(file):
    print("                                            ")
    print("Filling ....: " + str(file))
    print("                                            ")

    source_ = nib.load(file)
    n1_header = source_.header
    # n1_header = n1_header.get_dim_info()
    # print(n1_header)
    source_ = np.uint8(source_.get_fdata())
    if len(source_.shape)>3:
        source_=np.squeeze(source_, axis=3)


    # info_ = subprocess.check_output("fslinfo "+str(file), shell=True)    
    # info_ = info_.split("   ")
    # info_ = [word for line in info_ for word in line.split()]
    
    # pixdim1_ = info_[13]
    # pixdim2_ = info_[15]
    # pixdim3_ = info_[17]
    # print(info_[13], info_[15], info_[17])  
    #print(info_)   
    tmp_ = np.zeros(source_.shape)
    for ii in range(0, source_.shape[2]):
        tmp_[:,:,ii] = fillhole(source_[:,:,ii])
    
    tmp_ = (tmp_>0)*1*tmp_
    filled_img = nib.Nifti1Image(tmp_, np.eye(4), n1_header)
    nib.save(filled_img, str(file[0:len(file)-7])+'_corrected.nii.gz')
    # os.system("fslchpixdim "+(str(file[0:len(file)-7])+'_sharpened.nii.gz  ')+str(pixdim1_)+" "
    #                                                                        +str(pixdim2_)+" "
    #                                                                        +str(pixdim3_))
    print("                                            ")
    print("Finished filling for: "+str(file))
    print("                                            ")

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 


if __name__ == "__main__":
    main()