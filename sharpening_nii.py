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
    sharpening_interpolated(sys.argv[1], sys.argv[2])
    return 

def sharpening_interpolated(file, kernel_value):
    print("                                            ")
    print("Sharpening ....: " + str(file))
    print("                                            ")

    source_ = nib.load(file)
    n1_header = source_.header
    # n1_header = n1_header.get_dim_info()
    # print(n1_header)
    source_ = (source_.get_fdata())
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
    ################################################################
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, np.int(kernel_value),-1],
                              [-1,-1,-1]])
    ################################################################
    tmp_ = np.zeros(source_.shape)
    #print(kernel_value)
    for ii in range(0, source_.shape[2]):
        tmp_[:,:,ii]=cv2.filter2D(source_[:,:,ii], -1, kernel_sharpening)
    
    tmp_ = (tmp_>0)*1*tmp_
    filled_img = nib.Nifti1Image(tmp_, np.eye(4), n1_header)
    nib.save(filled_img, str(file[0:len(file)-7])+'_sharpened.nii.gz')
    # os.system("fslchpixdim "+(str(file[0:len(file)-7])+'_sharpened.nii.gz  ')+str(pixdim1_)+" "
    #                                                                        +str(pixdim2_)+" "
    #                                                                        +str(pixdim3_))
    print("                                            ")
    print("Finished sharpening for: "+str(file))
    print("                                            ")

if __name__ == "__main__":
    main()
