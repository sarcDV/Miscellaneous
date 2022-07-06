import os, sys
import random

# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --------
import nibabel as nib
from skimage.transform import resize
from models.ReconResNetV2 import ResNet


def main():
    """run with the following command:
       python test-clinical-evaluate-nii.py nii-file (file.nii.gz)
       filein, chkin, neuralmodel, eventually plus motion correction and brain extraction!
       python clinical-SSIM-regression.py TestClinicalData/subj-clinical-1/01.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18 Checkpoints/RN56-MC-plus-BE.pth.tar 
    """
    clinical_evaluate_nii(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])
    return

###########################################################################
##### Auxiliaries  ########################################################
###########################################################################
def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def PadResize(imgin, finalsize):
    ## find biggest size:
    dim_ = imgin.shape
    maxElement = np.where(dim_== np.amax(dim_))
    imgout = padding(imgin, dim_[maxElement[0][0]], dim_[maxElement[0][0]])
    imgres = resize(imgout, (finalsize, finalsize), anti_aliasing=True )
    return imgres # imgout, imgres
###########################################################################
##### Cut noise level  ####################################################
###########################################################################

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()-adjimg.min())

    return adjimg # np.abs(adjimg)+1e-16
###########################################################################

def clinical_evaluate_nii(filein, chkin, neuralmodel, chkMCin):
    ### ----------------------------------------------------- ###
    device = torch.device("cuda:0") 
    checkpoint = chkin
    # checkpoint = './Results/TEST-RN18-Regression-TIO-RealityMot-Combined_best.pth.tar'
    # checkpoint = './Results/TEST-RN18-Regression-TIO-RealityMot-Combined-NoContrastAug_best.pth.tar'
    batch_size_, patches, channels, size_= 1,1,1,256
    level_noise = 0.025
    ### ----------------------------------------------------- ###
    if neuralmodel == 'RN18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    else:
        model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'] )
    model.eval()
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    orig_stdout = sys.stdout
    f = open(str(filein[0:len(filein)-7])+'_report.txt', 'w')
    sys.stdout = f
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    a = nib.load(filein).get_fdata()
    ahead = nib.load(filein).header
    print(str(filein))
    print("\n","#### HEADER #####\n\n",ahead,"\n")
    ## original resolution & size ###
    print("#### Size #####\n\n",ahead['dim'][1:4],"\n")
    print("#### Resolution #####\n\n",ahead['pixdim'][1:4],"\n")
    if len(a.shape)>3:
        a = a[:,:,:,0]
    
    SSIMarray = np.zeros((a.shape[2]))
    print("#### Slice, SSIM ####")
    with torch.no_grad():
        for ii in range(0, a.shape[2]):
            img = cutNoise(a[:,:,ii], level_noise)
            img = PadResize(img, size_)
            img = torch.unsqueeze(torch.tensor(img).to(device),1)
            img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
            pred = model(img.float())
            print("Slice: "+str(ii+1)+", "+str(pred.detach().cpu().numpy()[0][0]))
            SSIMarray[ii] = pred.detach().cpu().numpy()[0][0]
    
    print("\n")
    print("Maximum: ", str(np.max(SSIMarray)))
    print("Minimum: ", str(np.min(SSIMarray)))
    print("Mean value: ", str(np.mean(SSIMarray)))
    print("Standard dev.: ", str(np.std(SSIMarray)))
    print("\n")
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    resblocks_ = 56
    modelMC=ResNet(in_channels=channels,out_channels=channels,
	     res_blocks=resblocks_, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, 			
	     do_batchnorm=False, res_drop_prob=0.2,
         out_act="sigmoid", forwardV=0, 
	     upinterp_algo='convtrans', post_interp_convtrans=False, 
	     is3D=False)

    modelMC.to(device)    
    chkMCBE = torch.load(chkMCin, map_location=device)
    modelMC.load_state_dict(chkMCBE['state_dict'] )
    modelMC.eval()

    # corrected_ = np.zeros((ahead['dim'][1:4]))
    corrected_ = np.zeros((256,256,a.shape[2]))
    with torch.no_grad():
        for ii in range(0, a.shape[2]):
            img = cutNoise(a[:,:,ii], level_noise)
            img = PadResize(img, size_)
            img = torch.unsqueeze(torch.tensor(img).to(device),1)
            img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
            out = modelMC(img.float())
            corrected_[:,:,ii] = out.detach().cpu().numpy().squeeze()
    
    corrected_img = nib.Nifti1Image(corrected_, np.eye(4))
    nib.save(corrected_img, str(filein[0:len(filein)-7])+'_corrected.nii.gz')
    sys.stdout = orig_stdout
    f.close()
if __name__ == "__main__":
	main()
