import os, sys
import random
from timeit import default_timer as timer
# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --------
import nibabel as nib
from skimage.transform import resize

def main():
    """run with the following command:
       python test-clinical-evaluate-nii.py nii-file (file.nii.gz)
       python test-clinical-evaluate-nii.py nii.file, checkpoint, model (RN18), CPU or GPU
    """
    test_clinical_evaluate_nii(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])
    return

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg 

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

def PadAndResize(img, width, height):
    dsize = (width, height)
    if img.shape[0]==img.shape[1]:
        img  = resize(img, dsize)
    elif img.shape[0]>img.shape[1]:
        img = pad2DD2(img, img.shape[0])
        img = resize(img, dsize)
    elif img.shape[0]<img.shape[1]:
        img = pad2DD1(img, img.shape[1])
        img = resize(img, dsize)
    return img

def ResizeAndPadVolume(img, width, height):
    """2d resize and padding """
    dsize = (width, height)
    imgdsize_ = np.zeros((width, height, img.shape[2]))
    if img.shape[0] == img.shape[1]:
        for ii in range(0, img.shape[2]):
            imgdsize_[:,:,ii] = resize(img[:,:,ii], dsize)
    else:
        ## find largest in-plane size:
        maxdim_ = np.argmax((img.shape[0],img.shape[1]))
        if maxdim_ == 0:
            ## calculate the ratio:
            dsize0_ = (width, int((width/img.shape[0])*img.shape[1]))
            for ii in range(0,img.shape[2]):
                imgdsize_[:,:,ii]= pad2D(resize(img[:,:,ii], dsize0_), width, height)
        elif maxdim_ == 1:
            ## calculate the ratio:
            dsize1_ = (int((height/img.shape[1])*img.shape[0]), height )
            for ii in range(0,img.shape[2]):
                imgdsize_[:,:,ii]=pad2D(resize(img[:,:,ii], dsize1_), width, height)

    ## imgdsize_ = pad3D(imgdsize_,  width, height, width)
    return imgdsize_

def checkSIZE(img, width, height):
    if img.shape[0] < width or img.shape[1] < height :
        img = pad2D(img, width, height)

    return img
###########################################################################

def test_clinical_evaluate_nii(filein, chkin, neuralmodel, dev='GPU'):
    ### ----------------------------------------------------- ###
    if dev == 'GPU':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
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
    # print("Evaluating: " + str(filein))
    a = nib.load(filein).get_fdata()
    if len(a.shape)>3:
        a = a[:,:,:,0]
    
    # start = timer()
    # ...
    # end = timer()
    # print(end - start) # Time in seconds, e.g. 5.38091952400282
    start = timer()
    with torch.no_grad():
        for ii in range(0, a.shape[2]):
            img = cutNoise(a[:,:,ii], level_noise)
            img = PadAndResize(img, size_, size_)
            img = checkSIZE(img, size_, size_)
            img = torch.unsqueeze(torch.tensor(img).to(device),1)
            img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
            pred = model(img.float())
            # print(str(ii+1)+" "+str(pred.detach().cpu().numpy()[0][0]))
        end = timer()
    
    print(end - start) # Time in seconds, e.g. 5.38091952400282


    
    #  a = a/a.max()
    # with torch.no_grad():
    #     if a.shape[2]<10:
    #         for ii in range(0, a.shape[2]):
    #             img = cutNoise(a[:,:,ii], level_noise)
    #             img = PadAndResize(img, size_, size_)
    #             img = checkSIZE(img, size_, size_)
    #             img = torch.unsqueeze(torch.tensor(img).to(device),1)
    #             img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
    #             pred = model(img.float())
    #             print("Predicted SSIM value for slice: "+str(ii+1)+" "+str(pred.detach().cpu().numpy()))
    #     else:
    #         for ii in range(a.shape[2]//2-a.shape[2]//4, a.shape[2]//2+a.shape[2]//4):
    #             img = cutNoise(a[:,:,ii], level_noise)
    #             img = PadAndResize(img, size_, size_)
    #             img = checkSIZE(img, size_, size_)
    #             img = torch.unsqueeze(torch.tensor(img).to(device),1)
    #             img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
    #             pred = model(img.float())
    #             print("Predicted SSIM value for slice: "+str(ii+1)+" "+str(pred.detach().cpu().numpy()))

if __name__ == "__main__":
	main()
