import os, sys
sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the executors
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './utils')
sys.path.insert(0, './pLoss')
sys.path.insert(0, './models')
import random, math
from glob import glob
from tqdm import tqdm
import logging
from statistics import median
# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# --------
from torch.utils.tensorboard import SummaryWriter
import torchio as tio
from torchio.data.io import read_image
import nibabel as nib

from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)

from utilities_new import MoCoDatasetRegression, tensorboard_correction, getSSIM, kspace_clearcore, \
                          kspace_cropcenter, from_kspacecore_toImg, tensorboard_correction_kspace, from_kspace_toImg
from pLoss.perceptual_loss import PerceptualLoss
from models.ReconResNetV2 import ResNet

### ----------------------------------------------------- ###
# os.environ["CUDA_VISIBLE_DEVICES"] = 0
# cuda = 1
# non_deter = False
# seed = 1701
# torch.backends.cudnn.benchmark = non_deter 
# if not non_deter:
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

device = torch.device("cuda:2") # if torch.cuda.is_available() and cuda else "cpu")
log_path = './TBLogs'
trainID = 'RN56-ALL-BS1-kspace-outer32-debug'

save_path = './Results'
tb_writer = SummaryWriter(log_dir = os.path.join(log_path,trainID))
os.makedirs(save_path, exist_ok=True)
logname = os.path.join(save_path, 'log_'+trainID+'.txt')

logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
torch.manual_seed(0)

traindata = glob('/pool/alex/Motion-Correction-3D/samples/output/train-new/*.nii.gz')
valdata = glob('/pool/alex/Motion-Correction-3D/samples/output/val-new/*.nii.gz')


batch_size_ = 1
patches = 10
channels = 2 #1
resblocks_ = 56
size_ = 256
coresize_ = 32
sigma_range = (1.0,2.0)
orientation = 3 # 
level_noise = 0.025
trainset = MoCoDatasetRegression(traindata, patches=patches, 
                       size=size_,
                       orientation=orientation,
                       sigma_range=sigma_range,
                       level_noise= level_noise,
                       transform=None)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, sampler=None, shuffle=True)

valset = MoCoDatasetRegression(valdata, patches=patches,size=size_,
                    orientation=orientation,
                    sigma_range=sigma_range,
                    level_noise= level_noise,
                    transform=None)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_, sampler=None, shuffle=True)

### ----------------------------------------------------- ###
model=ResNet(in_channels=channels,out_channels=channels,
	     res_blocks=resblocks_, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, 			
	     do_batchnorm=False, res_drop_prob=0.2,
         out_act="tanh", forwardV=0,  # out_act="sigmoid"
	     upinterp_algo='convtrans', post_interp_convtrans=False, 
	     is3D=False)

model.to(device)
### ----------------------------------------------------- ###
start_epoch = 0
num_epochs= 2000
learning_rate = 3e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scaler = GradScaler(enabled=True)
log_freq = 10
save_freq = 1
checkpoint = ""

ploss_level = math.inf
ploss_type = "L1"
# loss_func = nn.MSELoss()
loss_func = PerceptualLoss(device=device, loss_model="resnext1012D", 
                           n_level=ploss_level, loss_type=ploss_type)
if checkpoint:
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'])
    optimizer.load_state_dict(chk['optimizer'])
    scaler.load_state_dict(chk['AMPScaler'])  
    best_loss = chk['best_loss']  
    start_epoch = chk['epoch'] + 1
else:
    start_epoch = 0
    best_loss = float('inf')

### ----------------------------------------------------- ###
for epoch in range(start_epoch, num_epochs):
    ### --- Train --- ###
    model.train()
    runningLoss = []
    train_loss = []
    print('Epoch '+ str(epoch)+ ': Train')
    for idx, (cor, img, ssimtmp) in enumerate(tqdm(train_loader)):
        cor = cor.to(device)
        img = img.to(device)
        ssimtmp = ssimtmp.to(device)
        
        for jj in range(0, patches):
            inp, gt = cor[0,jj,:,:], img[0,jj,:,:]
            inp = torch.unsqueeze(torch.unsqueeze(inp,0), 0)
            gt = torch.unsqueeze(torch.unsqueeze(gt,0), 0)
            inp = Variable(inp).float().to(device)
            gt = Variable(gt).float().to(device)
            
            optimizer.zero_grad()

            with autocast(enabled=True):
                inpksp =kspace_clearcore(inp, size_, coresize_) # kspace_cropcenter(inp, coresize_)
                gtksp = kspace_clearcore(gt, size_, coresize_) # kspace_cropcenter(gt, coresize_)
                out = model(inpksp.to(device))
                outimg = from_kspace_toImg(out) # from_kspacecore_toImg(out, size_, coresize_)
                gtimg = from_kspace_toImg(gtksp) #from_kspacecore_toImg(gtksp, size_, coresize_)
                loss = loss_func(outimg.to(device), gtimg.to(device))
                # out = model(inp)
                # loss = loss_func(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = round(loss.data.item(),4)
            train_loss.append(loss)
            runningLoss.append(loss)
            logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, idx, len(train_loader), loss))

            if idx % log_freq == 0:
                niter = epoch*len(train_loader)+idx
                tb_writer.add_scalar('Train/Loss', median(runningLoss), niter)
                inpimg = from_kspace_toImg(inpksp) # from_kspacecore_toImg(inpksp, size_, coresize_)
                tensorboard_correction_kspace(tb_writer, inp, inpimg, outimg, gtimg, gt, epoch, 'train')
                # tensorboard_correction(tb_writer, inp, out.detach(), gt, epoch, 'train')
                runningLoss = []

    
    
    if epoch % save_freq == 0:            
        checkpoint = {
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'AMPScaler': scaler.state_dict()         
                    }
        torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
                    
    tb_writer.add_scalar('Train/EpochLoss', median(train_loss), epoch)
    # ### ------  validation ------  ####
    if val_loader:
        model.eval()
        with torch.no_grad():
            runningLoss = []
            val_loss = []
            runningAcc = []
            val_acc = []
            print('Epoch '+ str(epoch)+ ': Val')
            for i, (cor, img, ssimtmp) in enumerate(tqdm(val_loader)):
                cor = cor.to(device)# torch.unsqueeze(cor.to(device),1).float()
                img = img.to(device)# torch.unsqueeze(img.to(device),1).float()
                ssimtmp = ssimtmp.to(device)# torch.unsqueeze(ssimtmp.to(device),1).float()
                    
                for jj in range(0, patches):
                    inp, gt = cor[0,jj,:,:], img[0,jj,:,:]
                    inp = torch.unsqueeze(torch.unsqueeze(inp,0), 0)
                    gt = torch.unsqueeze(torch.unsqueeze(gt,0), 0)
                    inp = Variable(inp).float().to(device)
                    gt = Variable(gt).float().to(device)

                    with autocast(enabled=True):
                        inpksp =kspace_clearcore(inp, size_, coresize_) # kspace_cropcenter(inp, coresize_)
                        gtksp = kspace_clearcore(gt, size_, coresize_) # kspace_cropcenter(gt, coresize_)
                        out = model(inpksp.to(device))
                        outimg = from_kspace_toImg(out) # from_kspacecore_toImg(out, size_, coresize_)
                        gtimg = from_kspace_toImg(gtksp) #from_kspacecore_toImg(gtksp, size_, coresize_)
                        loss = loss_func(outimg.to(device), gtimg.to(device))
                        # out = model(inp)
                        # loss = loss_func(out, gt)

                    ssim = getSSIM(gtimg.detach().cpu().numpy(), outimg.detach().cpu().numpy(), gt_flag=[True]*gtimg.shape[0], data_range=1)

                    # ssim = getSSIM(gt.detach().cpu().numpy(), out.detach().cpu().numpy(), gt_flag=[True]*gt.shape[0], data_range=1)

                    loss = round(loss.data.item(),4)
                    val_loss.append(loss)
                    runningLoss.append(loss)
                    val_acc.append(ssim)
                    runningAcc.append(ssim)
                    logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))

                    #For tensorboard
                    if i % log_freq == 0:
                        niter = epoch*len(val_loader)+i
                        tb_writer.add_scalar('Val/Loss', median(runningLoss), niter)
                        tb_writer.add_scalar('Val/SSIM', median(runningAcc), niter)
                        # tensorboard_correction(tb_writer, inp, out.detach(), gt, epoch, 'val') 
                        inpimg = from_kspace_toImg(inpksp) #  inpimg = from_kspacecore_toImg(inpksp, size_, coresize_)
                        tensorboard_correction_kspace(tb_writer, inp, inpimg, outimg, gtimg, gt, epoch, 'val')
                                   
                        runningLoss = []
                        runningAcc = []
            
                if median(val_loss) < best_loss:
                    best_loss = median(val_loss)
                    checkpoint = {
                                    'epoch': epoch,
                                    'best_loss': best_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'AMPScaler': scaler.state_dict()         
                                }
                    torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
                
        tb_writer.add_scalar('Val/EpochLoss', median(val_loss), epoch)
        tb_writer.add_scalar('Val/EpochSSIM', median(val_acc), epoch)
