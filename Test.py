import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from dataset import MyDataSet
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

from unet import UNet
from resnet50model import Resnet_Unet as RUNet

#参数
data_file='./data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=1
Model_path1='./model_2_epoch.pth'
Model_path2='./model_2_epoch.pth'
Model_path3='./model_2_epoch.pth'
Model_path4='./model_2_epoch.pth'

transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

Model1 = UNet(n_channels=1, n_classes=1).to(device=device)
Model1.load_state_dict(torch.load(Model_path1))
Model2 = UNet(n_channels=1, n_classes=1).to(device=device)
Model2.load_state_dict(torch.load(Model_path2))
Model3 = UNet(n_channels=1, n_classes=1).to(device=device)
Model3.load_state_dict(torch.load(Model_path3))
Model4 = UNet(n_channels=1, n_classes=1).to(device=device)
Model4.load_state_dict(torch.load(Model_path4))
#print(Model)
Model1.eval()
Model2.eval()
Model3.eval()
Model4.eval()

for i, batch in enumerate(data_loader):
        #print(batch)
        img,true_masks=batch
        
        mask_sum=torch.ones(true_masks.shape[1],true_masks.shape[2]).type(dtype=torch.float32)
        
        for k in range(0,4):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
            if k==0:
                masks_pred = Model1(img[k])
            elif k==1:
                masks_pred = Model2(img[k])
            elif k==2:
                masks_pred = Model3(img[k])
            elif k==3:
                masks_pred = Model4(img[k])
            masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)
            plt.subplot(2,3,k+1)
            plt.imshow(masks_pred[0,0,:,:].cpu())
            mask_sum.add_(masks_pred[0,0,:,:].cpu())
        plt.subplot(2,3,5)
        plt.imshow(mask_sum)
        plt.subplot(2,3,6)
        plt.imshow(true_masks[0,:,:])
        plt.show()
            
        '''
        for k in range(0, 4):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
        masks_pred = Model(img)
        masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)
        print(masks_pred)
        plt.subplot(211)
        plt.imshow(masks_pred[0,0,:,:].cpu())
        plt.subplot(212)
        plt.imshow(true_masks[0,:,:])
        plt.show()
        '''


