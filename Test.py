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
Model_path='./model_2_epoch.pth'

transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

Model = UNet(n_channels=4, n_classes=1).to(device=device)
Model.load_state_dict(torch.load(Model_path))

#print(Model)
Model.eval()

for i, batch in enumerate(data_loader):
        #print(batch)
        img,true_masks=batch
        
        mask_sum=torch.ones(true_masks.shape[1],true_masks.shape[2]).type(dtype=torch.float32)

        for k in range(0,4):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            #img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
            #print(img[k].size())
        img = torch.stack((img[0], img[1], img[2],img[3]), dim=1)
        #print(img.size())
        masks_pred = Model(img)
        plt.subplot(121)
        plt.imshow(masks_pred[0, 0, :, :].cpu())
        plt.subplot(122)
        plt.imshow(true_masks[0, :, :])
        plt.show()
        '''
         for k in range(0,4):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
            
            masks_pred = Model(img[k])
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


