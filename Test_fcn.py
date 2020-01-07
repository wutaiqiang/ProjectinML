import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from dataset import *
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

from unet import UNet

from FCNmodel import FCNs,VGGNet
#参数
data_file= 'data_test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=1
Model_path= 'FCN_lr0.0001_model_20_epoch.pth'

transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


vgg_model = VGGNet(requires_grad=True)
Model = FCNs(pretrained_net=vgg_model, n_class=2).to(device=device)
Model.load_state_dict(torch.load(Model_path))
#print(Model)
Model.eval()

for i, batch in enumerate(data_loader):
        #print(batch)
        img,true_masks=batch

        '''
        for k in range(0,4):
            img[k]=gray2rgb_tensor(img[k]).to(device=device)
            masks_pred = Model(img[k])
            masks_pred = torch.ge(masks_pred[0,0,:,:], 0.0000001).type(dtype=torch.float32)
            #masks_pred = torch.sigmoid(masks_pred)
            plt.subplot(221)
            plt.suptitle('PICTURE {}'.format(k+1))
            plt.imshow(img[k][0,0,:,:].cpu().numpy())
            plt.subplot(222)
            plt.imshow(masks_pred.cpu())
            plt.subplot(223)
            plt.imshow(true_masks[0, :, :])
            #plt.pause(2)
            plt.show()
        '''
        mask_sum=torch.zeros(256,320)
        for k in range(0,4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device)
            masks_pred = Model(img[k])
            masks_pred = torch.ge(masks_pred[0, 0, :, :], 0.0000001).type(dtype=torch.float32)
            plt.subplot(2,3,k+1)
            #plt.imshow(masks_pred.cpu())
            plt.imshow(img[k][0,0,:,:].cpu())
            mask_sum.add_(masks_pred.cpu())
        plt.subplot(2,3,5)
        plt.imshow(mask_sum)
        plt.subplot(2,3,6)
        plt.imshow(true_masks[0,:,:])
        #plt.pause(2)
        plt.show()



