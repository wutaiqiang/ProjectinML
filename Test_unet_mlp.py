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

from unet_mlp import UNet
from resnet50model import Resnet_Unet as RUNet

#参数
data_file='./data_val'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=1
Model_path='./UNET_mlp_model_lr__epoch.pth'

dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
Model = UNet(n_channels=3, n_classes=1).to(device=device)
Model.load_state_dict(torch.load(Model_path))

Model.eval()
for i, batch in enumerate(data_loader):

        img,true_masks=batch
        for k in range(0,4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
        masks_pred = Model(img)
        masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)
        plt.subplot(211)
        plt.imshow(masks_pred[0,0,:,:].cpu())
        plt.subplot(212)
        plt.imshow(true_masks[0,:,:])
        plt.pause(2)
        #plt.show()


