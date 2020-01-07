import numpy as np
import torch
import torch.nn as nn
import copy
from dataset import *
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from FCNmodel import FCNs,VGGNet
from unet_mlp import UNet as UMnet
from unet import UNet

# 参数
data_file = './data_val'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
FCN_Model_path = 'FCN_lr0.0001_model_30_epoch.pth'
UMnet_Model_path = 'UNET_mlp_bias_model+0.0001_lr_30_epoch.pth'
Unet_Model_path = 'UNET_model+0.0001_lr_30_epoch.pth'

# 划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# 读入FCN
vgg_model = VGGNet(requires_grad=True)
FCN_Model = FCNs(pretrained_net=vgg_model, n_class=2).to(device=device)
FCN_Model.load_state_dict(torch.load(FCN_Model_path))
# 读入UMnet
UMnet_Model = UMnet(n_channels=3, n_classes=2).to(device=device)
UMnet_Model.load_state_dict(torch.load(UMnet_Model_path))
# 读入Unet
Unet_Model = UNet(n_channels=3, n_classes=1).to(device=device)
Unet_Model.load_state_dict(torch.load(Unet_Model_path))

# 开始测试
FCN_Model.eval()
UMnet_Model.eval()
Unet_Model.eval()
titles = ['Pre Phase','Arterial Phase','Portal Phase','Delay Phase']
for i, batch in enumerate(data_loader):
        img, true_masks = batch

        # UMnet
        img_bake = copy.deepcopy(img)
        for k in range(0, 4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
        mask_UMnet_pred = UMnet_Model(img)
        # 对输出图像进行二值化
        mask_UMnet_pred = torch.ge(mask_UMnet_pred, 0.01).type(dtype=torch.float32)

        # Unet
        img = copy.deepcopy(img_bake)
        mask_Unet_pred_sum = torch.zeros(256, 320)
        for k in range(0, 4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
            mask_Unet_pred = Unet_Model(img[k])
            mask_Unet_pred = torch.ge(mask_Unet_pred, 0.5).type(dtype=torch.float32)
            # 将4个二值化的分割图像进行叠加得到最后分割结果
            mask_Unet_pred_sum.add_(mask_Unet_pred[0, 0, :, :].cpu())

        # FCN
        img = copy.deepcopy(img_bake)
        mask_FCN_pred_sum = torch.zeros(256,320)
        for k in range(0,4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device)
            masks_FCN_pred = FCN_Model(img[k])
            # 只取前景分割图出来进行二值化
            masks_FCN_pred = torch.ge(masks_FCN_pred[0, 0, :, :], 0.0000001).type(dtype=torch.float32)
            # 将4个二值化的分割图像进行叠加得到最后分割结果
            mask_FCN_pred_sum.add_(masks_FCN_pred.cpu())

        # 显示3个网络模型的分割结果
        for k in range(0, 4):
            plt.subplot(2, 4, k + 1)
            plt.imshow(img[k][0, 0, :, :].cpu())
            plt.title(titles[k])
        plt.subplot(2, 4, 5)
        plt.imshow(mask_Unet_pred_sum)
        plt.title('Unet')
        plt.subplot(2, 4, 6)
        plt.imshow(mask_UMnet_pred[0,0,:,:].cpu())
        plt.title('UMnet')
        plt.subplot(2, 4, 7)
        plt.imshow(mask_FCN_pred_sum)
        plt.title('FCN')
        plt.subplot(2, 4, 8)
        plt.imshow(true_masks[0, :, :])
        plt.title('Label')
        plt.tight_layout()
        #plt.pause(2)
        plt.show()



