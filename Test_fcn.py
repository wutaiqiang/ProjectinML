import numpy as np
import torch
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from FCNmodel import FCNs,VGGNet


# 参数
data_file = './data_val'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
Model_path = 'FCN_lr0.0001_model_30_epoch.pth'

# 划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# 建立网络模型
vgg_model = VGGNet(requires_grad=True)
Model = FCNs(pretrained_net=vgg_model, n_class=2).to(device=device)
# 读入保存好的网络参数
Model.load_state_dict(torch.load(Model_path))

# 开始测试
Model.eval()
titles = ['Pre Phase','Arterial Phase','Portal Phase','Delay Phase']
for i, batch in enumerate(data_loader):
        img,true_masks = batch
        mask_sum = torch.zeros(256,320)
        for k in range(0,4):
            # 将灰度图转成伪彩色图像
            img[k] = gray2rgb_tensor(img[k]).to(device=device)
            masks_pred = Model(img[k])
            # 只取前景分割图出来进行二值化
            masks_pred = torch.ge(masks_pred[0, 0, :, :], 0.0000001).type(dtype=torch.float32)
            plt.subplot(2,3,k+1)
            plt.imshow(img[k][0,0,:,:].cpu())
            plt.title(titles[k])
            # 将4个分割结果叠加作为最后的分割结果
            mask_sum.add_(masks_pred.cpu())
        plt.subplot(2,3,5)
        plt.imshow(mask_sum)
        plt.title('FCN Segmentation Results')
        plt.subplot(2,3,6)
        plt.imshow(true_masks[0,:,:])
        plt.title('Label')
        plt.tight_layout()
        #plt.pause(2)
        plt.show()



