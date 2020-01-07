import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import time
from dataset import *
from FCNmodel import FCNs,VGGNet


# 可调节参数
val_percent = 0.2
data_file = './data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
epochs = 30
learnrate = 1e-4

# 划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None,add_labeled_sample=False)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 生成网络
vgg_model = VGGNet(requires_grad=True)
net = FCNs(pretrained_net=vgg_model, n_class=2).to(device=device)

# 设置优化器
optimizer = optim.Adam(net.parameters(), lr=learnrate, betas=(0.9, 0.99))

# 设置损失函数算法
criterion = nn.BCELoss().to(device)

# 开始训练
for epoch in range(epochs):
    net.train()
    start = time.time()
    running_loss = 0.0
    ii = 0
    for i, batch in enumerate(train_loader):
        img, true_masks = batch
        # 跳过无肿瘤的样本
        if not label_tumor_exist(true_masks.squeeze().cpu().numpy()):
            continue
        ii = ii + 1
        # 将原本的标签图像拓展成前后背景两个标签图像，注意这里的维度[batch-size，channel，w，h]
        true_masks = layer2_label(true_masks).to(device=device)
        for k in range(0,4):
            # 将灰度图转成伪彩色图像
            img[k] = gray2rgb_tensor(img[k]).to(device=device)
            masks_pred = net(img[k])
            masks_pred = torch.sigmoid(masks_pred)
            # 计算loss值
            loss = criterion(masks_pred, true_masks)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每10个样本输出一次训练状态信息
        if ii % 10 == 9:
            end = time.time()
            print('[epoch {},images {}] training loss = {:.5f}  time: {:.3f} s'.
                    format(epoch+1, (ii+1)*batch_size, running_loss/(10*batch_size*4), end-start))
            start = time.time()
            running_loss = 0.0

    # 在验证集上测试
    val_loss = 0
    net.eval()
    for i, batch in enumerate(val_loader):
        img, true_masks = batch
        true_masks = layer2_label(true_masks).to(device=device)
        for k in range(0, 4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device)
            masks_pred = net(img[k])
            masks_pred = torch.sigmoid(masks_pred)
            val_loss += criterion(masks_pred, true_masks).item()
    print('epoch {}'.format(epoch+1),end='\t')
    print('val_loss:{}'.format(val_loss / (4 * len(val_loader))))

# 保存网络模型
torch.save(net.state_dict(), './FCN_lr'+str(learnrate)+'_model_'+str(epochs)+'_epoch.pth')
