import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from dataset import MyDataSet
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import time

from unet import UNet
from resnet50model import Resnet_Unet as RUNet
#可调节参数
val_percent=0.1
data_file='./data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=1
epochs=10
learnrate=1e-3
pretrain=False
Model_path='./model.pth'
transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#训练网络
#net = RUNet(BN_enable=True, resnet_pretrain=False).to(device)
#if pretrain:
#    net.load_state_dict(torch.load(Model_path))

net=UNet(n_channels=4, n_classes=1).to(device=device)
#print(net)

#optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
#optimizer = optim.Adam(net.parameters(), lr=learnrate)
optimizer=optim.SGD(net.parameters(), lr=learnrate,momentum=0.9)
#criterion = nn.BCELoss().to(device)

if net.n_classes > 1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    net.train()
    start = time.time()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        img, true_masks = batch
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        true_masks = Variable(torch.unsqueeze(true_masks, dim=1).float(), requires_grad=False)
        #print(batch)
        '''       
        for k in range(0,1):#仅仅使用第一个
            img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)            
            masks_pred = net(img[k])

            #masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)  # 二值化
            #masks_pred=torch.sigmoid(masks_pred)
            #注意这里的维度【batch-size，channel，w，h】
            loss = criterion(masks_pred, true_masks)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        '''
        for k in range(0, 4):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            #img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
            print(img[k].size())
        img = torch.stack((img[0], img[1], img[2],img[3]), dim=1)
        #print(img.size())
        masks_pred = net(img)
        # masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)  # 二值化
        # masks_pred=torch.sigmoid(masks_pred)
        # 注意这里的维度【batch-size，channel，w，h】
        loss = criterion(masks_pred, true_masks)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            end=time.time()
            print('[epoch {},images {}] training loss = {:.5f}  time: {:.3f} s'.
                    format(epoch + 1, (i+1)*batch_size, running_loss / (4*10),end-start))
            start = time.time()
            running_loss = 0.0
    val_loss = 0
    net.eval()
    for i, batch in enumerate(val_loader):
        img, true_masks = batch
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        true_masks = Variable(torch.unsqueeze(true_masks, dim=1).float(), requires_grad=False)
        for k in range(0,1):
            img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = Variable(torch.unsqueeze(img[k], dim=1).float(), requires_grad=False)
            masks_pred = net(img[k])
            #masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)  # 二值化
            val_loss += criterion(masks_pred, true_masks).item()
    print('epoch {}'.format(epoch+1),end='\t')
    print('val_loss:{}'.format(val_loss / (4 * len(val_loader))))

torch.save(net.state_dict(), './using1only_model_'+str(epochs)+'_epoch.pth')
#考虑加入dropout
