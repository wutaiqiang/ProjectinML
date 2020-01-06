import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import time
from dataset import *
from unet import UNet
from resnet50model import Resnet_Unet as RUNet

from FCNmodel import FCNs,VGGNet
#可调节参数
val_percent=0.2
data_file='./data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')
print(device)
batch_size=5
epochs=5
learnrate=1e-3
pretrain=False
Model_path='./model.pth'
transform = transforms.Compose([
    #transforms.ToTensor(),
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None,add_labeled_sample=True)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#训练网络
#net = RUNet(BN_enable=True, resnet_pretrain=False).to(device)
#if pretrain:
#    net.load_state_dict(torch.load(Model_path))
vgg_model = VGGNet(requires_grad=True)
net = FCNs(pretrained_net=vgg_model, n_class=1).to(device=device)
#print(net)

#optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
#optimizer = optim.Adam(net.parameters(), lr=learnrate)
optimizer=optim.SGD(net.parameters(), lr=learnrate,momentum=0.9)
#criterion = nn.BCELoss().to(device)

#criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
criterion = SoftDiceLoss()

for epoch in range(epochs):
    net.train()
    start = time.time()
    running_loss = 0.0
    ii=0
    for i, batch in enumerate(train_loader):
        img, true_masks = batch
        if not label_tumor_exist(true_masks.squeeze().cpu().numpy()):
            continue
        ii=ii+1
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        true_masks = Variable(torch.unsqueeze(true_masks, dim=1).float(), requires_grad=False)
        #print(batch)

        for k in range(0,4):
            #img[k] = img[k].to(device=device, dtype=torch.float32)
            if img[k].size(0)!=5:
                print(img[k].size())
            img[k]=gray2rgb_tensor(img[k]).to(device=device)
            #print(img[k].size())
            #img[k] = Variable(img[k], requires_grad=False)
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
            img[k] = Variable(img[k],requires_grad=False)
            #print(img[k].size())
        img = torch.stack((img[1], img[2],img[3]), dim=1)
        #print(img.size())
        masks_pred = net(img)
        # masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)  # 二值化
        # masks_pred=torch.sigmoid(masks_pred)
        # 注意这里的维度【batch-size，channel，w，h】
        #print(masks_pred.size())
        #print(true_masks.size())
        #masks_pred[masks_pred < 0.0] = 0.0
        #masks_pred[masks_pred > 1.0] = 1.0
        loss = criterion(masks_pred, true_masks)

        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if ii % 10 == 9:
            end=time.time()
            print('[epoch {},images {}] training loss = {:.5f}  time: {:.3f} s'.
                    format(epoch + 1, (ii+1)*batch_size, running_loss /(10*batch_size*4),end-start))
            start = time.time()
            running_loss = 0.0

    val_loss = 0
    net.eval()
    for i, batch in enumerate(val_loader):
        img, true_masks = batch
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        true_masks = Variable(torch.unsqueeze(true_masks, dim=1).float(), requires_grad=False)
        for k in range(0, 4):
            #img[k] = img[k].to(device=device, dtype=torch.float32)
            img[k] = gray2rgb_tensor(img[k])
            #img[k] = Variable(img[k], requires_grad=False)
            masks_pred = net(img[k])
            #masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)  # 二值化
            #masks_pred[masks_pred < 0.0] = 0.0
            #masks_pred[masks_pred > 1.0] = 1.0
            val_loss += criterion(masks_pred, true_masks).item()
    print('epoch {}'.format(epoch+1),end='\t')
    print('val_loss:{}'.format(val_loss / (4 * len(val_loader))))

torch.save(net.state_dict(), './added_data_stack_model_'+str(epochs)+'_epoch.pth')
