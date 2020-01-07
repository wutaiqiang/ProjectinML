import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from dataset import *
from unet_mlp import UNet

val_percent=0.2
data_file='./data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=1
epochs=30
learnrate=0.0001
pretrain=False

transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
#划分数据集
dataset = MyDataSet(data_file,transform_data=None,transform_label=None,add_labeled_sample=False)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#训练网络
#net = RUNet(BN_enable=True, resnet_pretrain=False).to(device)
#if pretrain:
#    net.load_state_dict(torch.load(Model_path))
net = UNet(n_channels=3, n_classes=2).to(device=device)


#optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
optimizer = optim.Adam(net.parameters(), lr=learnrate, betas=(0.9, 0.99))
#optimizer=optim.SGD(net.parameters(), lr=learnrate,momentum=0.9)
#criterion = nn.BCELoss().to(device)

criterion = nn.BCELoss()

allloss=[]
for epoch in range(epochs):
    net.train()
    start = time.time()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        img, true_masks = batch
        if not label_tumor_exist(true_masks.squeeze().cpu().numpy()):
            continue

        true_masks = layer2_label(true_masks).to(device=device)

        for k in range(0,4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
        masks_pred = net(img)
        masks_pred = torch.sigmoid(masks_pred)

        loss = criterion(masks_pred, true_masks)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            end=time.time()
            print('[epoch {},images {}] training loss = {:.5f}  time: {:.3f} s'.
                    format(epoch + 1, (i+1)*batch_size, running_loss / (4*batch_size*10),end-start))
            start = time.time()
            running_loss = 0.0

    val_loss = 0
    net.eval()
    for i, batch in enumerate(val_loader):
        img, true_masks = batch
        true_masks = layer2_label(true_masks).to(device=device)
        for k in range(0, 4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
        masks_pred = net(img)
        masks_pred = torch.sigmoid(masks_pred)
        val_loss += criterion(masks_pred, true_masks).item()

    allloss.append(val_loss / (4 * len(val_loader)))

    print('epoch {}'.format(epoch+1),end='\t')
    print('val_loss:{}'.format(val_loss / (4 * len(val_loader))))
    if epoch+1 in range(10,35,5):
        torch.save(net.state_dict(), './UNET_mlp_bias_model+'+str(learnrate)+'_lr_'+str(epoch+1)+'_epoch.pth')
    if epoch == epochs-1:
        plt.plot(allloss)
        plt.xlabel('Epoch')
        plt.ylabel('val loss')
        plt.title('Val Loss for Unet_MLP Model')
        plt.savefig("valloss.png")
        plt.show()

