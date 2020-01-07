
from dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet import UNet

# 参数
data_file = './data_train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
Model_path = './UNET_model+0.0001_lr_30_epoch.pth'

dataset = MyDataSet(data_file, transform_data=None,transform_label=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
Model = UNet(n_channels=3, n_classes=1).to(device=device)
Model.load_state_dict(torch.load(Model_path))

Model.eval()
for i, batch in enumerate(data_loader):
        img,true_masks = batch
        mask_sum = torch.zeros(256,320)
        for k in range(0, 4):
            img[k] = gray2rgb_tensor(img[k]).to(device=device, dtype=torch.float32)
            masks_pred = Model(img[k])
            masks_pred = torch.ge(masks_pred, 0.5).type(dtype=torch.float32)
            plt.subplot(2, 3, k+1)
            plt.imshow(masks_pred[0,0,:,:].cpu())
            mask_sum.add_(masks_pred[0,0,:,:].cpu())
        plt.subplot(2, 3, 5)
        plt.imshow(mask_sum)
        plt.subplot(2, 3, 6)
        plt.imshow(true_masks[0,:,:])
        plt.pause(2)
        #plt.show()


