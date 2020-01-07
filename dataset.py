import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

'''

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
'''
def show_img(data,label):
    '''
    将nii得到的图片依次可视化
    '''
    for i in range(data[0].shape[0]):
        for k in range(0,4):
            plt.subplot(2,3,k+1)
            plt.imshow(data[k][i,:,:])
            plt.xlabel('CT '+str(k)+'-'+str(i))
        plt.subplot(2,3,6)
        plt.imshow(label[i,:,:])
        plt.xlabel('Label'+str(i))
        #plt.colorbar()
        plt.show()


def nii2array(image_data):
    '''
    缩放到[0,255]的numpy uint8数组
    '''
    cmin = image_data[:,:].min()
    cmax = image_data[:,:].max()
    if cmax ==0:
        return image_data
    else:
        high = 255
        low = 0
        cscale = cmax - cmin
        scale = float(high - low) / cscale
        bytedata = (image_data[:, :] - cmin) * scale + low
        #image = (bytedata.clip(low, high) + 0.5).astype(np.uint8)
        return bytedata


def readfolder(folder):
    '''
    读取文件夹内的nii
    '''
    dname=['pre.nii','artery.nii','portal.nii','delay.nii']
    l_path = os.path.join(folder,'label.nii')
    img = sitk.ReadImage(l_path)
    label = sitk.GetArrayFromImage(img)

    data=[]
    for k in range(0,4):
        d_path = os.path.join(folder,dname[k])
        img2 = sitk.ReadImage(d_path)
        data.append(sitk.GetArrayFromImage(img2))

    return data,label

class MyDataSet(Dataset):
    '''
    获取数据集，以FZMC开头的nii才行
    '''
    def __init__(self,data_file,transform_data=None,transform_label=None,add_labeled_sample=False):
        self.data_file=data_file
        self.transform1=transform_data
        self.transform2=transform_label
        mfolders = [file for file in os.listdir(data_file) if file.startswith('FZMC')]
        a_data1=[]
        a_data2=[]
        a_data3=[]
        a_data4=[]
        a_label=[]
        for mfolder in mfolders:
            mdata,mlabel=readfolder(data_file+'/'+mfolder)
            aa=min(mdata[0].shape[0],mdata[1].shape[0],mdata[2].shape[0],mdata[3].shape[0])
            for i in range(aa):
                a_data1.append(mdata[0][i,:,:])
                a_data2.append(mdata[1][i,:,:])
                a_data3.append(mdata[2][i,:,:])
                a_data4.append(mdata[3][i,:,:])
                a_label.append(mlabel[i,:,:])
                if add_labeled_sample and label_tumor_exist(mlabel[i,:,:]):
                    a_data1.append(np.fliplr(mdata[0][i,:,:]))
                    a_data2.append(np.fliplr(mdata[1][i,:,:]))
                    a_data3.append(np.fliplr(mdata[2][i,:,:]))
                    a_data4.append(np.fliplr(mdata[3][i,:,:]))
                    a_label.append(np.fliplr(mlabel[i,:,:]))
        self.a_data1=a_data1
        self.a_data2=a_data2
        self.a_data3=a_data3
        self.a_data4=a_data4
        self.a_label=a_label

    def __getitem__(self, index):
        if self.transform1 is not None:

            image1 = self.transform1(self.a_data1[index][:,:])
            image2 = self.transform1(self.a_data2[index][:,:])
            image3 = self.transform1(self.a_data3[index][:,:])
            image4 = self.transform1(self.a_data4[index][:,:])
            
        else:
            image1 = self.a_data1[index][:256,:]
            image2 = self.a_data2[index][:256,:]
            image3 = self.a_data3[index][:256,:]
            image4 = self.a_data4[index][:256,:]
        if self.transform2 is not None:  
            mask = self.transform2(self.a_label[index][:,:])
        else:
            mask = self.a_label[index][:256,:]

        return [[image1/1.0,image2/1.0,
                image3/1.0,image4/1.0],mask/1.0]


    def __len__(self):
        return len(self.a_label)


def label_tumor_exist(a):
    '''
    检测Label是否存在肿瘤
    '''
    aa=np.sum(a)
    return aa>0


def gray2rgb_tensor(input):
    '''
    将输入的灰度图变成[3,h,w]的图
    '''
    output = torch.zeros(input.size(0),3,input.size(1),input.size(2))
    for b in range(input.size(0)):
        a=input[b,:,:]
        maxa = torch.max(a)
        mina = torch.min(a)
        a = (a - mina) / (maxa - mina)
        output[b,:,:,:] = torch.stack((a, a, a), dim=0)
    #print(output.size())
    return output

def layer2_label(input):
    '''
    将label分割成前景与背景
    '''
    output = torch.zeros(input.size(0), 2, input.size(1), input.size(2))
    for j in range(input.size(0)):
        input1 = input[j,:,:]
        input2 = torch.ones_like(input1)
        input2 = input2 - input1
        output[j,:,:,:] = torch.stack((input1, input2), dim=0)
    return output

      
if __name__=='__main__':
    a=MyDataSet('./data_train',add_labeled_sample=False)
    train_loader = DataLoader(a, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    '''
    print(len(a),end='\t')
    T=0
    for ii,batch in enumerate(train_loader):
        i,mask=batch
        if label_tumor_exist(mask.squeeze().cpu().numpy()):
            T=T+1
    print("in which {} with tumor,{} wothout tumor".format(T,len(a)-T))
    '''
    for ii, batch in enumerate(train_loader):
        i,mask=batch
        for j in range(0,4):
            a= gray2rgb_tensor(i[j])
            a= a.cpu().numpy()
            plt.subplot(2, 3, j + 1)
            plt.imshow(a)
        plt.subplot(2,3,6)
        plt.imshow(mask.squeeze())
        plt.show()


       
        
    
    


    

