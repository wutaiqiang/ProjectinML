
import numpy as np
import matplotlib.pyplot as plt

def readtxt(txtdir):
    valloss = []
    trainloss = []
    loss=[]
    with open(txtdir,'r+',encoding='utf8') as f:
        while True:
            a = f.readline()
            if a[:1] == '[':
                loss.append(float(a.split('=')[1][0:7]))
            if a[:5]== 'epoch':
                epoch,val = a.split('\t',2)
                valloss.append(float(val.split(':',2)[1][0:7]))
                trainloss.append(loss)
                loss = []
            if not a:
                break
    tloss = [np.mean(a) for a in trainloss]
    return tloss,valloss

if __name__=='__main__':

    twith,vwith = readtxt(R"Withbias.txt")
    tfcn, vfcn = readtxt(R"FCN_lr0.0001_30_epoch.txt")
    tunet, vunet = readtxt(R"unet.txt")
    epoch = 30
    x = range(1,epoch+1)
    # plot
    plt.subplot(211)
    plt.plot(x, twith[:epoch],color="deeppink",linewidth=1,linestyle=':',label='UMnet', marker='.')
    plt.plot(x, tfcn[:epoch], color="darkblue",linewidth=1,linestyle='-.',label='FCN', marker='x')
    plt.plot(x, tunet[:epoch],color="goldenrod",linewidth=1,linestyle='-',label='Unet', marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.tight_layout()
    plt.xticks(range(1,epoch+1))
    plt.grid()

    plt.subplot(212)
    plt.plot(x, vwith[:epoch], color="deeppink", linewidth=1, linestyle=':', label='UMnet', marker='.')
    plt.plot(x, vfcn[:epoch], color="darkblue", linewidth=1, linestyle='-.', label='FCN', marker='x')
    plt.plot(x, vunet[:epoch], color="goldenrod", linewidth=1, linestyle='-', label='Unet', marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Validation loss')
    plt.legend()
    plt.tight_layout()
    plt.xticks(range(1, epoch+1))
    plt.grid()

    plt.show()