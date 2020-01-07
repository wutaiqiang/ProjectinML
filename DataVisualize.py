
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

    txtdir = R"C:\Users\Taki5\Desktop\Withbias.txt"
    t,v = readtxt(txtdir)
    # plot
    plt.subplot(211)
    plt.plot(t)
    plt.subplot(212)
    plt.plot(v)
    plt.show()