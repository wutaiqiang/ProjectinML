# ProjectinML

## 数据准备
制作./data_train,./data_test两个文件夹
将文件夹命名成FZMC开头
将四个阶段的.nii以及label.nii放在其中

## Train
设置好参数 直接运行train_**.py即可

## Test
设置好参数 直接运行Test_**.py即可

### 注：
将plt.show()改成plt.pause(2)即可自动切图

## 其他文件
DataVisualize.py 是将loss可视化的程序
test_all.py 是将三个模型的输出一起plot出来