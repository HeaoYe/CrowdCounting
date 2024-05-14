# CrowdCounting
基于CSRNet的人群密度估计 PyTorch实现

可以得到图片或视频的人群密度估计

# 一、环境
Windows系统
python3.8
- pytorch
- torchvision
- numpy
- scipy
- tqdm
- opencv-python
- pygame>=2.0

# 二、运行

## 1 命令行运行
查看帮助
```
python main.py -h
```
## 2 图形化界面运行
```
python window.py
```
拖入图片或视频，点击Start

# 三、训练
1.解压缩dataset/dataset.tar到dataset文件夹下

2.运行python train.py

3.模型会保存在pth_save文件夹下 

4.修改main.py第18行的模型文件为自己训练的模型文件
