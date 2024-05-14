from scipy.ndimage.filters import gaussian_filter
import torch.utils.data
import scipy.io
import random
import numpy
import cv2
import os


class ShanghaiTech(torch.utils.data.Dataset):
    """ShanghaiTech数据集"""

    def __init__(self, typ='B', train=True, shuffle=True, transform=None):
        # 数据根目录
        root_path = f'dataset/part_{typ}_final/' + ('train' if train else 'test') + '_data/'
        # 图片目录
        self.img_root = os.path.join(root_path, 'images')
        # 初始化图片路径
        self.images_paths = []
        paths = os.listdir(self.img_root)
        if train:
            paths = paths * 2
        if shuffle:
            random.shuffle(paths)
        for path in paths:
            self.images_paths.append(os.path.join(self.img_root, path))
        # 初始化transform
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # 获取img和target路径
        img_path = self.images_paths[idx]
        target_path = img_path.replace('images', 'ground_truth').replace('.jpg', '.mat').replace('IMG', 'GT_IMG')
        # 加载img
        img = cv2.imread(img_path)
        h, w, channel = img.shape
        if self.transform:
            img = self.transform(img)
        # 加载target
        mat = scipy.io.loadmat(target_path)
        head_pos = mat['image_info'][0, 0][0, 0][0]
        k = numpy.zeros((h, w))
        for i in range(0, len(head_pos)):
            if int(head_pos[i][0]) < w and int(head_pos[i][1]) < h:
                k[int(head_pos[i][1]), int(head_pos[i][0])] = 1
        target = gaussian_filter(k, 10)
        target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
        return img, target
