# -*- coding: utf-8 -*-
__author__ = 'Vee'

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):
    """
    获取图片地址，据此训练、验证、测试划分数据集
    """
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = False

        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # 区别测试集和训练集
        # if self.test:
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        # else:
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            # 只取 70% 的训练图片
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        # 对图片进行变换
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),       # 224 * 224
                    T.CenterCrop(224),  # 裁剪
                    T.ToTensor(),       # 变成 0～1
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
