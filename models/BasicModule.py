# -*- coding: utf-8 -*-
__author__ = 'Vee'

import torch
import torch.nn as nn
import time

"""
对nn.Module的简单封装
"""

class BasicModule(nn.Module):
    """
    主要提供save和load方法
    """

    def __int__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        指定路径加载
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name


class Flat(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)

