# -*- coding: utf-8 -*-
__author__ = 'Vee'

"""
此文件用来指定一些固定配置
例如训练和测试的文件夹
"""

import warnings

class DefaultConfig(object):
    env = 'default'     # visdom 环境
    model = 'ResNet34'

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test/'  # 测试集存放路径
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128
    use_gpu = False
    num_workers = 4
    print_freq = 30     # print every N epoch

    debug_file = './debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95     # when val_loss increase, lr = lr*lr_decay

    weight_decay = 1e-4


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)
    #
    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    #     if not k.startswith('__'):
    #         print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()





