# -*- coding: utf-8 -*-
__author__ = 'Vee'


import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



import warnings
warnings.filterwarnings("ignore")

# 图片变化
data_transform = transforms.Compose([
    transforms.Resize(84),      # 变为 84 * 84
    transforms.CenterCrop(84),  # 从中心裁剪图片，长宽都是84
    transforms.ToTensor(),      # 读取图片像素，转换为 0～1 的数字
    # 读取像素并将数值转换为标准差和均值都为 0.5 的数据机
    # 数据由 0～1 变为 -1～1
    transforms.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
])

# 数据的加载
train_dataset = datasets.ImageFolder(
    'data/train/',
    transform=data_transform,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

test_dataset = datasets.ImageFolder(
    'data/test/',
    transform=data_transform,
)
test_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)


# 定义网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 18 * 18, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3

        return x

# 新建网络
net = Net()

# 开始训练

cirterion = nn.CrossEntropyLoss()   # 损失函数
# 使用随机梯度下降
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# 开始训练
print("start training!")
for epoch in range(3):
    runnging_loss = 0.0

    for i,data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()   # 梯度归零
        outputs = net(inputs)   # 输入数据
        loss = cirterion(outputs, labels)
        loss.backward()         # loss反向传播
        optimizer.step()

        runnging_loss += loss.data[0]

        if i % 2000 == 1999:
            print('[%d %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("finished training!")










