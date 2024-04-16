import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入通道为1，输出通道为32，卷积核大小为3x3，步幅为1，填充为1
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        # 第二个卷积层：输入通道为32，输出通道为64，卷积核大小为3x3，步幅为1，填充为1
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # 第三个卷积层：输入通道为64，输出通道为128，卷积核大小为3x3，步幅为1，填充为1
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # 第四个卷积层：输入通道为128，输出通道为256，卷积核大小为3x3，步幅为1，填充为1
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # 第五个卷积层：输入通道为256，输出通道为512，卷积核大小为3x3，步幅为1，填充为1
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # 第六个卷积层：输入通道为512，输出通道为1024，卷积核大小为3x3，步幅为1，填充为1
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        # 最大池化层：池化窗口大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：输入特征数量为1024*7*7，输出特征数量为1024
        self.fc1 = nn.Linear(1024 * 7 * 7, 1024)
        # 全连接层：输入特征数量为1024，输出特征数量为10（假设是10分类任务）
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # 第一次卷积和ReLU
        x = torch.relu(self.conv1(x))
        # 第二次卷积和ReLU
        x = torch.relu(self.conv2(x))
        # 第一次池化
        x = self.pool(x)
        # 第三次卷积和ReLU
        x = torch.relu(self.conv3(x))
        # 第四次卷积和ReLU
        x = torch.relu(self.conv4(x))
        # 第二次池化
        x = self.pool(x)
        # 第五次卷积和ReLU
        x = torch.relu(self.conv5(x))
        # 第六次卷积和ReLU
        x = torch.relu(self.conv6(x))
        # 第三次池化
        x = self.pool(x)
        # 将特征图展平为一维向量
        x = x.view(-1, 1024 * 7 * 7)
        # 全连接层和ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 最终的全连接层
        x = self.fc2(x)
        return x

# 创建模型实例
model = CNN()
print(model)
