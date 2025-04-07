import torch.nn as nn
import torch

# 官方预训练模型的权重地址字典
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=False):
        # 调用父类nn.Module的构造函数，完成模块的初始化
        super(VGG, self).__init__()
        # 保存卷积层和池化层等特征提取部分（通常由make_features生成）
        self.features = features
        #构建分类器，VGG网络最后的全连接部分
        self.classifier = nn.Sequential(
            # 将特征图展平后输入全连接层，输入维度为512*7*7，输出4096
            nn.Linear(512*7*7, 4096),
            # 激活函数：ReLU，inplace操作，节省内存
            nn.ReLU(True),
            #神经元随机失活，防止过拟合
            nn.Dropout(p=0.5),
            # 第二个全连接层，将4096维映射到4096维
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # 最后一个全连接层，将4096维映射到类别数（默认1000，用于ImageNet分类）
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            # 如果init_weights为True，则初始化所有层的权重
            self._initialize_weights()

    # N x 3 x 224 x 224：输入张量，N表示批次大小，3为通道数（RGB图像），224x224为图像尺寸
    def forward(self, x):
        # 经过特征提取部分，输出大小通常为 N x 512 x 7 x 7
        x = self.features(x)
        # 将除批次维度外的所有维度展平，转化为二维张量
        x = torch.flatten(x, start_dim=1)
        # 将展平后的特征送入分类器部分进行全连接计算
        x = self.classifier(x)
        return x

    #初始化网络权重函数
    def _initialize_weights(self):
        #遍历整个神经网络层
        for m in self.modules():
            #如果是卷积网络
            if isinstance(m, nn.Conv2d):
                # 采用Xavier均匀分布初始化
                nn.init.xavier_uniform_(m.weight)
                #如果存在偏差项
                if m.bias is not None:
                    # 则将其初始化为0
                    nn.init.constant_(m.bias, 0)
            #如果是全连接层
            elif isinstance(m, nn.Linear):
                #对全连接层权重同样采用Xavier均匀分布初始化
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 全连接层的偏置项初始化为0
                nn.init.constant_(m.bias, 0)

#定义特征提取层
def make_features(cfg: list):
    # 用于存储所有的层
    layers = []
    # 初始输入通道为3（RGB图像）
    in_channels = 3

    for v in cfg:
        # 若配置项为"M"，则添加池化层：采用核大小为2、步幅为2的最大池化层
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 若配置项为整数，表示卷积层输出通道数
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 添加卷积层和ReLU激活层（inplace操作）
            layers += [conv2d, nn.ReLU(True)]
            # 更新下一层的输入通道数为当前卷积层的输出通道数
            in_channels = v
    return nn.Sequential(*layers)

#不同大小模型的字典
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)

    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
