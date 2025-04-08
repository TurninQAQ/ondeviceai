import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    #expansion 用于表示输出通道数扩展倍数，本模块中保持不变为 1
    expansion = 1


    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        #定义第1个3×3卷积层，输入通道为 in_channel，输出通道为 out_channel，kernel_size=3, stride为传入参数，padding=1 保持尺寸不变，且不使用偏置
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        # 1st卷积后接 BatchNorm 归一化，通常用在卷积层和relu之间，用来加速收敛的
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # 定义第2个3×3卷积层，输入输出通道均为 out_channel，stride固定为1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 若输入与输出维度不匹配，则通过 downsample 进行调整
        self.downsample = downsample

    def forward(self, x):
        # 保存输入 x 作为残差连接的“旁路”（shortcut）
        identity = x
        # 若需要下采样，则对 identity 进行变换
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    # 扩展系数设置为4，输出通道为基本通道数的4倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # 根据 groups 和 width_per_group 计算瓶颈层的中间输出通道宽度
        width = int(out_channel * (width_per_group / 64.)) * groups

        # 第1个1x1卷积层，用于将输入通道数压缩到 width
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)

        # 第2个3x3卷积层，分组卷积实现，输入输出均为 width，
        # 使用传入的 stride（可能为2实现下采样），padding=1保持尺寸
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        # -----------------------------------------
        # 第3个1x1卷积层，将通道数扩展至 out_channel*expansion
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,# 指定构建网络时使用的残差块类型（BasicBlock 或 Bottleneck）
                 blocks_num, # 每个阶段（layer）中残差块的数量列表，如 [3, 4, 6, 3]
                 num_classes=1000,# 分类任务类别数，默认适用于 ImageNet
                 include_top=True,# 是否包含最后的全连接分类层
                 groups=1, # 分组数，用于分组卷积（支持 ResNeXt 结构）
                 width_per_group=64):# 每个分组的基本通道数
        super(ResNet, self).__init__()
        # 保存是否包含 top 层的标志
        self.include_top = include_top
        # 网络最开始的输出通道数为64
        self.in_channel = 64

        # 保存分组参数，后面构造残差块时需要
        self.groups = groups
        self.width_per_group = width_per_group

        # 第一个卷积层，输入为 RGB 图像（3通道），使用7x7大卷积核，stride=2，padding=3 保持特征图尺寸
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，用于进一步下采样，kernel_size=3, stride=2, padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 依次构建后续的四个阶段，每个阶段由多个残差块组成
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)


        if self.include_top:
            # 若包含 top 分类部分，则使用自适应平均池化将特征图大小调整为 1x1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接层，将特征向量映射到类别数（乘以 block.expansion 考虑了扩展倍数）
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 如果步长不为1或输入输出通道数不匹配，则需要通过下采样层进行映射
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                # 使用 1x1 卷积调整通道数和空间尺寸
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 第一块残差块，可能需要进行下采样处理
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        # 更新 in_channel 为当前阶段输出通道（注意乘以 expansion）
        self.in_channel = channel * block.expansion

        # 构建剩余 block_num-1 个相同尺寸的残差块，不再需要下采样
        #没有写步长默认为1
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 将残差块列表组装成 nn.Sequential 模块返回
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
