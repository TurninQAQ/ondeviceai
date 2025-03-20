from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    保证输出的通道数能被 divisor 整除，
    这样在实际硬件实现时更友好（例如内存对齐）。
    """
    if min_ch is None:
        min_ch = divisor
    # 先将通道数加上 divisor/2 后整除，再乘以 divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 如果舍入后下降超过 10%，则增加一个 divisor
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#因为整个mobilenetv2中用到了很多3x3卷积层、BN、relu6的组合，所以创建一个类整合起来。（深度卷积）
#nn.Sequential 允许你将多个神经网络层按顺序组合在一起，形成一个新的模块。在进行前向传播时，输入数据会依次通过这些层，最终得到输出结果。
class ConvBNReLU(nn.Sequential):
# __init__是类的构造函数，当创建ConvBNReLU类的实例时会自动调用\in_channel：输入特征图的通道数\out_channel：输出特征图的通道数。\kernel_size：卷积核的大小，默认值为 3。\stride：卷积操作的步长，默认值为 1。\groups：分组卷积的组数，默认值为 1。
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
#这里计算了卷积层的填充值。通过(kernel_size - 1) // 2的计算方式，可以确保在使用奇数大小的卷积核时，输入和输出的特征图尺寸保持一致。
        padding = (kernel_size - 1) // 2
#调用父类nn.Sequential的构造函数
        super(ConvBNReLU, self).__init__(
#二维卷积层，用于对输入特征图进行卷积操作。bias=False表示不使用偏置项。
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
#二维批量归一化层，用于加速网络收敛并提高模型的稳定性。
            nn.BatchNorm2d(out_channel),
#修正线性单元激活函数，其输出范围被限制在[0, 6]之间。inplace=True表示直接在原地修改输入，以节省内存。
            nn.ReLU6(inplace=True)
        )

#倒转残差块
#nn.Module 是 PyTorch 深度学习框架里一个极为关键的类，处于 torch.nn 模块中。该类是所有神经网络模块的基类，借助继承它，能够构建自定义的神经网络模块。
class InvertedResidual(nn.Module):
#__init__ 是类的构造函数，in_channel：输入特征图的通道数，out_channel：输出特征图的通道数，stride：卷积操作的步长，用于控制特征图的尺寸变化，expand_ratio：扩展比例，用于确定中间特征图的通道数
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
#调用父类 nn.Module 的构造函数，确保正确初始化
        super(InvertedResidual, self).__init__()
#计算中间特征图的通道数，即通过输入通道数乘以扩展比例得到。
        hidden_channel = in_channel * expand_ratio
#判断是否使用捷径连接（shortcut connection）。当步长为 1 且输入通道数等于输出通道数时，才使用捷径连接，捷径连接可以让网络学习到残差信息，有助于训练。
        self.use_shortcut = stride == 1 and in_channel == out_channel

#初始化一个空列表，用于存储网络层
        layers = []
#如果扩展比例不等于 1，说明需要进行通道扩展，添加一个 1x1 的逐点卷积层（Pointwise Convolution），这里使用 ConvBNReLU 类将卷积层、批量归一化层和激活函数层组合在一起。
        if expand_ratio != 1:
# 1x1 pointwise conv append函数是想列表中添加一个项，构建点卷积
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
#添加后续的层，expend连续添加函数
        layers.extend([
# 3x3 depthwise conv 深度卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
# 1x1 pointwise conv(linear) 点卷积使用了线性激活函数
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
#批量归一化层，用于加速网络收敛和提高稳定性。批量归一化是一种用于深度神经网络的技术，其主要目的是加速网络的训练过程，提高模型的稳定性和泛化能力。通过对每个小批量（mini-batch）的数据进行归一化处理，使得网络各层输入数据的分布更加稳定，从而减少了内部协变量偏移，缓解梯度消失或梯度爆炸，加快收敛速度。（Internal Covariate Shift）问题，缓解梯度消失或梯度爆炸，加快收敛速度。
            nn.BatchNorm2d(out_channel),
        ])

#使用 nn.Sequential 将 layers 列表中的网络层按顺序封装成一个新的模块，方便后续进行前向传播。
        self.conv = nn.Sequential(*layers)

#定义前向传播方法，当调用 InvertedResidual 类的实例时，会自动调用该方法。
    def forward(self, x):
#如果满足使用捷径连接的条件，将输入 x 与经过 self.conv 模块处理后的结果相加，实现shortcut。
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

#MobileNetV2 继承自 nn.Module，是 PyTorch 神经网络模块
class MobileNetV2(nn.Module):
#num_classes：分类任务的类别数，默认 1000、alpha：通道缩放因子（0 < α ≤ 1），用于调整网络宽度，减少计算量。、round_nearest：通道数需为 round_nearest 的整数倍（如 8），优化硬件兼容性。
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
# 调用父类 nn.Module 的构造函数，确保正确初始化
        super(MobileNetV2, self).__init__()
#指向 InvertedResidual 类，即倒残差块
        block = InvertedResidual
#初始卷积层的输出通道数，通过 _make_divisible 调整
        input_channel = _make_divisible(32 * alpha, round_nearest)
#最后一个卷积层的输出通道数，通常为 1280（调整后）
        last_channel = _make_divisible(1280 * alpha, round_nearest)
#网络结构配置
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
#features 是一个空列表，用于存储特征提取层
        features = []
# conv1 layer第一层卷积，升维，输入通道3，输入通道作为后续卷积层的输入层，步长为2
        features.append(ConvBNReLU(3, input_channel, stride=2))
#构造残差块
        for t, c, n, s in inverted_residual_setting:
#使输出通道满足参数要求
            output_channel = _make_divisible(c * alpha, round_nearest)
#n 是 inverted_residual_setting 中当前子列表指定的倒残差块重复次数。循环会执行 n 次，也就是会添加 n 个倒残差块到 features 列表
            for i in range(n):
#此语句用于确定当前倒残差块的步长。当 i 等于 0 时，也就是在第一次循环时，步长采用 s（即 inverted_residual_setting 中指定的步长）；其他情况下，步长设为 1。这种设置的目的在于，只有第一个倒残差块可能会改变特征图的尺寸（通过非 1 的步长），后续的倒残差块保持特征图尺寸不变。
                stride = s if i == 0 else 1
#把一个 InvertedResidual （倒残差块）模块添加到 features 列表中，input_channel：当前倒残差块的输入通道数，output_channel：当前倒残差块的输出通道数，stride：当前倒残差块的步长，expand_ratio=t：倒残差块的扩展比例，对应 inverted_residual_setting 中的 t。
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#这行代码将当前倒残差块的输出通道数赋值给 input_channel。这样做的原因是，下一个倒残差块的输入通道数就是当前倒残差块的输出通道数，保证了通道数在不同倒残差块之间的连续性。
                input_channel = output_channel
# 最后一层
        features.append(ConvBNReLU(input_channel, last_channel, 1))
#self.features = nn.Sequential(*features) 的主要作用是将之前存储在 features 列表中的多个神经网络层或模块按顺序组合成一个新的序列化模块 self.features。
        self.features = nn.Sequential(*features)

#经过前面一系列的卷积层和倒残差块后，会得到一个具有多个通道的特征图。通过 self.avgpool 进行全局平均池化，将特征图转化为一个一维向量，这个向量包含了每个通道的全局特征信息，方便后续进行分类任务。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
#这是一个 Dropout 层，0.2 表示在训练过程中，每个神经元有 20% 的概率被随机置为 0。Dropout 是一种正则化技术，它可以防止模型过拟合。在训练时，随机丢弃一些神经元的输出，使得模型不能过度依赖某些特定的神经元，从而迫使模型学习更具泛化能力的特征表示。在测试阶段，Dropout 层不发挥作用，所有神经元都会正常工作。
            nn.Dropout(0.2),
#这是一个全连接层，也称为线性层。last_channel 是前面特征提取部分最后一层的输出通道数，经过全局平均池化后，会得到一个长度为 last_channel 的一维向量。num_classes 表示分类任务的类别数量。该全连接层的作用是将输入的特征向量映射到 num_classes 个类别上，通过学习权重矩阵，为每个类别分配一个得分，最终可以通过 Softmax 函数将这些得分转化为概率分布，从而确定输入图像所属的类别。
            nn.Linear(last_channel, num_classes)
        )

# 这段代码的作用是对 MobileNetV2 模型中的不同类型的层进行权重初始化操作。合适的权重初始化方法对于神经网络的训练至关重要，它能够帮助模型更快地收敛，避免梯度消失或梯度爆炸等问题。
        for m in self.modules():
#用于判断当前子模块 m 是否为二维卷积层（nn.Conv2d）。如果是，则执行以下的初始化操作
            if isinstance(m, nn.Conv2d):
#nn.init.kaiming_normal_ 是 PyTorch 提供的 Kaiming 初始化方法，也称为 He 初始化。这种初始化方法能够在使用 ReLU 激活函数的神经网络中，保持每层输入的方差在传播过程中大致不变，从而缓解梯度消失或梯度爆炸问题。mode='fan_out' 表示根据输出通道数来计算缩放因子，确保输出的方差在传播过程中保持稳定。这里将卷积层 m 的权重使用 Kaiming 正态分布进行初始化。
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
#判断卷积层 m 是否存在偏置项。有些卷积层在定义时可能会将 bias 参数设置为 False，即不使用偏置项。
                if m.bias is not None:
#如果卷积层存在偏置项，则将偏置项初始化为零。
                    nn.init.zeros_(m.bias)
#如果当前子模块 m 不是卷积层，而是二维批量归一化层（nn.BatchNorm2d），则执行以下的初始化操作。
            elif isinstance(m, nn.BatchNorm2d):
#将批量归一化层的权重初始化为 1。在批量归一化层中，权重（也称为缩放因子）通常初始化为 1，这样在训练初期，批量归一化层的作用相当于恒等映射。
                nn.init.ones_(m.weight)
#将批量归一化层的偏置初始化为 0。偏置（也称为偏移因子）初始化为 0 可以确保在训练初期，批量归一化层不会对输入数据进行额外的偏移。
                nn.init.zeros_(m.bias)
#如果当前子模块 m 既不是卷积层也不是批量归一化层，而是全连接层（nn.Linear），则执行以下的初始化操作。
            elif isinstance(m, nn.Linear):
#nn.init.normal_ 是 PyTorch 提供的正态分布初始化方法。这里将全连接层的权重初始化为均值为 0，标准差为 0.01 的正态分布。
                nn.init.normal_(m.weight, 0, 0.01)
#将全连接层的偏置初始化为 0。
                nn.init.zeros_(m.bias)

#这段代码定义了 MobileNetV2 模型的前向传播过程。前向传播是指输入数据在神经网络中从输入层依次经过各个隐藏层，最终到达输出层得到预测结果的过程。
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
#torch.flatten 是 PyTorch 提供的用于展平张量的函数。
#1 表示从第 1 个维度开始展平（索引从 0 开始），也就是保留第 0 个维度（batch_size），将其余维度展平成一维向量。经过这一步操作，特征张量 x 的形状变为 (batch_size, channels)，方便后续输入到全连接层进行分类。
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
