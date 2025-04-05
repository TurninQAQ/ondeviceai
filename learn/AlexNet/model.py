import torch.nn as nn # 导入torch.nn模块，并简写为nn，用于构建神经网络层
import torch # 导入torch模块，用于张量操作和其他相关操作

# 定义AlexNet类，继承自nn.Module，这是所有神经网络模块的基类
class AlexNet(nn.Module):
    # 构造函数，参数num_classes指定分类的类别数，init_weights控制是否初始化权重
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()# 调用父类构造函数进行必要的初始化
        # 定义特征提取部分，使用nn.Sequential将多个层顺序组合起来
        self.features = nn.Sequential(
            # 第一层卷积：输入通道数3，输出通道数48，卷积核大小11，步幅4，填充2
            # 输入尺寸[3, 224, 224]，输出尺寸计算后为[48, 55, 55]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),# 使用ReLU激活函数，提高非线性；inplace=True表示直接修改输入，节省内存
            # 最大池化层：池化核大小3，步幅2，将尺寸从[48, 55, 55]减小到[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]

            # 第二层卷积：输入通道48，输出通道128，卷积核大小5，填充2
            # 输出尺寸为[128, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            # 第二个池化层：池化核大小3，步幅2，尺寸从[128, 27, 27]减小到[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]

            # 第三层卷积：输入通道128，输出通道192，卷积核大小3，填充1
            # 输出尺寸保持为[192, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),

            # 第四层卷积：输入通道192，输出通道192，卷积核大小3，填充1
            # 输出尺寸仍为[192, 13, 13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),

            # 第五层卷积：输入通道192，输出通道128，卷积核大小3，填充1
            # 输出尺寸为[128, 13, 13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),

            # 第三个池化层：池化核大小3，步幅2，将尺寸减小到[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        # 定义分类器部分，用于将特征映射转换为类别预测
        self.classifier = nn.Sequential(
            # Dropout层，防止过拟合，随机丢弃50%的神经元
            nn.Dropout(p=0.5),
            # 全连接层，将128*6*6的特征映射展平后映射到2048个神经元
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 第二个全连接层，将2048个神经元映射到2048个神经元
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 最后一层全连接层，将2048个神经元映射到num_classes个输出，对应分类数
            nn.Linear(2048, num_classes),
        )
        # 如果初始化权重标志为True，则调用权重初始化函数
        if init_weights:
            self._initialize_weights()

    # 前向传播函数，定义输入x经过网络各层的计算过程
    def forward(self, x):
        x = self.features(x)# 输入先经过特征提取部分
        x = torch.flatten(x, start_dim=1)# 将多维特征展平，从第1维开始展平（保留batch维度）
        x = self.classifier(x)# 展平后的特征送入分类器部分得到输出
        return x

    # 权重初始化函数，遍历网络中的所有模块并进行相应的初始化
    def _initialize_weights(self):
        for m in self.modules():
            # 如果当前模块是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用kaiming_normal_方法初始化卷积核权重，适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果卷积层有偏置项，则初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果当前模块是全连接层
            elif isinstance(m, nn.Linear):
                # 用均值为0、标准差为0.01的正态分布初始化权重
                nn.init.normal_(m.weight, 0, 0.01)
                # 将偏置初始化为0
                nn.init.constant_(m.bias, 0)
