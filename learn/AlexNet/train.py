import os # 导入操作系统模块，用于路径操作等
import sys # 导入系统模块，用于系统相关操作，如标准输出
import json # 导入JSON模块，用于读写JSON格式的数据

import torch # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入神经网络模块，包含常用网络层和损失函数
from torchvision import transforms, datasets, utils # 导入torchvision中的数据变换、数据集和工具函数
import matplotlib.pyplot as plt # 导入matplotlib，用于数据可视化
import numpy as np # 导入numpy库，用于数值计算
import torch.optim as optim # 导入优化器模块，用于模型参数的更新
from tqdm import tqdm  # 导入tqdm模块，用于显示循环进度条

from model import AlexNet # 从model模块中导入AlexNet模型定义


def main():
    # 设置设备：若有GPU可用，则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 定义数据预处理操作：训练和验证阶段采用不同的预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224), # 随机裁剪为224×224大小
                                     transforms.RandomHorizontalFlip(), # 随机水平翻转
                                     transforms.ToTensor(), # 转换为Tensor张量对于图像数据，ToTensor()会将原始的(H, W, C)（高、宽、通道）格式转换为PyTorch的标准格式(C, H, W)（通道优先），与卷积层的输入维度一致。
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), # 归一化，均值为0.5
        "val": transforms.Compose([transforms.Resize((224, 224)), # 调整大小为224×224（注意：必须传入元组）
                                   transforms.ToTensor(), # 转换为Tensor张量
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])} # 归一化处理

    # 获取数据根目录：当前工作目录的上上级目录
    data_root = "D:\work\pycharm\pycharmproject\learn\AlexNet"  # get data root path
    # 构造花卉数据集的路径：在数据根目录下的data_set/flower_data
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # 断言数据路径存在，不存在则报错
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 加载训练数据集，使用ImageFolder读取文件夹数据，同时应用训练时的数据预处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset) # 获取训练集图片数量

    # 获取类别映射字典，例如：{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflower': 3, 'tulips': 4}
    # 当使用 ImageFolder 加载数据集时，PyTorch 会按以下规则自动生成 class_to_idx：扫描数据集的根目录（如 train/），找到所有子文件夹（每个子文件夹代表一个类别）。按字母顺序排序子文件夹（例如 daisy 会排在 dandelion 前面）。为每个类别分配一个唯一的整数索引，从 0 开始依次递增。
    flower_list = train_dataset.class_to_idx
    # 将字典的键值对反转，使得映射为 {0: 'daisy', 1: 'dandelion', ...}
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将类别字典写入JSON文件，便于后续使用
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 设置batch大小
    #batch size的定义是什么？在训练神经网络时，batch size指的是每次迭代中用于更新模型参数的样本数量。大的batch size可以带来更稳定的梯度估计，因为每次更新基于更多的数据，这可能加快训练速度，尤其是在使用GPU并行计算时，能够更充分地利用计算资源
    batch_size = 64
    # 计算dataloader的工作进程数量，取CPU核心数、batch_size（至少为1）和8中的最小值
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 创建训练数据的DataLoader，支持多线程加载数据，并随机打乱数据
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    # 加载验证数据集，使用ImageFolder读取验证集数据，应用验证时的数据预处理
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)# 获取验证集图片数量

    # 创建验证数据的DataLoader，batch大小为4，不打乱数据
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=32, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = next(test_data_iter)
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # 实例化AlexNet模型，设置类别数为5，并初始化权重
    net = AlexNet(num_classes=5, init_weights=True)

    # 将模型发送到设备上（GPU或CPU）
    net.to(device)
    # 定义交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    # 定义优化器，这里采用Adam优化器，学习率为0.0002
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # 设置训练轮数为10轮
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0# 记录最佳验证准确率，初始值为0
    train_steps = len(train_loader)# 计算每个epoch训练时的步数

    # 开始训练循环
    for epoch in range(epochs):
        # train
        net.train()# 设置模型为训练模式
        running_loss = 0.0# 初始化累计损失
        # 使用tqdm显示训练进度
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data# 获取当前批次的图像和标签
            optimizer.zero_grad()# 清空梯度
            outputs = net(images.to(device))# 将数据发送到设备上，并前向传播得到输出
            loss = loss_function(outputs, labels.to(device)) # 计算损失
            loss.backward()# 反向传播，计算梯度
            optimizer.step()# 更新模型参数

            # print statistics
            # 累加当前批次损失
            running_loss += loss.item()

            # 更新进度条描述，显示当前epoch和损失值
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate# 进入验证阶段
        net.eval()# 设置模型为评估模式
        acc = 0.0   # 初始化正确预测的数量
        with torch.no_grad(): # 验证时不计算梯度，节省内存和计算资源
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data# 获取验证批次数据
                outputs = net(val_images.to(device))# 模型预测
                predict_y = torch.max(outputs, dim=1)[1]# 获取预测结果（最大概率对应的索引）
                # 累加预测正确的数量
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # 计算本轮验证的准确率
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 如果当前验证准确率优于之前的最佳准确率，则保存模型参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':
    main()
