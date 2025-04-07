import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
#用于显示循环进度条，便于监控训练过程
from tqdm import tqdm
#从自定义的模块 model 中导入 vgg 函数，该函数用于构造VGG模型
from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 定义数据预处理方式
    data_transform = {
        #对训练集进行裁剪、水平翻转、转换为张量、对张量进行归一化处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # 对训练集进行裁剪、水平翻转、转换为张量、对张量进行归一化处理
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 获取数据集根目录
    data_root = os.getcwd()  # get 当前目录
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 加载训练集，使用ImageFolder从指定路径读取数据，并应用预处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 获取训练集样本数量
    train_num = len(train_dataset)

    # 获取类别名称和对应的索引，例：{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 反转字典，使其变为 {0: 'daisy', 1: 'dandelion', ...}
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将类别字典转换为JSON字符串，便于保存查看
    json_str = json.dumps(cla_dict, indent=4)
    # 将类别字典写入JSON文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 每个批次的样本数
    batch_size = 32
    # 计算dataloader使用的子进程数，取CPU数量、批次大小和8中的最小值
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 构造训练数据的DataLoader，设置批次大小、是否打乱数据及工作进程数
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    # 加载验证集，同样使用ImageFolder读取数据，应用验证时的预处理方式
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # 获取验证集样本数量
    val_num = len(validate_dataset)


    # 构造验证数据的DataLoader
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 下面这两行代码被注释掉，可用于测试DataLoader
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # 指定使用的模型名称
    model_name = "vgg16"
    # 构造VGG模型，类别数设为5（对应5种花卉），并初始化权重
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #训练轮次
    epochs = 30
    #存放最佳验证准确率
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    # 每个epoch中的训练步数
    train_steps = len(train_loader)

    # 开始训练循环
    for epoch in range(epochs):
        # 将模型设置为训练模式
        net.train()
        # 记录累计损失
        running_loss = 0.0
        # 使用tqdm显示训练进度
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # 获取验证数据的图像和标签
            images, labels = data
            # 清空梯度，防止累积
            optimizer.zero_grad()
            # 将图像传入模型进行前向传播，计算输出
            outputs = net(images.to(device))
            # 计算损失
            loss = loss_function(outputs, labels.to(device))
            # 反向传播，计算梯度
            loss.backward()
            # 根据梯度更新参数
            optimizer.step()
            # 累计当前批次的损失
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 验证阶段，将模型设置为评估模式，不启用Dropout等训练时行为
        net.eval()
        # 累计正确预测的样本数量
        acc = 0.0
        # 关闭梯度计算，减少内存消耗，提高验证速度
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                # 获取验证数据的图像和标签
                val_images, val_labels = val_data
                # 模型预测输出
                outputs = net(val_images.to(device))
                # 取预测概率最大的类别索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 统计预测正确的样本数量
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # 计算本轮验证准确率
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 如果当前验证准确率超过之前的最佳准确率，则更新最佳准确率并保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
