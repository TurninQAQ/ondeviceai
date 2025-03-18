import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt

from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

#第一次下载时要将第一行和最后的if语句去掉注释
def main():

#compose是容器，将多个图像转换操作组合成一个序列。进行图像预处理时，通常需要一次执行多个操作借助Compose就能把这些操作组合起来，从而按顺序执行。transforms.ToTensor()是一个转换操作，其作用是把 PIL 图像（Python Imaging Library 图像）或者 NumPy 数组形式的图像转换为 PyTorch 的张量（Tensor）。在 PyTorch 里，模型输入一般要求是张量形式，所以该操作是图像预处理的常见步骤。
#transforms.Normalize用于对张量进行归一化处理。它接收两个参数：第一个参数(0.5, 0.5, 0.5)是一个包含三个元素的元组，分别代表图像三个通道（红、绿、蓝）的均值。第二个参数(0.5, 0.5, 0.5)同样是一个包含三个元素的元组，分别代表图像三个通道的标准差。
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集，第一个参数是下载位置，第二个是是否下载训练集，第三个是是否下载，第四个是图形处理函数（需要自己定义），pytorch用torchvision.datasets.这个函数准备了很多数据集，还有imagenet
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    #数据集分批次，是否打乱，num_workers 参数指定了用于数据加载的子进程数量。当 num_workers 设置为 0 时，表示数据加载将在主进程中进行，即不使用额外的子进程来并行加载数据。如果将 num_workers 设置为大于 0 的值，比如 num_workers=4，则会启动 4 个子进程来并行加载数据，这样可以加快数据加载的速度，特别是在处理大规模数据集时。
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #数据集中的图片展示
    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(val_image))
    # # print labels
    # print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(4)))


    net = LeNet()
#自动包含softmax函数
    loss_function = nn.CrossEntropyLoss()
#网络可训练的参数输入优化器中
    optimizer = optim.Adam(net.parameters(), lr=0.001)

#将训练集迭代多少轮
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
