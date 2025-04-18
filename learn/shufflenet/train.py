import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # 用于可视化训练过程
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x1_0# 导入自定义的模型结构
from my_dataset import MyDataSet# 导入自定义的数据集类
from utils import read_split_data, train_one_epoch, evaluate# 工具函数


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()# 创建 TensorBoard 记录器，输出在 runs 文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights") # 如果不存在 weights 目录，就创建一个用于保存模型权重

    # 读取训练集和验证集图像路径及对应标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 定义训练与验证过程中的图像预处理操作
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),# 将PIL图像或numpy数组转换为PyTorch张量，同时会自动将数值范围从[0,255]转换到[0,1]
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    batch_size = args.batch_size
    # 根据 CPU 数量和 batch size 决定 num_workers 数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,#数据加载的子进程数量（根据CPU核心数设置）
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)# 自定义的批处理函数（collate function）

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)

    # 加载预训练权重（如果指定）
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 只加载参数数量匹配的部分
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    # 优化器设置，只对可训练参数进行优化
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 使用余弦退火学习率衰减
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:\work\pycharm\pycharmproject\learn\data\\flower_photos\\flower_photos")

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
