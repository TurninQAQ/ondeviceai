from PIL import Image # 导入 PIL 库中的 Image，用于读取和处理图像
import torch
from torch.utils.data import Dataset# 从 torch.utils.data 中导入 Dataset 基类


class MyDataSet(Dataset):
    """自定义数据集"""

    # 初始化方法，构造数据集实例时调用
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path    # images_path: 存放所有图像文件路径的列表
        self.images_class = images_class    # images_class: 存放对应图像类别标签的列表
        self.transform = transform    # transform: 可选的图像预处理/增强操作（如 torchvision.transforms）

    def __len__(self):# 返回数据集的大小（样本总数）
        return len(self.images_path)

    def __getitem__(self, item):# 根据索引 item 获取一个样本，供 DataLoader 调用
        img = Image.open(self.images_path[item])# 通过 PIL 打开对应路径的图像文件
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None: # 如果用户提供了 transform 操作
            img = self.transform(img)# 则对图像进行预处理/增强

        return img, label

    @staticmethod
    def collate_fn(batch):# 自定义的 batch 拼接函数，用于 DataLoader 中的 collate_fn 参数
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))# batch 是一个列表，每个元素是 (img, label) 对；
        images = torch.stack(images, dim=0) # 将图像列表沿第 0 维拼接成一个四维张量 (batch_size, C, H, W)
        labels = torch.as_tensor(labels)# 将标签列表转换为张量，类型默认为 torch.int64
        return images, labels
