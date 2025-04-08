import os
import torch
import torch.nn as nn
from model import resnet34


def main():
    # 选择运行设备：如果有可用的 GPU（CUDA），就用第一个 GPU，否则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # ===== 方案一：先加载完整预训练模型，再替换最后的全连接层 =====
    net = resnet34()
    # 加载预训练权重到模型，map_location 确保权重加载到上面选定的 device
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu', weights_only=False))

    net = net.to(device)
    # 修改模型的全连接层，使其输出类别数变为 5
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5).to(device)

    # option2
    # net = resnet34(num_classes=5)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
