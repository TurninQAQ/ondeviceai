import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    # 如果有GPU（CUDA）可用，则设置设备为"cuda:0"，否则使用"cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        # 将图像调整为224×224的大小
        [transforms.Resize((224, 224)),
         # 将图像转换为Tensor（张量）
         transforms.ToTensor(),
         # 对图像的每个通道进行归一化，均值和标准差均设为0.5
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./tulip.jpg"
    # 断言图像路径存在；如果不存在则报错并输出提示信息
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # 使用PIL打开图像
    img = Image.open(img_path)
    # 使用matplotlib显示原始图像
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)

    # 增加一个批次维度，使图像张量的维度变为 [1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # 以只读方式打开JSON文件，并将其内容加载为字典，存储类别索引
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # 加载模型权重
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    # 将模型设置为评估模式，此时会关闭Dropout等仅在训练时启用的层
    model.eval()
    # 在不计算梯度的上下文中进行推理，加快运算并节省内存
    with torch.no_grad():
        # 将预处理后的图像送入模型，得到输出结果
        # 使用torch.squeeze移除输出中多余的维度，并将结果从GPU转移到CPU
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # 找到概率最大的类别索引，并转换为numpy格式
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
