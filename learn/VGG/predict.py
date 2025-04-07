import os
import json
import torch
from PIL import Image # 从PIL库中导入Image，用于图像加载和处理
from torchvision import transforms
import matplotlib.pyplot as plt # 导入matplotlib，用于图像显示

from model import vgg# 从model.py中导入vgg函数，用于构建VGG模型


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # 对图像进行预处理，转换为[N, C, H, W]格式
    img = data_transform(img)
    # 扩展batch维度，使图像shape变为[1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # 读取类别索引映射文件（class_indices.json）
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    # 创建模型：使用vgg16模型，并指定类别数为5（对应5种花卉）
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # 加载预训练模型权重
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 设置模型为评估模式，关闭Dropout等训练专用层
    model.eval()
    # 在验证/推理时关闭梯度计算，节省内存和加速计算
    with torch.no_grad():
        # squeeze去除多余的维度，并将结果移动到CPU
        output = torch.squeeze(model(img.to(device))).cpu()
        # 对输出应用softmax，得到每个类别的概率分布
        predict = torch.softmax(output, dim=0)
        # 获取概率最大的类别索引
        predict_cla = torch.argmax(predict).numpy()
    # 打印预测结果，包含类别名称和对应的概率
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    # 遍历每个类别的预测概率，并打印出来
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    # 显示图像窗口
    plt.show()


if __name__ == '__main__':
    main()
