import torch
import os
# 自动选择设备 (优先 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch 使用的设备类型: {device}")

# 如果有 GPU，打印详细名称
if device.type == "cuda":
    print(f"GPU 名称: {torch.cuda.get_device_name(device.index)}")

nw = min([os.cpu_count(), 32 if 32 > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))