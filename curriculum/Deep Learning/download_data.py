import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义数据集存储路径
DATA_PATH = r"D:\Github\learning-repository\assets\data"

# 创建目录（如果不存在）
os.makedirs(DATA_PATH, exist_ok=True)

# 定义数据预处理（转换为 Tensor）
transform = transforms.ToTensor()

# 下载 Fashion-MNIST 数据集（训练集 & 测试集）
print("正在下载 Fashion-MNIST 数据集...")
datasets.FashionMNIST(root=DATA_PATH, train=True, transform=transform, download=True)
datasets.FashionMNIST(root=DATA_PATH, train=False, transform=transform, download=True)

print(f"数据集下载完成，存放于 {DATA_PATH}")
