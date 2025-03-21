import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import Lambda

RAW_DATA_PATH = r"D:\Github\learning-repository\assets\data"
PROCESSED_DATA_PATH = r"D:\Github\learning-repository\assets\data\FashionMNIST\processed"

# 确保目录存在
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
])
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

# 加载数据集时直接应用转换
train_dataset = datasets.FashionMNIST(
    root=RAW_DATA_PATH, train=True, download=True, transform=transform, target_transform=target_transform
)
test_dataset = datasets.FashionMNIST(
    root=RAW_DATA_PATH, train=False, download=True, transform=transform, target_transform=target_transform
)


# 预处理数据并保存
def preprocess_and_save(dataset, save_path):
    data = torch.cat([img.unsqueeze(0) for img, _ in dataset])  # 直接转换为 Tensor
    labels = torch.stack([label for _, label in dataset])  # 直接获取 label
    torch.save((data, labels), save_path)
    print(f"预处理数据已保存至 {save_path}")


preprocess_and_save(train_dataset, os.path.join(PROCESSED_DATA_PATH, "train_data.pth"))
preprocess_and_save(test_dataset, os.path.join(PROCESSED_DATA_PATH, "test_data.pth"))
