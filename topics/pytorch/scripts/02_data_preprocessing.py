import os

import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root=r"D:\Github\learning-repository\assets\data", # 设置数据存储路径
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# 下载并加载测试集
test_data = datasets.FashionMNIST(
    root=r"D:\Github\learning-repository\assets\data", # 设置数据存储路径
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

#可视化
# for train_features, train_labels in train_dataloader:
#         # 这里处理 train_features 和 train_labels
#         print(f"Feature batch shape: {train_features.size()}")
#         print(f"Labels batch shape: {train_labels.size()}")
#         img = train_features[0].squeeze()
#         label = train_labels[0].item()
#         plt.imshow(img, cmap="gray")
#         plt.show()
#         print(f"Label: {label}")

