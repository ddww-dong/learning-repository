## 数据集可以从官方导入
### 如TorchVision可以加载FasionMNIST数据集
```
from torch.utils.data import Dataset  
from torchvision import datasets  
from torchvision.transforms import ToTensor  

training_data = datasets.FashionMNIST(  
    root=r"D:\Github\learning-repository\assets\data", # 设置数据存储路径
    train=True,  
    download=True,  
    transform=ToTensor()  
)
```
## 创建自定义`Dataset`数据集时需要实现以下三个方法
\__init__、\__len__ 和 \__getitem__  
以下是具体实现
```
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

## 使用 `DataLoaders` 准备数据以进行训练
首先将数据集包装成一个可迭代对象
```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
```
其中，`batch_size`为每批次样本数量，`DataLoader`会根据`batch_size`把数据集分成若干个批次，__每一次对数据集的完整遍历为一个`epoch`__。

`shuffle`指每次加载数据时会随机打乱数据集顺序，这样有助于减少模型在训练时的偏差，避免模型记住数据的顺序特征。

接下来即可使用迭代器获取数据
```
for train_features, train_labels in train_dataloader:
    # 这里处理 train_features 和 train_labels
```
数据准备完成