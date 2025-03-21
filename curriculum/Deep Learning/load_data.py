import os
import torch
import torch.utils.data as Data


# 数据加载函数
def load_data(data_dir, batch_size=64):
    train_data, train_labels = torch.load(os.path.join(data_dir, "train_data.pth"))
    test_data, test_labels = torch.load(os.path.join(data_dir, "test_data.pth"))

    train_dataset = Data.TensorDataset(train_data, train_labels)
    test_dataset = Data.TensorDataset(test_data, test_labels)

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
