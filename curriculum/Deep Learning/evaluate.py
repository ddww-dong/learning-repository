import os

import torch
import torch.nn as nn
import torch.utils.data as Data

from model import AlexNet,CNNClassifier

# 设备选择（如果有 GPU 就用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DATA_DIR = r"D:\Github\learning-repository\assets\data\FashionMNIST\processed"

TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_data.pth")
test_data, test_labels = torch.load(TEST_DATA_PATH)
test_dataset = Data.TensorDataset(test_data, test_labels)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNNClassifier().to(device)

# 模型加载路径
MODEL_PATH = f"./models/fashion_mnist_CNNClassifier.pth"
# 加载模型权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))  # 加载权重到模型
    print("模型已成功加载！")
else:
    print("模型文件不存在，请检查路径！")

# 测试模型的损失
model.eval()
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    correct = 0
    test_loss = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # 计算损失
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()

        # 计算预测正确的数量
        predicted = outputs.argmax(dim=1)  # 预测类别索引
        actual = labels.argmax(dim=1)  # 真实类别索引（适用于 one-hot 编码）
        correct += (predicted == actual).sum().item()  # 计算正确预测数

        total += labels.size(0)  # 计算总样本数

    accuracy =  correct / total * 100
    print(f"测试准确率: {accuracy:.2f}%")

avg_test_loss = test_loss / len(test_loader)
print(f"测试集平均损失: {avg_test_loss:.4f}")
