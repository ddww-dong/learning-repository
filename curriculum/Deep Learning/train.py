import os

import torch
import torch.nn as nn
import torch.utils.data as Data

from model import AlexNet,CNNClassifier
from plot_loss import plot_loss_curve

# 设备选择（如果有 GPU 就用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径
PROCESSED_DATA_DIR = r"D:\Github\learning-repository\assets\data\FashionMNIST\processed"
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data.pth")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_data.pth")

# 加载数据
print("加载数据...")
train_data, train_labels = torch.load(TRAIN_DATA_PATH)
test_data, test_labels = torch.load(TEST_DATA_PATH)

# 创建 DataLoader
train_dataset = Data.TensorDataset(train_data, train_labels)
test_dataset = Data.TensorDataset(test_data, test_labels)

train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNNClassifier().to(device)
model_name = model.__class__.__name__

learning_rate = 1e-3
batch_size = 64
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(model, train_loader, loss_fn, optimizer, epochs):
    model.train()  # 确保模型处于训练模式（BatchNorm 和 Dropout 有影响）
    loss_pic = []

    for epoch in range(epochs):  # 训练多个 Epoch
        print(f"Epoch {epoch + 1}/{epochs} 开始训练...")
        total_loss = 0

        for batch, (X, y) in enumerate(train_loader):
            pred = model(X)
            loss = loss_fn(pred, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch + 1) % 100 == 0:
                print(f"  Batch {batch + 1}, loss: {loss.item():.6f}")
        avg_loss = total_loss / len(train_loader)  # 计算平均损失
        loss_pic.append(avg_loss)
        print(f"Epoch {epoch + 1} 完成，平均 Loss: {avg_loss:.6f}\n")

    plot_loss_curve(loss_pic, model_name)


train(model, train_loader, loss_fn, optimizer, epochs)

print("训练完成")

# 保存模型
MODEL_PATH = f"./models/fashion_mnist_{model_name}.pth"
os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"模型已保存至 {MODEL_PATH}")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

accuracy = correct / total * 100
print(f"测试准确率: {accuracy:.2f}%")
