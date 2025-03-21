import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from model import CNNClassifier, AlexNet

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(data_dir, batch_size=64):
    test_data, test_labels = torch.load(os.path.join(data_dir, "test_data.pth"))
    test_dataset = Data.TensorDataset(test_data, test_labels)
    return Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def load_model(model_class, model_path):
    model = model_class().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("模型已成功加载！")
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查路径！")
    return model


def evaluate(model_class, model_path, data_dir):
    test_loader = load_test_data(data_dir)
    model = load_model(model_class, model_path)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    correct, test_loss, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            test_loss += loss_fn(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            actual = labels.argmax(dim=1)
            correct += (predicted == actual).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    avg_test_loss = test_loss / len(test_loader)

    # 美化输出
    print("=" * 50)
    print(f"模型名称: {model_class.__name__}")
    print(f"测试准确率: {accuracy:.2f}%")
    print(f"测试集平均损失: {avg_test_loss:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    data_dir = r"D:\\Github\\learning-repository\\assets\\data\\FashionMNIST\\processed"
    model_path = "./models/fashion_mnist_CNNClassifier.pth"
    evaluate(CNNClassifier, model_path, data_dir)
