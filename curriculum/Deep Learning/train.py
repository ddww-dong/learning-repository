import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from model import AlexNet, CNNClassifier
from plot_loss import plot_loss_curve
from load_data import load_data
from evaluate import evaluate
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练函数
def train(model, train_loader, loss_fn, optimizer, epochs):
    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch + 1}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} 完成，平均 Loss: {avg_loss:.6f}\n")

    plot_loss_curve(loss_history, model.__class__.__name__)


# 主函数
def main():
    data_dir = r"D:\\Github\\learning-repository\\assets\\data\\FashionMNIST\\processed"
    train_loader, test_loader = load_data(data_dir)

    model = CNNClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, loss_fn, optimizer, epochs=10)

    model_path = "./models/fashion_mnist.pth"
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")

    evaluate(CNNClassifier, model, test_loader)


if __name__ == "__main__":
    main()
