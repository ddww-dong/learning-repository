import os

import matplotlib.pyplot as plt


def plot_loss_curve(loss_pic, model_name, save_dir="./picture"):
    """
    绘制训练损失曲线并保存

    :param loss_pic: 训练损失列表
    :param model_name: 模型名称（用于文件命名）
    :param save_dir: 图片保存目录，默认 './picture'
    """
    plt.figure(figsize=(8, 6))  # 设置图片大小

    # 绘制损失曲线
    plt.plot(loss_pic, label="Training Loss", color='b', linestyle='-', linewidth=1)

    # 添加标题和标签
    plt.title(f"Loss Curve of {model_name}", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    # 添加网格
    plt.grid(True, linestyle="--", alpha=0.6)

    # 添加图例
    plt.legend()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构造文件保存路径
    save_path = os.path.join(save_dir, f"{model_name}_loss.png")

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图片（可选）

    loss_data_path = os.path.join(save_dir, f"{model_name}_loss_data.txt")
    with open(loss_data_path, 'w') as f:
        for epoch, loss in enumerate(loss_pic):
            f.write(f"Epoch {epoch + 1}: Loss = {loss:.6f}\n")

    print(f"Loss curve saved at: {save_path}")
