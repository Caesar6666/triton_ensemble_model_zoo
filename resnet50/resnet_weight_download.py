import os
import argparse
import torch
import torchvision.models as models

def save_resnet50_weights(save_dir='weights', weight_name="resnet50_imagenet_v1.pth"):
    """
    下载 ResNet-50 的 ImageNet 预训练权重，并保存到指定目录。

    Args:
        save_dir (str): 权重保存目录的路径
    """
    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 定义保存路径
    save_path = os.path.join(save_dir, weight_name)

    # 检查是否已存在，避免重复下载
    if os.path.exists(save_path):
        print(f"文件已存在: {save_path}")
        return

    print("正在下载 ResNet-50 预训练权重 (IMAGENET1K_V2) ...")
    # 使用 torchvision 官方权重（V2 版本准确率更高）
    # weights 参数会自动下载并缓存，但我们单独保存到指定目录
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 提取 state_dict 并保存
    torch.save(model.state_dict(), save_path)
    print(f"权重已保存至: {save_path}")


if __name__ == "__main__":
    save_resnet50_weights()
