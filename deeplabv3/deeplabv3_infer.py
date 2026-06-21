import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# ========== 1. 下载并加载预训练模型 ==========
def load_model(model_name="deeplabv3_resnet50", device="cuda"):
    """加载 TorchVision 语义分割预训练模型，自动下载权重
    
    Args:
        model_name: 模型名称，可选 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large', 
                    'fcn_resnet50', 'fcn_resnet101'
        device: 计算设备
    
    Returns:
        model: 加载好的模型
        transforms: 模型对应的预处理转换
    """
    # 定义可用模型及其对应的权重
    models_dict = {
        "deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
        "deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
        "fcn_resnet50": torchvision.models.segmentation.fcn_resnet50,
        "fcn_resnet101": torchvision.models.segmentation.fcn_resnet101
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_dict.keys())}")
    
    # 使用预训练权重，首次运行会自动下载
    # torchvision.models.segmentation 中的预训练模型权重基于 COCO 数据集训练，
    # progress=True 会显示下载进度条
    model = models_dict[model_name](pretrained=True, progress=True)
    
    # 设置为评估模式
    model.eval()
    
    # 移动到 GPU（如果可用）
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device


# ========== 2. 预处理函数 ==========
def get_preprocess_transform():
    """获取 TorchVision 语义分割模型的预处理转换
    
    注意事项：
        所有预训练模型都期望输入图像经过相同的归一化处理。
        图像必须先缩放到 [0, 1] 范围，然后使用指定的均值和标准差进行标准化。
        训练时输入图像的最小尺寸被调整为 520 像素，因此推理时也建议统一缩放。
    """
    return T.Compose([
        T.ToTensor(),                    # 将 PIL Image 或 numpy.ndarray 转换为 tensor，并缩放到 [0.0, 1.0]
        T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 归一化均值
                    std=[0.229, 0.224, 0.225])    # ImageNet 归一化标准差
    ])


def preprocess_image(image, target_size=(520, 520)):
    """对输入图像进行预处理
    
    Args:
        image: PIL Image 或 tensor，输入图像
        target_size: 目标尺寸 (height, width)
        
    Returns:
        batch: 预处理后的批处理张量 (1, 3, H, W)
    """
    # 调整图像大小，确保输入分辨率一致
    transform_resize = T.Compose([
        T.Resize(target_size),          # 将图像缩放到目标尺寸
    ])
    resized_image = transform_resize(image)
    
    # 应用归一化转换
    transform = get_preprocess_transform()
    input_tensor = transform(resized_image)
    
    # 添加 batch 维度 (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0)
    
    return input_batch


# ========== 3. 后处理函数 ==========
def postprocess_segmentation(output, original_image_size=None):
    """对模型输出进行后处理
    
    Args:
        output: 模型输出张量 (1, C, H, W)，C 是类别数（包含背景类）
        original_image_size: 原始图像的尺寸 (width, height)，如果需要恢复原图大小则提供
        
    Returns:
        mask: 分割掩码，每个像素对应的类别索引 (H, W)
    """
    # 获取预测结果：取每个像素位置的最大 logits 对应的类别
    # output['out'] 的形状为 (batch_size, num_classes, height, width)
    output_tensor = output['out']
    
    # 在类别维度上取 argmax，得到每个像素的预测类别索引
    mask = output_tensor.argmax(dim=1)  # 形状: (1, H, W)
    
    # 去掉 batch 维度
    mask = mask.squeeze(0).cpu().numpy()  # 形状: (H, W)
    
    # 如果需要恢复原始图像大小，进行上采样
    if original_image_size is not None:
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mask = torch.nn.functional.interpolate(mask, size=original_image_size[::-1], mode='nearest')
        mask = mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    
    return mask


# ========== 4. 可视化函数 ==========
def visualize_segmentation(image, mask, class_names=None, save_path=None):
    """可视化原始图像和分割结果
    
    Args:
        image: 原始 PIL Image
        mask: 分割掩码，每个像素对应的类别索引
        class_names: 类别名称列表，用于图例显示
        save_path: 如果指定，则保存图像到文件
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 显示分割结果
    # 使用 colormap 将类别索引映射到颜色
    im = axes[1].imshow(mask, cmap='tab20', alpha=0.8)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1], label='Class Index')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    plt.show()


# ========== 5. 主推理流程 ==========
def main():
    print("="*50)
    print("TorchVision 语义分割推理脚本")
    print("="*50)
    
    # 1. 加载模型
    print("\n[1/5] 正在加载预训练模型...")
    model, device = load_model(model_name="deeplabv3_resnet50")
    print(f"模型加载完成，使用设备: {device}")
    
    # 2. 加载图像（示例：从 URL 读取，也可以替换为本地文件路径）
    print("\n[2/5] 正在加载图像...")
    # 使用示例图像（可选：替换为你的本地图像路径）
    # image = Image.open("your_image.jpg").convert("RGB")
    url = "images/bus.jpg"
    image = Image.open(url).convert("RGB")
    # response = requests.get(url)
    # image = Image.open(BytesIO(response.content)).convert("RGB")
    original_size = image.size  # (width, height)
    print(f"图像尺寸: {original_size}")
    
    # 3. 预处理
    print("\n[3/5] 正在预处理图像...")
    input_batch = preprocess_image(image)
    
    # 4. 模型推理
    print("\n[4/5] 正在进行推理...")
    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)
    
    # 5. 后处理与可视化
    print("\n[5/5] 正在后处理和可视化结果...")
    mask = postprocess_segmentation(output, original_image_size=original_size)
    print(f"分割掩码形状: {mask.shape}, 类别范围: [{mask.min()}, {mask.max()}]")
    
    # 可视化
    visualize_segmentation(image, mask, save_path="segmentation_result.png")
    
    print("\n推理完成！")


if __name__ == "__main__":
    main()