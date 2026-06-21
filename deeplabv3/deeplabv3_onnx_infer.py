"""
ONNX 推理脚本（单张图片） - DeepLabV3 语义分割
- 使用 OpenCV 读取图片
- 使用 NumPy 进行预处理/后处理
- 输入尺寸：520x520
- 归一化：ImageNet 均值 [0.485,0.456,0.406]，标准差 [0.229,0.224,0.225]
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os

class DeepLabV3:
    def __init__(self, onnx_path) -> None:
        # ========== 配置参数 ==========
        self.input_height = 512
        self.input_width = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.session = ort.InferenceSession(onnx_path)

    def preprocess_image(self, image_bgr):
        """
        使用 OpenCV 和 NumPy 预处理图像
        Args:
            image_bgr: BGR 格式的图像 (H, W, 3)，dtype=uint8
        Returns:
            input_tensor: (1, 3, H, W) float32 numpy array，已归一化
        """
        # 1. BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize 到固定尺寸 (520, 520)，使用双线性插值
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # 3. 归一化：uint8 [0,255] -> float [0,1]
        img_float = resized.astype(np.float32) / 255.0
        
        # 4. 标准化 (ImageNet 统计)
        img_norm = (img_float - self.mean) / self.std
        
        # 5. HWC -> CHW 并添加 batch 维度
        img_chw = np.transpose(img_norm, (2, 0, 1))   # (3, H, W)
        input_tensor = np.expand_dims(img_chw, axis=0) # (1, 3, H, W)
        
        return input_tensor

    def postprocess_mask(self, output, original_size=None):
        """
        后处理：取 argmax，可选上采样回原始尺寸
        Args:
            output: (1, num_classes, H, W) numpy array
            original_size: (width, height) 原始图像尺寸
        Returns:
            mask: (H, W) 或 (orig_H, orig_W) numpy array，dtype=uint8
        """
        # 取类别索引
        mask = np.argmax(output, axis=1)  # (1, H, W)
        mask = mask[0].astype(np.uint8)   # (H, W)
        
        if original_size is not None:
            # 使用最近邻插值上采样，保持标签值
            orig_w, orig_h = original_size
            print(orig_w, orig_h)
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def inference(self, input_np):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: input_np})[0]  # (1, num_classes, H, W)
        return outputs

    def predict(self, image):
        # 预处理
        origin_height, origin_width = image.shape[:2]
        input_np = self.preprocess_image(image)
        print(f"输入形状: {input_np.shape}")
        
        # 推理
        print("推理中...")
        outputs = self.inference(input_np)
        print(f"输出形状: {outputs.shape}")
        
        # 后处理
        mask = self.postprocess_mask(outputs, original_size=(origin_width, origin_height))
        print(f"掩码形状: {mask.shape}, 类别范围: [{mask.min()}, {mask.max()}]")
        return mask

    def visualize_result(self, image_bgr, mask, save_path=None):
        """
        可视化：显示原始图像和分割掩码
        Args:
            image_bgr: 原始 BGR 图像 (H, W, 3)
            mask: 分割掩码 (H, W)
            save_path: 保存路径，若 None 则显示
        """
        # BGR -> RGB 用于 matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        im = axes[1].imshow(mask, cmap='tab20', alpha=0.8)
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Class Index')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果保存至: {save_path}")
        else:
            plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="ONNX 推理 (OpenCV + NumPy)")
    parser.add_argument("--model", type=str, default=r'weights\deeplabv3_resnet50_coco.onnx', help="ONNX 模型路径")
    parser.add_argument("--input", type=str, default=r'images/bus.jpg', help="输入图像路径")
    parser.add_argument("--save_dir", type=str, default=r'results', help="结果保存路径（如 result.png）")
    args = parser.parse_args()
    
    # 初始化deeplabv3
    print(f"加载模型: {args.model}")
    deeplabv3 = DeepLabV3(args.model)
    
    # 读取图片 (OpenCV 默认 BGR)
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {args.input}")
    original_size = (img_bgr.shape[1], img_bgr.shape[0])  # (width, height)
    print(f"原始图像尺寸: {original_size}")
    
    # 模型推理
    mask = deeplabv3.predict(img_bgr)
    print(list(mask.shape))
    
    # 可视化/保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, os.path.basename(args.input))
    deeplabv3.visualize_result(img_bgr, mask, save_path=save_path)
    print("推理完成！")

if __name__ == "__main__":
    main()