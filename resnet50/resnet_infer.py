import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple

class ResNet50:
    """
    使用 OpenCV 读取图像，ResNet-50 图像分类器（本地 .pth 权重）
    预处理、推理、后处理分别封装为独立方法
    """
    def __init__(self, weight_path: str):
        """
        初始化模型、预处理参数和类别标签
        
        Args:
            weight_path: 本地权重文件路径（如 'resnet50_imagenet_v2.pth'）
            device: 运行设备，'cuda' 或 'cpu'；若为 None 则自动检测
        """
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 检查权重文件
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        
        # 1. 加载模型结构（不加载预训练权重）
        self.model = resnet50(weights=None)
        
        # 2. 加载本地权重
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("模型加载成功。")
        
        # 3. 定义预处理参数（无变换对象，在 preprocess 中手动实现）
        self.resize_size = 256
        self.crop_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 4. 加载 ImageNet 类别标签
        try:
            self.categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
            print(f"已加载 {len(self.categories)} 个 ImageNet 类别。")
        except Exception as e:
            print(f"加载 torchvision 元数据失败: {e}")
            alt_path = "imagenet_classes.txt"
            if os.path.exists(alt_path):
                with open(alt_path, "r") as f:
                    self.categories = [line.strip() for line in f.readlines()]
                print(f"从 {alt_path} 加载了 {len(self.categories)} 个类别。")
            else:
                raise RuntimeError("无法加载类别标签，请确保 torchvision 可用或提供 imagenet_classes.txt")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        使用 OpenCV 读取图像，并转换为 RGB 格式
        
        Args:
            image_path: 图片路径
        
        Returns:
            RGB 格式的 numpy 数组，形状 (H, W, 3)
        """
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise RuntimeError(f"无法读取图片: {image_path}")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        对 RGB 图像进行预处理：缩放 → 中心裁剪 → 转为 Tensor → 归一化
        
        Args:
            image: RGB 格式的 numpy 数组，形状 (H, W, 3)
        
        Returns:
            预处理后的张量，形状 (1, 3, crop_size, crop_size)
        """
        # 使用 PIL 或直接使用 numpy 实现（此处使用 PIL 保持与 torchvision 一致）
        pil_img = Image.fromarray(image)
        
        # 缩放
        pil_img = transforms.Resize(self.resize_size)(pil_img)
        # 中心裁剪
        pil_img = transforms.CenterCrop(self.crop_size)(pil_img)
        # 转为 Tensor (自动将 HWC 转为 CHW，并归一化到 [0,1])
        tensor = transforms.ToTensor()(pil_img)
        # 归一化
        tensor = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        # 增加 batch 维度
        batch_tensor = tensor.unsqueeze(0)
        return batch_tensor.to(self.device)
    
    def preprocess_cv(self, image):
        # 1. 获取原图尺寸
        h, w = image.shape[:2]

        # 2. 短边缩放至 256
        if w < h:
            new_w = self.resize_size
            new_h = int(h * self.resize_size / w)
        else:
            new_h = self.resize_size
            new_w = int(w * self.resize_size / h)

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # (256, 256, 3)

        # 3. 中心裁剪 224x224
        start_x = (new_w - self.crop_size) // 2
        start_y = (new_h - self.crop_size) // 2
        img_cropped = img_resized[start_y:start_y + self.crop_size,
                                  start_x:start_x + self.crop_size]  # (224, 224, 3)

        # 4. 归一化到 [0,1] 并转为 float32
        img_norm = img_cropped.astype(np.float32) / 255.0

        # 5. 标准化
        img_resized = (img_norm - self.mean) / self.std

        # 6. 转化通道
        img_resized = np.transpose(img_resized, (2, 0, 1))   # (H,W,C) -> (C,H,W)
        img_resized = np.expand_dims(img_resized, axis=0) # [1, 3, 224, 224]
        return torch.from_numpy(img_resized)
    
    @torch.no_grad()
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        模型推理，输出 logits
        
        Args:
            input_tensor: 预处理后的张量，形状 (1, 3, H, W)
        
        Returns:
            logits 张量，形状 (1, 1000)
        """
        logits = self.model(input_tensor)
        return logits
    
    def postprocess(self, logits: torch.Tensor, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        后处理：对 logits 进行 softmax，提取 top_k 的类别和概率
        
        Args:
            logits: 模型输出，形状 (1, 1000)
            top_k: 返回前 k 个结果
        
        Returns:
            (labels, probabilities) 元组
            labels: 类别名称列表
            probabilities: 对应的置信度列表（0~1 浮点数）
        """
        probs = torch.nn.functional.softmax(logits, dim=1)  # (1, 1000)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()
        
        labels = [self.categories[idx] for idx in top_indices]
        probabilities = [float(p) for p in top_probs]
        return labels, probabilities
    
    def predict(self, image_path: str, top_k: int = 5, verbose: bool = True) -> Tuple[List[str], List[float]]:
        """
        完整预测流程：读取图片 → 预处理 → 推理 → 后处理
        
        Args:
            image_path: 图片路径
            top_k: 返回前 k 个结果
            verbose: 是否打印结果
        
        Returns:
            (labels, probabilities) 元组
        """
        # 1. 读取图像
        img = self.load_image(image_path)
        
        # 2. 预处理
        input_tensor_0 = self.preprocess(img)
        print(f'input_tensor_0: f{input_tensor_0.shape}')
        input_tensor = self.preprocess_cv(img)
        print(f'input_tensor: f{input_tensor.shape}')
        
        # 3. 推理
        logits = self.inference(input_tensor)
        
        # 4. 后处理
        labels, probs = self.postprocess(logits, top_k)
        
        # 5. 可选打印
        if verbose:
            print(f"\n图片: {image_path}")
            print(f"预测结果 (Top-{top_k}):")
            for i, (label, prob) in enumerate(zip(labels, probs)):
                print(f"  {i+1}. {label:30} ({prob*100:.2f}%)")
        
        return labels, probs


# ------------------------- 使用示例 -------------------------
if __name__ == "__main__":
    # 初始化分类器（请修改为实际权重路径）
    classifier = ResNet50(weight_path="weights/resnet50_imagenet_v2.pth")
    
    # 单张图片推理
    image_file = r"images/000001.jpg"
    labels, probs = classifier.predict(image_file, top_k=5, verbose=True)
    