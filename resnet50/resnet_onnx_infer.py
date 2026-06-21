import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple
import os

# 导入标签（如果 torchvision 可用，否则使用备选文件）
try:
    from torchvision.models import ResNet50_Weights
    _categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
except ImportError:
    _categories = None

class ResNet50:
    """
    使用 ONNX Runtime 进行 ResNet-50 推理（使用 OpenCV 读取图像）
    预处理、推理、后处理分别封装为独立方法
    """
    def __init__(self, weight_path: str, device: str = None):
        """
        初始化 ONNX 推理会话、预处理参数和类别标签
        
        Args:
            weight_path: ONNX 模型文件路径（如 'resnet50_imagenet_v2.onnx'）
            device: 运行设备，'cuda' 或 'cpu'；若为 None 则自动检测（优先 CUDA）
        """
        # 确定推理提供者
        if device is None:
            # 尝试 CUDA 是否可用（通过 onnxruntime 检测）
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        print(f"使用设备: {self.device}")
        
        # 检查权重文件
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {weight_path}")
        
        # 设置 ONNX Runtime 后端
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight_path, providers=providers)
        print("ONNX 模型加载成功。")
        
        # 获取输入输出信息（用于调试）
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 224
            self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 224
        else:
            self.input_height, self.input_width = 224, 224
        print(f"模型期望输入尺寸: {self.input_height}×{self.input_width}")
        
        # 定义预处理参数（与训练时一致）
        self.resize_size = 256
        self.crop_size = self.input_height   # 使用模型实际输入尺寸
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # 构建预处理流水线（输出 PIL Image 的 Tensor，后续转换为 numpy）
        self.preprocess_pipeline = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 加载 ImageNet 类别标签
        global _categories
        if _categories is not None:
            self.categories = _categories
            print(f"已加载 {len(self.categories)} 个 ImageNet 类别。")
        else:
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
        return img_resized.astype(np.float32)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        对 RGB 图像进行预处理：缩放 → 中心裁剪 → 转为 Tensor → 归一化
        
        Args:
            image: RGB 格式的 numpy 数组，形状 (H, W, 3)
        
        Returns:
            预处理后的 numpy 数组，形状 (1, 3, crop_size, crop_size)，dtype=float32
        """
        # 将 numpy 转为 PIL Image
        pil_img = Image.fromarray(image)
        # 应用预处理流水线（输出 torch.Tensor）
        tensor = self.preprocess_pipeline(pil_img)   # shape: (3, crop_size, crop_size)
        # 增加 batch 维度并转换为 numpy
        batch_numpy = tensor.unsqueeze(0).numpy()    # shape: (1, 3, crop_size, crop_size)
        return batch_numpy.astype(np.float32)
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        ONNX Runtime 推理，输出 logits
        
        Args:
            input_tensor: 预处理后的 numpy 数组，形状 (1, 3, H, W)
        
        Returns:
            logits 数组，形状 (1, 1000)，dtype=float32
        """
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]   # shape: (1, 1000)
        return logits
    
    def postprocess(self, logits: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
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
        # 使用 numpy 实现 softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)   # (1, 1000)
        probs = probs[0]   # (1000,)
        
        # 获取 top_k 索引
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]
        
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
        input_tensor = self.preprocess_cv(img)
        
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
    # 初始化分类器（请修改为实际 ONNX 模型路径）
    classifier = ResNet50(weight_path="weights/resnet50_imagenet_v2.onnx")
    
    # 单张图片推理
    # image_file = r"images/000003.jpg"
    image_file = r'../yolov26/images/bus.jpg'
    labels, probs = classifier.predict(image_file, top_k=5, verbose=True)
    