import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        self.input_height = 512
        self.input_width = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def img_preprocess(self, image_bgr) -> np.ndarray:
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
        input_tensor = np.transpose(img_norm, (2, 0, 1))   # (3, H, W)
        
        return input_tensor

    def base64_to_image(self, base64_str):
        try:
            raw_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img

        except Exception as error:
            logger.error(f"Error: {error}")
            logger.error(f"error line: {error.__traceback__.tb_lineno}")

    def execute(self, requests):
        responses = []
        for request in requests:
            # 1. 获取原始图像数据 (base64), 是一个批次
            in_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            raw_batch = in_tensor.as_numpy()
            batch_size = raw_batch.shape[0]
            imgs_resized = []
            origins_shape = []
            
            # 2. 处理批次中的每个图像
            for i in range(batch_size):
                base64_str = raw_batch[i][0]  # 获取第i个图像的字节
                # 3. 把 base64 转成 image
                img = self.base64_to_image(base64_str)
                if img is None:
                    # 如果解码失败，可以插入一个黑色图像？或者报错？这里我们插入一个黑色图像
                    logger.info("Base64 converted to image failed!")
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    origins_shape.append([self.input_width, self.input_height])
                else:
                    logger.info("Base64 converted to image Successfully!")
                    orig_h, orig_w = img.shape[:2]
                    origins_shape.append([orig_h, orig_w])
                    # 4. 图像预处理
                    img_resized = self.img_preprocess(img)
                    imgs_resized.append(img_resized)
            if len(imgs_resized)==0:
                # 如果没有图像，创建一个0批次
                batch_imgs = np.zeros((batch_size, 3, self.input_width, self.input_height), dtype=np.float32)
                batch_shapes = np.zeros((batch_size, 2), dtype=np.int64)
            else:
                batch_imgs = np.stack(imgs_resized, axis=0)  # (batch, 3, 520, 520)
                batch_shapes = np.array(origins_shape, dtype=np.int64)  # (batch_size, 2)

            # 5. 构建输出张量
            out_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", batch_imgs)
            out_shape = pb_utils.Tensor("IMAGE_SHAPE", batch_shapes)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor, out_shape])
            responses.append(response)
        return responses