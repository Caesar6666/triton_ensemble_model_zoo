import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        # 预处理参数
        self.resize_size = 256
        self.crop_size = 224
        # ImageNet 均值与标准差 (RGB 顺序)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def img_preprocess(self, img) -> np.ndarray:
        
        # 1. 获取原图尺寸
        h, w = img.shape[:2]

        # 2. 短边缩放至 256
        if w < h:
            new_w = self.resize_size
            new_h = int(h * self.resize_size / w)
        else:
            new_h = self.resize_size
            new_w = int(w * self.resize_size / h)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # (256, 256, 3)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # BGR to RGB

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

        return img_resized.astype(np.float32)

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
            if in_tensor is None:
                logger.error("Input tensor 'RAW_IMAGE' not found!")
                continue
            raw_batch = in_tensor.as_numpy()
            batch_size = raw_batch.shape[0]
            imgs_resized = []
            
            # 2. 处理批次中的每个图像
            for i in range(batch_size):
                base64_str = raw_batch[i][0]  # 获取第i个图像的字节
                # 3. 把 base64 转成 image
                img = self.base64_to_image(base64_str)
                if img is None:
                    # 如果解码失败，可以插入一个黑色图像？或者报错？这里我们插入一个黑色图像
                    logger.info("Base64 converted to image failed!")
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    logger.info("Base64 converted to image Successfully!")
                    # 3. 图像预处理
                    img_resized = self.img_preprocess(img)
                    imgs_resized.append(img_resized)
            if len(imgs_resized)==0:
                # 如果没有图像，创建一个0批次
                batch_imgs = np.zeros((batch_size, 3, self.img_size, self.img_size), dtype=np.float32)
            else:
                batch_imgs = np.stack(imgs_resized, axis=0)  # (batch, 3, 224, 224)

            # 4. 构建输出张量
            out_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", batch_imgs)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses