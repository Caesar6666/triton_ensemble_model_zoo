import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    def initialize(self, args):
        
        self.img_size = 640
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 255.0

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(shape[1] * scale), int(shape[0] * scale)
        
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, scale, (dw, dh)

    def resize_image(self, img):
        img_resized, scale, padding = self.letterbox(img, new_shape=(self.img_size, self.img_size))  # Letterbox 处理
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR to RGB
        img_resized = (img_resized - self.mean) * self.std  # Normalize
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
        return img_resized, scale, padding


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
            # 1. 获取输入张量，该张量包含一个批次
            in_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            if in_tensor is None:
                logger.error("Input tensor 'RAW_IMAGE' not found!")
                continue
            # raw_batch = pb_utils.deserialize_bytes_tensor(in_tensor.as_numpy())
            raw_batch = in_tensor.as_numpy()
            batch_size = raw_batch.shape[0]
            imgs_resized = []
            origins_shape = []
            scales = []
            paddings = []


            # 2. 处理批次中的每个图像
            for i in range(batch_size):
                base64_str = raw_batch[i][0]  # 获取第i个图像的字节
                # 3. 把 base64 转成 image
                img = self.base64_to_image(base64_str) 
                if img is None:
                    # 如果解码失败，可以插入一个黑色图像？或者报错？这里我们插入一个黑色图像
                    logger.info("Base64 converted to image failed!")
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    origins_shape.append([self.img_size, self.img_size])
                    scales.append(1)
                    paddings.append((0,0))
                    
                else:
                    logger.info("Base64 converted to image Successfully!")
                    orig_h, orig_w = img.shape[:2]
                    origins_shape.append([orig_h, orig_w])
                    # 4. 预处理
                    img_resized, scale, padding= self.resize_image(img)
                    imgs_resized.append(img_resized)
                    scales.append(scale)
                    paddings.append(padding)

            # 将列表转换为一个批次
            if len(imgs_resized) == 0:
                # 没有有效的图像，创建一个全零批次
                batch_imgs = np.zeros((batch_size, 3, self.img_size, self.img_size), dtype=np.float32)
                batch_shapes = np.zeros((batch_size, 2), dtype=np.int32)
                batch_scales = np.zeros((batch_size,1), dtype=np.float32)
                batch_paddings = np.zeros((batch_size, 2), dtype=np.float32)

            else:
                batch_imgs = np.stack(imgs_resized, axis=0)  # (batch_size, 3, 640, 640)
                batch_shapes = np.array(origins_shape, dtype=np.int32)  # (batch_size, 2)
                batch_scales = np.array(scales, dtype=np.float32).reshape(-1, 1)
                batch_paddings = np.array(paddings, dtype=np.float32)

            # 构造输出张量
            out_img = pb_utils.Tensor("PREPROCESSED_IMAGE", batch_imgs)
            out_shape = pb_utils.Tensor("IMAGE_SHAPE", batch_shapes)
            out_scale = pb_utils.Tensor("SCALE", batch_scales)
            out_padding = pb_utils.Tensor("PADDING", batch_paddings)

            response = pb_utils.InferenceResponse(output_tensors=[out_img, out_shape, out_scale, out_padding])
            responses.append(response)

        return responses
    