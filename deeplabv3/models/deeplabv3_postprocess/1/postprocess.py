import triton_python_backend_utils as pb_utils
import numpy as np
import base64
import json
import cv2


class TritonPythonModel:
    def initialize(self, args):
        pass

    def mask_to_base64(self, mask):
        """
        input:
            mask: [h, w], 像素值在[0, 21]
        output:
            base64_encoding：base64
        """
        mask_image = mask[:, :, np.newaxis]
        success, buffer = cv2.imencode('.png', mask_image)
        if not success:
            raise ValueError("图像编码失败")
        base64_mask = base64.b64encode(buffer).decode('utf-8')
        return base64_mask
        
    def postprocess_mask(self, output, original_size=None):
        """
        后处理：取 argmax，可选上采样回原始尺寸
        Args:
            output: (num_classes, H, W) numpy array
            original_size: (width, height) 原始图像尺寸
        Returns:
            mask: (H, W) 或 (orig_H, orig_W) numpy array，dtype=uint8
        """
        # 取类别索引
        mask = np.argmax(output, axis=0)  # (H, W)
        mask = mask.astype(np.uint8)   # (H, W)
        
        if original_size is not None:
            orig_h, orig_w = original_size
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return mask

    def execute(self, requests):
        responses = []
        for request in requests:
            masks_tensor = pb_utils.get_input_tensor_by_name(request, "MASKS")
            masks = masks_tensor.as_numpy()

            shape_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_SHAPE")
            origins_shape = shape_tensor.as_numpy()

            batch_size = masks.shape[0]
            batch_results = []
            for i in range(batch_size):
                mask = self.postprocess_mask(masks[i], origins_shape[i])
    
                mask_item = {
                    "mask": self.mask_to_base64(mask),
                    "mask_shape": list(mask.shape),
                    "min_class": int(mask.min()),
                    "max_class": int(mask.max()),
                    # "origin shape:": list(origins_shape[i])
                }
                result_json_str = json.dumps(mask_item)
                batch_results.append(result_json_str.encode('utf-8'))

            output_array = np.array(batch_results, dtype=object)
            out_tensor = pb_utils.Tensor("OUTPUT_RESULTS", output_array)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses