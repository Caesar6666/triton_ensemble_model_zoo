import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        """初始化：加载类别映射和阈值"""
        self.model_config = json.loads(args['model_config'])
        
        # 获取置信度阈值
        conf_thresh_str = self.model_config.get('parameters', {}).get('CONF_THRESHOLD', {}).get('string_value', '0.25')
        self.conf_threshold = float(conf_thresh_str)
        
        # COCO 80 类名称映射 (根据你的实际训练集修改)
        # 如果你的模型只训练了特定类别，请相应缩减此列表或修改索引逻辑
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def converted_origin_box(self, bbox, shape, scale, padding):
        """
        将 YOLOv26 模型输出的检测框还原到原始图像坐标系
        
        参数:
            bbox: list/tuple [x1, y1, x2, y2]，模型输出的坐标 (基于 640x640 输入图)
            shape: list/tuple [orig_h orig_w]，原始图像的宽高
            scale: float，预处理时的缩放比例 (r)
            padding: tuple/list (dw, dh)，预处理时的填充量 (左右填充，上下填充)
        
        返回:
            final_bbox: list [final_x1, final_y1, final_x2, final_y2]，原图上的坐标
        """
        x1, y1, x2, y2 = bbox
        orig_h, orig_w = shape
        
        # 解包填充量 (dw, dh)
        dw, dh = padding
        
        # --- 核心修正逻辑 ---
        
        # 1. 去除填充偏移 (Undo Padding)
        # 模型预测的坐标是包含灰边的，所以要减去左上角的填充量
        x1_adj = x1 - dw
        y1_adj = y1 - dh
        x2_adj = x2 - dw
        y2_adj = y2 - dh
        
        # 2. 逆向缩放 (Undo Scaling)
        # 将 640 图中的坐标乘以缩放比，还原到原图尺寸
        # 注意：这里是除法，不是乘法！
        final_x1 = x1_adj / scale
        final_y1 = y1_adj / scale
        final_x2 = x2_adj / scale
        final_y2 = y2_adj / scale
        
        # 3. 边界截断 (Clipping)
        # 防止因浮点误差导致坐标略微超出原图范围
        final_x1 = max(0.0, min(final_x1, float(orig_w)))
        final_y1 = max(0.0, min(final_y1, float(orig_h)))
        final_x2 = max(0.0, min(final_x2, float(orig_w)))
        final_y2 = max(0.0, min(final_y2, float(orig_h)))
    
        return [float(final_x1), float(final_y1), float(final_x2), float(final_y2)]

    def execute(self, requests):
        responses = []

        for request in requests:
            # 1. 获取模型原始输出 [Batch, 300, 6]
            det_tensor = pb_utils.get_input_tensor_by_name(request, "DETECTION_OUTPUT")
            if det_tensor is None:
                continue
            detections_raw = det_tensor.as_numpy()
            
            # 2. 获取原图尺寸 [Batch, 2] (H, W)
            shape_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_SHAPE")
            if shape_tensor is None:
                continue
            img_shapes = shape_tensor.as_numpy()

            # 3. 获取压缩图片比例 [Batch, 1]
            scales_tensor = pb_utils.get_input_tensor_by_name(request, "SCALE")
            if scales_tensor is None:
                continue
            scales = scales_tensor.as_numpy()

            # 4. 获取边缘填充 [Batch, 2] (pw, ph)
            paddings_tensor = pb_utils.get_input_tensor_by_name(request, "PADDING")
            if paddings_tensor is None:
                continue
            paddings = paddings_tensor.as_numpy()

            batch_results = []
            batch_size = detections_raw.shape[0]

            # 5. 逐帧处理
            for b in range(batch_size):
                boxes = detections_raw[b]  # Shape: [300, 6]
                
                valid_detections = []

                # 6. 遍历 300 个候选框
                # 格式：[x1, y1, x2, y2, score, class_id]
                # 注意：yolov26模型包含了nms，这里的 box 已经是去重过的，只需过滤分数
                for i in range(300):
                    x1, y1, x2, y2, score, class_id = boxes[i]

                    # 7. 置信度过滤
                    if score < self.conf_threshold:
                        continue
                    
                    # 8. 还原目标框
                    final_x1, final_y1, final_x2, final_y2 = self.converted_origin_box([x1,y1,x2,y2], img_shapes[b], scales[b][0], paddings[b])

                    # 9. 构建标准返回对象
                    # 只保留用户要求的三个核心字段
                    detection_item = {
                        "bbox": [final_x1, final_y1, final_x2, final_y2],
                        "label": self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f"class_{int(class_id)}",
                        "score": float(score)
                    }
                    valid_detections.append(detection_item)

                # 将当前图片的结果序列化为 JSON 字符串
                # 如果没有检测到物体，valid_detections 为空列表 []，json.dumps 后为 "[]"
                result_json_str = json.dumps(valid_detections)
                batch_results.append(result_json_str.encode('utf-8'))

            # 5. 构造输出 Tensor
            if len(batch_results) > 0:
                # Triton String Tensor 需要 object 类型的 numpy 数组
                output_array = np.array(batch_results, dtype=object)
                print(f"output shape: {output_array.shape}")
                out_tensor = pb_utils.Tensor("OUTPUT_RESULTS", output_array)
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)

        return responses