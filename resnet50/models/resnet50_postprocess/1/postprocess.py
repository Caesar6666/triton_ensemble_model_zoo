import triton_python_backend_utils as pb_utils
import numpy as np
import os
import json


class TritonPythonModel:
    def initialize(self, args):
        
        # 获取当前模型（postprocess）的版本目录
        model_repository = args["model_repository"]  # 例如: /models/resnet50_postprocess
        model_version = args["model_version"]        # 例如: 1
        label_file = os.path.join(model_repository, model_version, "imagenet_classes.txt")

        # 加载ImageNet的1000个标签
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def execute(self, requests):
        responses = []
        for request in requests:
            # 1. 获取ONNX模型的输出logits
            logits_tensor = pb_utils.get_input_tensor_by_name(request, "LOGITS")
            logits = logits_tensor.as_numpy()  # shape: (batch_size, 1000)

            # 2. Softmax计算概率
            probs = self.softmax(logits)

            # 3. 获取Top-5的索引和对应标签
            top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
            batch_size = logits.shape[0]
            batch_results = []
            for i in range(batch_size):
                top5_labels = [self.labels[idx] for idx in top5_indices[i]]
                top5_probs = [float(probs[i][idx]) for idx in top5_indices[i]]
    
                classify_item = {
                    "top5_labels": top5_labels,
                    "top5_probs": top5_probs
                }
                result_json_str = json.dumps(classify_item)
                batch_results.append(result_json_str.encode('utf-8'))

            # 4. 输出最终结果（这里简单返回字符串和概率）
            output_array = np.array(batch_results, dtype=object)
            out_tensor = pb_utils.Tensor("OUTPUT_RESULTS", output_array)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses