import requests
import numpy as np
import os
import cv2
import json
import time
import base64
import argparse
import matplotlib.pyplot as plt

def base64_to_image(base64_str):
    try:
        raw_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img

    except Exception as error:
        print(f"Error: {error}")
        print(f"error line: {error.__traceback__.tb_lineno}")

def image_to_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    data_list = np.frombuffer(image_bytes, dtype=np.uint8).tolist()
    return data_list

def image_to_base64(image_path):
    # 1. 以二进制方式打开并读取图片
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
    
    # 2. 进行 Base64 编码 (返回的是 bytes 类型)
    encoded_bytes = base64.b64encode(binary_data)
    
    # 3. 转换为 UTF-8 字符串 (方便放入 JSON)
    encoded_str = encoded_bytes.decode('utf-8')
    
    return encoded_str


class HTTP_CLINET:
    def __init__(self, http_url='http://localhost:8000', model='resnet50_ensemble'):

        self.url = f"{http_url}/v2/models/{model}/infer"

    def post(self, image_base64):

        payload = {
            "inputs": [
                {
                    "name": "RAW_IMAGE",  # 必须与 preprocess/config.pbtxt 中的 input name 一致
                    "shape": [1, 1],
                    "datatype": "BYTES",  # 对应 preprocess/config.pbtxt 中的 data_type=TYPE_STRING
                    "data": [image_base64]
                }
            ]
        }

        # headers = {
        #         "Content-Type": "application/json",
        #         # "Authorization": f"Bearer {self.api_key}"  # 注意 Bearer 后面有一个空格
        #     }


        # 发送 POST 请求
        print(f"正在发送请求到 {self.url} ...")
        start_time = time.time()
        # response = requests.post(self.url, json=payload, headers=headers, timeout=30)
        response = requests.post(self.url, json=payload, timeout=30)
        end_time = time.time()
        print(f"请求耗时: {(end_time - start_time)*1000:.2f} ms")

        if response.status_code != 200:
            print(f"请求失败! 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return

        # 解析响应
        print(f"请求成功! 状态码: {response.status_code}")
        result_json = response.json()
        # print(result_json)
        output_json = result_json['outputs'][0]['data'][0]
        # print(output_json)
        output_result = json.loads(output_json)
        # print(output_result)
        # print(type(output_result))

        return output_result
    
    def draw_yolo(self, det_results, image_path, save_path):
        img = cv2.imread(image_path)
    
        for det_box in det_results:
            bbox = det_box['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
            label = det_box['label']
            score = det_box['score']
            color = ([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img,'%s:%.2f'%(label,score), (x1, y1), cv2.FONT_ITALIC, 1, color, 1)
        cv2.imwrite(save_path, img)
        print(f"draw result has saved in {save_path}")
    
    def draw_maskrcnn(self, det_results, image_path, save_path):
        try:
            image = cv2.imread(image_path)
            boxes = det_results['boxes']
            labels = det_results['labels']
            scores = det_results['scores']
            masks = det_results['masks']
            num_instances = len(boxes)
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(num_instances, 3)).tolist()
            overlay = np.zeros_like(image, dtype=np.float32)
            
            for idx in range(num_instances):
                x1, y1, x2, y2 = boxes[idx]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = labels[idx]
                score = float(scores[idx])
                color = colors[idx]
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {score:.2f}"
                cv2.putText(image, text, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                mask = masks[idx]
                mask_binary = base64_to_image(mask)
                # print(type(mask_binary), mask_binary.shape)
                # 关键修复：确保 mask 是 2D bool 类型
                mask_binary_2d = mask_binary.squeeze().astype(bool)
                
                # 使用 .any() 检查是否有 True 值
                if mask_binary_2d.any():
                    color_layer = np.zeros_like(image, dtype=np.float32)
                    # 直接使用 bool 数组索引
                    color_layer[mask_binary_2d] = color
                    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.7, 0)
            
            result = cv2.addWeighted(image, 0.5, overlay.astype(np.uint8), 0.5, 0)
            cv2.imwrite(save_path, result)
            print(f"可视化结果已保存至: {save_path}")
            
        except Exception as error:
            print(f"Error: {error}")
            print(f"error line: {error.__traceback__.tb_lineno}")
    
    def draw_deeplabv3(self, det_results, image_path, save_path=None):
        """
        可视化：显示原始图像和分割掩码
        Args:
            image_path: 图片路径
            mask: 分割掩码 (H, W)
            save_path: 保存路径，若 None 则显示
        """
        
        # BGR -> RGB 用于 matplotlib
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        mask_base64 = det_results["mask"]
        mask = base64_to_image(mask_base64)
        im = axes[1].imshow(mask, cmap='tab20', alpha=0.8)
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Class Index')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果保存至: {save_path}")
    
def parse_opt():
    parser = argparse.ArgumentParser(description="http client")

    parser.add_argument('--http', '-u',
                        type=str,
                        required=False,
                        default='http://localhost:8000',
                        help="http_url")
    
    parser.add_argument('--model', '-m',
                        type=str,
                        required=True,
                        default="yolov26_ensenble",
                        help='triton ensemble name')
    
    parser.add_argument('--input', '-i',
                        type=str,
                        required=False,
                        default="datasets/images/bus.jpg",
                        help="image path")
    
    parser.add_argument('--output_dir', '-o',
                        type=str,
                        required=False,
                        default="results",
                        help="output floder")
    
    parser.add_argument('--draw', '-d',
                        action='store_true',
                        help="draw result")
    return parser.parse_args()

def main(args):
    try:
        image_path = args.input
        if not os.path.exists(image_path):
            print(f"{image_path} is not exsit")
            return
        
        print(f"正在加载图片: {args.input} ...")
        image_base64 = image_to_base64(image_path)
        client = HTTP_CLINET(http_url=args.http, model=args.model)
        output = client.post(image_base64)
        print(output)

        if args.draw:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
            if args.model == 'yolov26_ensemble':
                client.draw_yolo(det_results=output, image_path=image_path, save_path=save_path)
            elif args.model == 'maskrcnn_ensemble':
                client.draw_maskrcnn(det_results=output, image_path=image_path, save_path=save_path)
            elif args.model == 'deeplabv3_ensemble':
                client.draw_deeplabv3(det_results=output, image_path=image_path, save_path=save_path)

    except Exception as error:
        print(f"Error: {error}")
        print(f"error line: {error.__traceback__.tb_lineno}")


if __name__ == "__main__":
    args = parse_opt()
    main(args)
    