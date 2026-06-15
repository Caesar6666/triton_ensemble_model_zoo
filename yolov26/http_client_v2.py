import requests
import numpy as np
import cv2
import json
import time
import base64
import os


class HTTP_CLINET:
    def __init__(self):
        
        http_url = 'http://localhost:8000'
        
        model_name = 'yolov26_ensemble'
        self.url = f"{http_url}/v2/models/{model_name}/infer"

        self.save_dir = r'results/http_client'
        os.makedirs(self.save_dir, exist_ok=True)

    
    def draw(self, image_path, det_results):
        img = cv2.imread(image_path)
    
        for det_box in det_results:
            bbox = det_box['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
            label = det_box['label']
            score = det_box['score']
            color = ([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img,'%s:%.2f'%(label,score), (x1, y1), cv2.FONT_ITALIC, 1, color, 1)
        
        save_path = os.path.join(self.save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img)
            

    def image_to_bytes(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        data_list = np.frombuffer(image_bytes, dtype=np.uint8).tolist()
        return data_list

    def image_to_base64(self, image_path):
        # 1. 以二进制方式打开并读取图片
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
        
        # 2. 进行 Base64 编码 (返回的是 bytes 类型)
        encoded_bytes = base64.b64encode(binary_data)
        
        # 3. 转换为 UTF-8 字符串 (方便放入 JSON)
        encoded_str = encoded_bytes.decode('utf-8')
        
        return encoded_str

    def post(self, image_path):
        # read image
        print(f"正在加载图片: {image_path} ...")
        encoded_data = self.image_to_base64(image_path)
        # encoded_data = self.image_to_bytes(image_path)
        

        payload = {
            "inputs": [
                {
                    "name": "RAW_IMAGE",  # 必须与 preprocess/config.pbtxt 中的 input name 一致
                    "shape": [1, 1],
                    "datatype": "BYTES",  # 对应 preprocess/config.pbtxt 中的 data_type=TYPE_STRING
                    "data": [encoded_data]
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
        print(output_result)
        # print(type(output_result))

        return output_result


if __name__ == "__main__":
    image_path = r"images/bus.jpg"
    # image_path = r"ultralytics\assets\bus.jpg"
    client = HTTP_CLINET()
    det_results = client.post(image_path)
    client.draw(image_path, det_results)
