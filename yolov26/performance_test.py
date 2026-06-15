import requests
import numpy as np
import cv2
import json
import time
import base64
import os
import concurrent.futures
import statistics
import random
import csv
from tqdm import tqdm

class HTTP_CLIENT:
    def __init__(self, http_url='http://localhost:8000', model_name='yolov26_ensemble'):
        self.url = f"{http_url}/v2/models/{model_name}/infer"
        self.save_dir = r'results/http_client'
        os.makedirs(self.save_dir, exist_ok=True)
    
    def draw(self, image_path, det_results):
        """绘制检测结果（仅用于单次验证）"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return
        for det_box in det_results:
            bbox = det_box['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            label = det_box['label']
            score = det_box['score']
            color = ([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{label}:{score:.2f}', (x1, y1), cv2.FONT_ITALIC, 1, color, 1)
        save_path = os.path.join(self.save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img)

    @staticmethod
    def image_to_base64(image_path):
        """读取图片并转为Base64字符串"""
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
        encoded_bytes = base64.b64encode(binary_data)
        return encoded_bytes.decode('utf-8')
    
    def post(self, encoded_data, verbose=True):
        """发送推理请求，返回检测结果或None（失败时）"""
        payload = {
            "inputs": [{
                "name": "RAW_IMAGE",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [encoded_data]
            }]
        }
        start_time = time.time()
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            elapsed = (time.time() - start_time) * 1000  # ms
            if response.status_code != 200:
                if verbose:
                    print(f"请求失败! 状态码: {response.status_code}, 错误: {response.text}")
                return None, elapsed
            result_json = response.json()
            output_json = result_json['outputs'][0]['data'][0]
            output_result = json.loads(output_json)
            if verbose:
                print(output_result)
            return output_result, elapsed
        except Exception as e:
            if verbose:
                print(f"请求异常: {e}")
            return None, (time.time() - start_time) * 1000

    def post_file(self, image_path, verbose=True):
        """直接传入图片路径，内部编码后发送（便于单次调试）"""
        encoded = self.image_to_base64(image_path)
        result, elapsed = self.post(encoded, verbose=verbose)
        return result

def load_images_from_folder(folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """
    读取文件夹中所有图片文件的路径
    :param folder_path: 文件夹路径
    :param extensions: 支持的图片扩展名
    :return: 图片路径列表
    """
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    if not image_paths:
        raise ValueError(f"在文件夹 {folder_path} 中未找到任何支持的图片文件")
    print(f"找到 {len(image_paths)} 张图片")
    return image_paths

def benchmark_multi_images(
    image_paths,
    num_requests=100,
    concurrency=10,
    http_url='http://localhost:8000',
    model_name='yolov26_ensemble',
    verbose=False
):
    """
    高并发吞吐量测试（随机从图片列表中选择图片）
    :param image_paths:  图片路径列表
    :param num_requests: 总请求数
    :param concurrency:  并发线程数
    :param http_url:     Triton HTTP地址
    :param model_name:   模型名称
    :param verbose:      是否打印详细错误
    :return: dict 包含性能指标
    """
    # 预加载所有图片并编码为Base64，保留原始路径映射（可选）
    print("预加载并编码所有图片...")
    encoded_images = []
    for path in image_paths:
        encoded_images.append(HTTP_CLIENT.image_to_base64(path))
    
    # 创建客户端实例（仅用于获取URL）
    client = HTTP_CLIENT(http_url=http_url, model_name=model_name)
    
    latencies = []
    errors = 0
    
    def worker(_):
        nonlocal errors
        session = requests.Session()
        # 随机选择一张图片的编码数据
        encoded = random.choice(encoded_images)
        payload = {
            "inputs": [{
                "name": "RAW_IMAGE",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [encoded]
            }]
        }
        start = time.time()
        try:
            resp = session.post(client.url, json=payload, timeout=30)
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                return elapsed
            else:
                errors += 1
                if verbose:
                    print(f"请求失败状态码: {resp.status_code}, 响应: {resp.text}")
                return None
        except Exception as e:
            errors += 1
            if verbose:
                print(f"请求异常: {e}")
            return None
        finally:
            session.close()
    
    start_total = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_requests, desc=f"并发{concurrency}"):
            result = future.result()
            if result is not None:
                latencies.append(result)
    end_total = time.time()
    total_time = end_total - start_total
    success_count = len(latencies)
    
    if success_count == 0:
        print("所有请求均失败！")
        return None
    
    qps = success_count / total_time
    avg_latency = statistics.mean(latencies)
    max_lat = max(latencies)
    min_lat = min(latencies)
    stdev_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted)*0.5)]
    p90 = latencies_sorted[int(len(latencies_sorted)*0.9)]
    p99 = latencies_sorted[int(len(latencies_sorted)*0.99)]
    
    results = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "success_count": success_count,
        "error_count": errors,
        "total_time_sec": round(total_time, 2),
        "qps": round(qps, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "std_latency_ms": round(stdev_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p99_latency_ms": round(p99, 2)
    }
    
    # print(f"\n并发数 {concurrency} 完成: QPS={qps:.2f}, 平均延迟={avg_latency:.2f}ms")
    print(f"\n========== 并发数 {concurrency} 测试结果 ==========")
    print(f"总请求数:        {num_requests}")
    print(f"成功请求数:      {success_count}")
    print(f"失败请求数:      {errors}")
    print(f"总耗时(s):       {total_time:.2f}")
    print(f"吞吐量(QPS):     {qps:.2f}")
    print(f"平均延迟(ms):    {avg_latency:.2f}")
    print(f"最小延迟(ms):    {min_lat:.2f}")
    print(f"最大延迟(ms):    {max_lat:.2f}")
    print(f"标准差(ms):      {stdev_lat:.2f}")
    print(f"P50 延迟(ms):    {p50:.2f}")
    print(f"P90 延迟(ms):    {p90:.2f}")
    print(f"P99 延迟(ms):    {p99:.2f}")
    print("==============================")
    return results

def run_benchmark_sweep(
    image_folder,
    concurrency_list=[1, 2, 4, 8, 16, 32],
    num_requests_per_test=200,
    http_url='http://localhost:8000',
    model_name='yolov26_ensemble',
    output_json='benchmark_results.json',
    output_csv='benchmark_results.csv'
):
    """
    遍历并发数进行吞吐量测试，并保存结果为JSON和CSV
    """
    # 读取所有图片路径
    image_paths = load_images_from_folder(image_folder)
    if not image_paths:
        print("没有找到图片，退出。")
        return
    
    all_results = []
    
    for concurrency in concurrency_list:
        print(f"\n========== 测试并发数: {concurrency} ==========")
        result = benchmark_multi_images(
            image_paths=image_paths,
            num_requests=num_requests_per_test,
            concurrency=concurrency,
            http_url=http_url,
            model_name=model_name,
            verbose=False
        )
        if result is not None:
            all_results.append(result)
        # 可选：每次测试后等待几秒让服务端恢复
        time.sleep(2)
    
    # 保存JSON
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n性能指标已保存至 {output_json}")
    
    # 保存CSV
    if all_results:
        keys = all_results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"性能指标已保存至 {output_csv}")
    else:
        print("没有成功的结果可保存。")

if __name__ == "__main__":
    # 单次请求测试（保持原始绘图功能）
    # single_image = r"images/bus.jpg"
    # client = HTTP_CLIENT()
    # result = client.post_file(single_image, verbose=True)
    # if result:
    #     client.draw(single_image, result)
    
    # 吞吐量测试：指定图片文件夹，遍历并发数，保存结果
    save_dir = 'perf_result'
    os.makedirs(save_dir, exist_ok=True)
    backend = 'tensorrt'
    device = 'A100'
    run_benchmark_sweep(
        image_folder=r"traffic/images/val",            # 存放测试图片的文件夹
        concurrency_list=[1, 16, 32, 64, 128, 256, 512],
        num_requests_per_test=1000,
        output_json=os.path.join(save_dir, f"{device}_{backend}_benchmark_results.json"),
        output_csv=os.path.join(save_dir, f"{device}_{backend}_benchmark_results.csv")
    )