import requests
import time
import base64
import os
import concurrent.futures
import statistics
import random
import csv
from tqdm import tqdm
import argparse


class PerformanceBenchmark:

    def __init__(
        self,
        http_url: str,
        model_name: str,
        image_folder: str,
        request_timeout: int = 30,
        verbose: bool = False,
    ):
        """
        :param http_url: Triton 服务基础 URL (如 http://localhost:8000)
        :param model_name: 模型名称（ensemble名称）
        :param image_folder: 存放测试图片的文件夹路径
        :param request_timeout: 单次请求超时(秒)
        :param verbose: 是否打印详细错误信息
        """
        self.http_url = http_url.rstrip("/")
        self.model_name = model_name
        self.infer_url = f"{self.http_url}/v2/models/{self.model_name}/infer"
        self.request_timeout = request_timeout
        self.verbose = verbose

        # 加载图片路径列表
        self.image_paths = self._load_images(image_folder)
        if not self.image_paths:
            raise RuntimeError(f"在文件夹 {image_folder} 中未找到任何支持的图片文件")

        # 预编码所有图片为 Base64（提高并发效率）
        self.encoded_images = self._preload_encoded_images()

    def _load_images(self, folder_path: str, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> list:
        """读取文件夹中所有图片文件的路径"""
        image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, file))
        if not image_paths:
            raise ValueError(f"在文件夹 {folder_path} 中未找到任何支持的图片文件")
        print(f"找到 {len(image_paths)} 张图片")
        return image_paths

    def _image_to_base64(self, image_path: str) -> str:
        """将图片文件转为 Base64 字符串"""
        with open(image_path, "rb") as f:
            binary_data = f.read()
        return base64.b64encode(binary_data).decode('utf-8')

    def _preload_encoded_images(self) -> list:
        """预加载所有图片并编码为 Base64，返回编码列表"""
        print("预加载并编码所有图片...")
        encoded = []
        for path in self.image_paths:
            encoded.append(self._image_to_base64(path))
        return encoded

    def _single_benchmark(
        self,
        num_requests: int,
        concurrency: int,
    ) -> dict:
        """单次并发测试（内部方法）"""
        latencies = []
        errors = 0

        def worker(_):
            nonlocal errors
            session = requests.Session()
            # 随机选择一张图片
            encoded = random.choice(self.encoded_images)
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
                resp = session.post(self.infer_url, json=payload, timeout=self.request_timeout)
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    return elapsed
                else:
                    errors += 1
                    if self.verbose:
                        print(f"请求失败状态码: {resp.status_code}, 响应: {resp.text}")
                    return None
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"请求异常: {e}")
                return None
            finally:
                session.close()

        start_total = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(num_requests)]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=num_requests,
                desc=f"并发{concurrency}",
            ):
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
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(len(latencies_sorted) * 0.5)]
        p90 = latencies_sorted[int(len(latencies_sorted) * 0.9)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

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
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
            "p99_latency_ms": round(p99, 2),
        }

        print(f"\n========== 并发数 {concurrency} 测试结果 ==========")
        print(f"总请求数:        {num_requests}")
        print(f"成功请求数:      {success_count}")
        print(f"失败请求数:      {errors}")
        print(f"总耗时(s):       {total_time:.2f}")
        print(f"吞吐量(QPS):     {qps:.2f}")
        print(f"平均延迟(ms):    {avg_latency:.2f}")
        print(f"最小延迟(ms):    {min_lat:.2f}")
        print(f"最大延迟(ms):    {max_lat:.2f}")
        print(f"P50 延迟(ms):    {p50:.2f}")
        print(f"P90 延迟(ms):    {p90:.2f}")
        print(f"P99 延迟(ms):    {p99:.2f}")
        print("==============================")
        return results

    def run_benchmark_sweep(
        self,
        concurrency_list: list,
        num_requests_per_test: int,
        output_csv: str,
        cooldown_sec: int = 2,
    ):
        """
        遍历多个并发数进行测试，并保存结果至 CSV。
        :param concurrency_list: 并发数列表，如 [1, 2, 4, 8, 16]
        :param num_requests_per_test: 每次测试的总请求数
        :param output_csv: 结果输出 CSV 文件路径
        :param cooldown_sec: 每轮测试后的冷却时间（秒）
        """
        all_results = []

        for concurrency in concurrency_list:
            print(f"\n========== 测试并发数: {concurrency} ==========")
            result = self._single_benchmark(
                num_requests=num_requests_per_test,
                concurrency=concurrency,
            )
            if result is not None:
                all_results.append(result)
            time.sleep(cooldown_sec)

        if all_results:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            print(f"性能指标已保存至 {output_csv}")
        else:
            print("没有成功的结果可保存。")


def parse_opt():
    parser = argparse.ArgumentParser(description="Triton 图像模型性能测试")
    parser.add_argument("--model", "-m", type=str, default="yolov26_ensemble", help="Triton ensemble 名称")
    parser.add_argument("--backend", "-b", type=str, default="trt", choices=["onnx", "trt"], help="后端类型（仅用于输出文件名标识）")
    parser.add_argument("--device", "-d", type=str, default="A10", help="设备类型（仅用于输出文件名标识）")
    parser.add_argument("--http", "-u", type=str, default="http://localhost:8000", help="Triton HTTP 地址")
    parser.add_argument("--max_batch_size", type=int, default=128, help="最大批次（仅用于输出文件名标识）")
    parser.add_argument("--cpu", "-p", type=int, default=32, help="CPU核心数（仅用于输出文件名标识）")
    parser.add_argument("--concurrency", "-c", type=str, default="1,16,32,64,128,256,512", help="并发数列表，逗号分隔")
    parser.add_argument("--num", "-n", type=int, default=1000, help="每次测试的总请求数")
    parser.add_argument("--input_dir", "-i", type=str, default="traffic/images/val", help="测试图片文件夹")
    parser.add_argument("--output_dir", "-o", type=str, default="performance_result", help="结果保存目录")
    parser.add_argument("--timeout", type=int, default=30, help="单次请求超时秒数")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细错误信息")
    return parser.parse_args()


def main():
    args = parse_opt()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 构造输出文件名
    output_csv = os.path.join(
        args.output_dir,
        f"{args.model}_{args.device}_{args.backend}_batch{args.max_batch_size}_cpu{args.cpu}_benchmark_results.csv"
    )

    # 解析并发数列表
    concurrency_list = [int(x.strip()) for x in args.concurrency.split(",")]

    # 初始化测试器
    benchmark = PerformanceBenchmark(
        http_url=args.http,
        model_name=args.model,
        image_folder=args.input_dir,
        request_timeout=args.timeout,
        verbose=args.verbose,
    )

    # 执行测试
    benchmark.run_benchmark_sweep(
        concurrency_list=concurrency_list,
        num_requests_per_test=args.num,
        output_csv=output_csv,
        cooldown_sec=2,          # 每轮测试后等待2秒
    )


if __name__ == "__main__":
    main()