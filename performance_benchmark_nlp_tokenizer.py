import requests
import time
import os
import json
import concurrent.futures
import statistics
import random
import csv
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import numpy as np


class PerformanceBenchmark:
    """
    ModernBERT 模型（文本）高性能并发推理测试器
    从 Alpaca 格式 JSON 文件加载文本，通过 Triton HTTP 请求进行推理性能测试。
    """

    def __init__(
        self,
        http_url: str,
        model_name: str,
        tokenizer_path: str,
        text_path: str,
        backend: str = "onnx",
        min_text_len: int = 128,
        max_text_len: int = 512,
        max_seq_len: int = 512,
        request_timeout: int = 30,
        verbose: bool = False,
    ):
        """
        :param http_url: Triton 服务基础 URL (如 http://localhost:8000)
        :param model_name: 模型名称
        :param tokenizer_path: HuggingFace 分词器路径
        :param text_path: 测试文本 JSON 文件路径（Alpaca 格式）
        :param backend: 后端类型 "onnx" 或 "trt"（影响 datatype）
        :param min_text_len: 保留文本的最小长度（去除空格和换行后）
        :param max_text_len: 文本截断到的最大长度（预处理时）
        :param max_seq_len: 模型实际接收的最大序列长度（分词时）
        :param request_timeout: 单次请求超时(秒)
        :param verbose: 是否打印详细错误信息
        """
        self.http_url = http_url.rstrip("/")
        self.model_name = model_name
        self.infer_url = f"{self.http_url}/v2/models/{self.model_name}/infer"
        self.backend = backend
        self.request_timeout = request_timeout
        self.verbose = verbose
        self.max_seq_len = max_seq_len

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 加载并预处理文本列表
        self.texts = self._load_texts(text_path, min_text_len, max_text_len)
        if not self.texts:
            raise RuntimeError("没有可用的测试文本，请检查数据文件或长度过滤条件。")

    def _load_texts(self, text_path: str, min_len: int, max_len: int) -> list:
        """从 Alpaca JSON 文件加载文本，拼接 instruction/input/output，过滤长度"""
        combined_texts = []
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                combined = instruction + input_text + output_text
                combined = combined.replace(" ", "").replace("\n", "")
                if len(combined) < min_len:
                    continue
                if len(combined) > max_len:
                    combined = combined[:max_len]
                combined_texts.append(combined)

            print(f"成功加载 {len(combined_texts)} 条有效文本（原始数据 {len(data)} 条）")
            return combined_texts

        except FileNotFoundError:
            print(f"错误: 文件 {text_path} 不存在")
        except json.JSONDecodeError:
            print(f"错误: 文件 {text_path} 不是有效的 JSON 格式")
        return []

    def _get_request_inputs(self, text: str) -> dict:
        """将单个文本转换为 Triton v2 请求的 payload"""
        # 分词
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="np",
        )
        # 根据后端选择数据类型
        dtype = np.int64 if self.backend == "onnx" else np.int32
        input_ids = encoded["input_ids"].astype(dtype)
        attention_mask = encoded["attention_mask"].astype(dtype)
        datatype = "INT64" if self.backend == "onnx" else "INT32"

        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": datatype,
                    "data": input_ids.tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": datatype,
                    "data": attention_mask.tolist(),
                },
            ]
        }
        return payload

    def _single_benchmark(
        self,
        num_requests: int,
        concurrency: int,
    ) -> dict:
        """单轮并发测试（内部方法）"""
        latencies = []
        errors = 0

        def worker(_):
            nonlocal errors
            session = requests.Session()
            text = random.choice(self.texts)
            payload = self._get_request_inputs(text)

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
    parser = argparse.ArgumentParser(description="Triton 性能测试")
    parser.add_argument("--model", "-m", type=str, default="modernbert_ensemble", help="Triton ensemble模型名称")
    parser.add_argument("--backend", "-b", type=str, default="trt", choices=["onnx", "trt"], help="后端类型")
    parser.add_argument("--device", "-d", type=str, default="A10", help="设备类型（仅用于文件名标识）")
    parser.add_argument("--http", "-u", type=str, default="http://localhost:8000", help="Triton HTTP 地址")
    parser.add_argument("--max_batch_size", type=int, default=16, help="最大批次（仅用于文件名标识）")
    parser.add_argument("--cpu", "-p", type=int, default=32, help="CPU核心数（仅用于文件名标识）")
    parser.add_argument("--concurrency", "-c", type=str, default="1,16,32,64,128,256,512", help="并发数列表，逗号分隔")
    parser.add_argument("--num", "-n", type=int, default=1000, help="每次测试的总请求数")
    parser.add_argument("--input", "-i", type=str, default="alpaca_gpt4_data_zh.json", help="测试文本 JSON 文件")
    parser.add_argument("--output_dir", "-o", type=str, default="performance_result", help="结果保存目录")
    parser.add_argument("--token_path", "-t", type=str, default="models/modernbert-base", help="分词器路径")
    parser.add_argument("--max_seq_len", type=int, default=512, help="模型最大序列长度")
    parser.add_argument("--timeout", type=int, default=30, help="单次请求超时秒数")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细错误信息")
    return parser.parse_args()


def main():
    args = parse_opt()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 构造输出文件名（与原脚本一致）
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
        tokenizer_path=args.token_path,
        text_path=args.input,
        backend=args.backend,
        min_text_len=128,
        max_text_len=args.max_seq_len,      # 预处理截断与序列长度保持一致
        max_seq_len=args.max_seq_len,
        request_timeout=args.timeout,
        verbose=args.verbose,
    )

    # 执行测试
    benchmark.run_benchmark_sweep(
        concurrency_list=concurrency_list,
        num_requests_per_test=args.num,
        output_csv=output_csv,
        cooldown_sec=2,
    )


if __name__ == "__main__":
    main()