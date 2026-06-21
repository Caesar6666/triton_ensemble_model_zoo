import torch
import torchvision
import onnx

# ========== 1. 加载预训练模型 ==========
model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None,          # 不使用内置预训练权重
    aux_loss=True,         # 启用 aux_classifier，以匹配权重文件
    num_classes=21         # COCO 标准是 21 类（含背景），请按需修改
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = r'weights/deeplabv3_resnet50_coco.pth'
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device).eval()

# ========== 2. 准备示例输入 ==========
# ONNX 导出需要提供一个示例输入张量，用于追踪计算图。
# 这里 batch_size 设为 1，但稍后会通过 dynamic_axes 声明为动态。
batch_size = 1
num_channels = 3
height, width = 512, 512   # DeepLabV3 期望的典型输入尺寸
dummy_input = torch.randn(batch_size, num_channels, height, width)

# ========== 3. 定义动态轴 ==========
# 在导出时，指定哪些维度是动态的（可变长度）。
# 我们通常希望 batch 维度动态变化，也可以让高度/宽度动态（但需要注意模型内部可能对尺寸有隐含约束）。
dynamic_axes = {
    "input": {0: "batch_size"},        # 输入张量的第 0 维是 batch
    "output": {0: "batch_size"},       # 输出字典中的 "out" 张量的第 0 维是 batch
    # 如果还需要动态图像大小，可以取消注释下面的行，但可能需要对预处理做额外处理
    # "input": {2: "height", 3: "width"},
    # "output": {2: "height", 3: "width"},
}

# ========== 4. 导出 ONNX 模型 ==========
onnx_file_path = "weights/deeplabv3_resnet50_coco_v2.onnx"
torch.onnx.export(
    model,
    dummy_input,                         # 示例输入
    onnx_file_path,                      # 保存路径
    input_names=["input"],               # 输入名字
    output_names=["output"],             # 输出名字
    dynamic_axes=dynamic_axes,           # 动态轴配置
    opset_version=11,                    # ONNX opset 版本，建议 >=11
    do_constant_folding=True,            # 折叠常量优化
    dynamo=False,                       # 是否打印导出日志
)

print(f"ONNX 模型已保存至: {onnx_file_path}")

# ========== 5. 验证导出的 ONNX 模型（可选） ==========
# 加载 ONNX 模型进行结构验证
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过！")

# ========== 6. 使用 ONNX Runtime 进行简单推理测试（可选）==========
try:
    import onnxruntime as ort
    import numpy as np

    # 创建 ONNX Runtime 推理会话
    ort_session = ort.InferenceSession(onnx_file_path)

    # 准备不同 batch 大小的输入，测试动态 batch 是否正常工作
    for test_batch in [1, 2, 4]:
        test_input = np.random.randn(test_batch, 3, height, width).astype(np.float32)
        outputs = ort_session.run(["output"], {"input": test_input})
        print(f"Batch size {test_batch} -> 输出形状: {outputs[0].shape}")
except ImportError:
    print("未安装 onnxruntime，跳过动态 batch 测试。可运行 pip install onnxruntime 安装")