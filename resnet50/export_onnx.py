import torch
import torchvision.models as models
import os

def convert_to_onnx(weight_path, onnx_path, input_size=(3, 224, 224)):
    """
    将 ResNet-50 的 PyTorch 权重转换为 ONNX 模型

    Args:
        weight_path (str): 本地 .pth 权重文件路径
        onnx_path (str): 输出的 ONNX 文件路径
        input_size (tuple): 输入张量的形状 (C, H, W)，默认 (3, 224, 224)
    """
    # 1. 创建模型结构（不加载预训练权重）
    model = models.resnet50(weights=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 加载本地权重
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 3. 设置设备并切换到推理模式
    model = model.to(device)
    model.eval()
    
    # 4. 构造示例输入
    batch_size = 1
    c, h, w = input_size
    example_input = torch.randn(batch_size, c, h, w).to(device)
    
    # 5. 执行一次前向传播，验证模型可用
    with torch.no_grad():
        output = model(example_input)
    print(f"前向验证通过，输出形状: {output.shape}")
    
    # 6. 导出 ONNX
    print(f"开始导出 ONNX 到: {onnx_path}")
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,          # 保存模型参数
        opset_version=11,            # ONNX opset 版本，常用 11/12
        do_constant_folding=True,    # 常量折叠优化
        input_names=['input'],       # 输入节点名
        output_names=['output'],     # 输出节点名
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },                            # 动态 batch 维度
        dynamo=False                  # 关键！强制使用 TorchScript 导出器
    )
    print("ONNX 导出成功！")

if __name__ == '__main__':
    weight_path = r'weights/resnet50_imagenet_v2.pth'
    onnx_path = r'weights/resnet50_imagenet_v2.onnx'
    input_size = (3, 224, 224)
    convert_to_onnx(weight_path, onnx_path, input_size)