# 检查系统中的 CUDA (GPU) 环境是否可用
import torch

# 输出 CUDA 的基本可用性状态
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 如果可用，打印 GPU 的详细信息
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 索引: {torch.cuda.current_device()}")
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
