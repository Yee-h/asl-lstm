import torch
import torch.nn as nn
import sys
import os
import io
import time
from tqdm import tqdm
import json

# 修复 Windows 终端中文乱码和进度条问题
if sys.platform == 'win32':
    # 设置标准输出和错误为 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 启用 Windows 终端的 ANSI 转义序列支持
    os.system('')

# 将 src 添加到 python 路径以允许导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders

def load_label_map_inverse():
    """加载标签映射并将其反转为 ID 到名称的映射。"""
    with open(cfg.LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 提取映射字典
    if 'id_to_label' in data:
        mapping = data['id_to_label']
    elif 'label_to_id' in data:
        mapping = data['label_to_id']
    else:
        mapping = data
    
    if not mapping:
        return {}
        
    # 检查第一个项以此推断映射方向
    # 我们需要返回 {int(ID): str(Label)}
    k, v = next(iter(mapping.items()))
    
    # 情况 1: 键是 ID (数字字符串或整数), 值是 Label
    # 例如: "0": "book"
    if str(k).isdigit():
        return {int(key): val for key, val in mapping.items()}
        
    # 情况 2: 值是 ID (数字), 键是 Label
    # 例如: "book": 0 (这是我们遇到的情况，即使键名是 id_to_label)
    if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
        return {int(val): key for key, val in mapping.items()}
        
    return {}

def evaluate_model():
    """
    在测试数据集上评估保存的模型。
    """
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.DEVICE == 'cuda' else 'cpu')
    print(f"当前使用的设备: {device}")

    # 2. 加载数据
    print("正在加载测试数据...")
    _, _, test_loader = get_dataloaders()
    print(f"测试数据加载完成。批次数量: {len(test_loader)}")

    # 3. 初始化模型
    model = get_model(use_attention=cfg.USE_ATTENTION).to(device)
    model_type = "BiLSTM+Attention" if cfg.USE_ATTENTION else "BiLSTM"
    print(f"模型架构: {model_type}")
    
    # 4. 加载权重
    model_path = cfg.TEST_MODEL_PATH
    
    print(f"正在加载模型: {model_path}")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重加载成功。")
    else:
        print(f"错误: 未在 {model_path} 找到模型文件")
        return

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 6. 评估循环
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    print("开始评估...")
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="评估中", ascii=True, ncols=100)
        for inputs, labels, lengths in pbar:
            inputs = inputs.to(device)
            labels = labels.long().to(device) # 确保标签是 LongTensor

            # 前向传播 (传递 lengths 参数)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # 统计指标
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 存储用于计算指标
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    duration = end_time - start_time
    
    # 7. 计算最终指标
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    print("\n" + "="*40)
    print(f"评估结果")
    print("="*40)
    print(f"模型: {os.path.basename(model_path)}")
    print(f"测试集 Loss:     {avg_loss:.4f}")
    print(f"测试集准确率: {accuracy:.2f}%")
    print(f"样本总数: {total}")
    print(f"耗时:    {duration:.2f}s")
    print("="*40)

    # 8. 详细报告 (如果可用 sklearn)
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        # 加载标签名称
        id_to_label = load_label_map_inverse()
        target_names = [id_to_label.get(i, str(i)) for i in range(cfg.NUM_CLASSES)]
        
        # 分类报告
        print("\n详细分类报告:")
        report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
        print(report)
        
        # --- 保存报告到 logs 目录 ---
        log_dir = os.path.join(cfg.PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"evaluation_report_{timestamp}.txt"
        log_path = os.path.join(log_dir, log_filename)
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("="*40 + "\n")
            f.write(f"评估报告\n")
            f.write("="*40 + "\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {model_path}\n")
            f.write(f"测试集 Loss:     {avg_loss:.4f}\n")
            f.write(f"测试集准确率: {accuracy:.2f}%\n")
            f.write(f"样本总数: {total}\n")
            f.write(f"耗时:    {duration:.2f}s\n")
            f.write("-" * 40 + "\n")
            f.write("详细分类报告:\n")
            f.write(report)
            
        print(f"\n详细评估报告已保存至: {log_path}")
            
    except ImportError:
        print("\n注意: 未安装 'scikit-learn'。跳过详细分类报告。")
    except Exception as e:
        print(f"\n无法生成详细报告: {e}")

if __name__ == "__main__":
    evaluate_model()
