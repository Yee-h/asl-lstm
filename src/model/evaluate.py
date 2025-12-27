#加载最佳模型和词汇表，在 Test 集上运行对模型进行评估。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.padding import CSLDataset, collate_fn
from src.model.model_lstm import CSL_BiLSTM

# 使用 config.py 统一配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    print("=== 开始测试 ===")
    
    if not os.path.exists(cfg.BEST_MODEL_PATH) or not os.path.exists(cfg.VOCAB_PATH):
        print("错误: 找不到模型文件或词汇表，请先运行 train_lstm.py")
        return

    # 1. 加载词汇表
    with open(cfg.VOCAB_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    print(f"加载词汇表: {len(label_map)} 类")
    
    # 2. 准备数据集 (Test)
    test_dataset = CSLDataset('test', label_map=label_map)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. 加载模型
    model = CSL_BiLSTM(
        input_dim=cfg.TOTAL_FEATURE_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=len(label_map),
        num_layers=cfg.NUM_LAYERS
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"成功加载模型: {cfg.BEST_MODEL_PATH}")
    
    # 4. 推理
    correct = 0
    total = 0
    
    # 存储错误样本以便分析
    error_cases = []
    
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录错误
            incorrect_mask = predicted != labels
            if incorrect_mask.any():
                idx_tensor = torch.arange(labels.size(0))[incorrect_mask]
                for idx in idx_tensor:
                    true_idx = labels[idx].item()
                    pred_idx = predicted[idx].item()
                    error_cases.append({
                        'true': test_dataset.idx_to_label[true_idx],
                        'pred': test_dataset.idx_to_label[pred_idx]
                    })
    
    acc = correct / total
    print(f"\n测试集准确率: {acc*100:.2f}%")
    
    if error_cases:
        print(f"\n错误样本示例 (前10个):")
        for i, case in enumerate(error_cases[:10]):
            print(f"  True: {case['true']} | Pred: {case['pred']}")

if __name__ == "__main__":
    test()