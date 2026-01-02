import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg

def validate(model, val_loader, criterion, device):
    """
    在验证集或测试集上评估模型性能。
    
    Args:
        model (nn.Module): 待评估的模型。
        val_loader (DataLoader): 验证集的数据加载器。
        criterion (loss): 损失函数。
        device (torch.device): 计算设备 (CPU 或 GPU)。
        
    Returns:
        epoch_loss (float): 平均损失。
        accuracy (float): 分类准确率 (%)。
    """
    model.eval() # 设置模型为评估模式 (关闭 Dropout 和 Batch Normalization)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 评估过程中不需要计算梯度，节省内存和计算资源
    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            # 迁移数据到设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # --- 前向传播 ---
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            
            # --- 统计指标 ---
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 累计预测正确的数量
            
    # 计算平均指标
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return epoch_loss, accuracy

if __name__ == "__main__":
    print("This script is intended to be imported by train_lstm.py")
