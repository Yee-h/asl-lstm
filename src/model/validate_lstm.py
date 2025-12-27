#用于在训练过程中评估模型在 Dev 集上的表现。

import torch
from tqdm import tqdm

def validate(model, dataloader, criterion, device):
    """
    验证模型性能
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率 (0.0 - 1.0)
    """
    model.eval() # 切换到评估模式 (关闭 Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 不计算梯度，节省显存
        for inputs, labels, lengths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # lengths 留在 CPU 上，因为 pack_padded_sequence 需要 CPU tensor
            
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy