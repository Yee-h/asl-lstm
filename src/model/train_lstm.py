#主训练循环，负责保存模型和标签映射。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import json
import time

# 路径 hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.padding import CSLDataset, collate_fn
from src.model.model_lstm import CSL_BiLSTM
from src.model.validate_lstm import validate

# 使用 config.py 统一配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"Using Device: {DEVICE}")
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. 准备数据集
    print("正在加载数据集...")
    train_dataset = CSLDataset('train')
    dev_dataset = CSLDataset('dev', label_map=train_dataset.label_map) # 共享词汇表
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=cfg.NUM_DATALOADER_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=cfg.NUM_DATALOADER_WORKERS)
    
    # 保存 Label Map 供测试使用
    with open(cfg.VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_dataset.label_map, f, ensure_ascii=False, indent=2)
    print(f"词汇表已保存至: {cfg.VOCAB_PATH}")

    # 2. 初始化模型
    num_classes = len(train_dataset.label_map)
    model = CSL_BiLSTM(
        input_dim=cfg.TOTAL_FEATURE_DIM, # 225
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT
    ).to(DEVICE)
    
    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # 学习率调度器: 如果验证集 Loss 不下降，则 LR 衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.LR_SCHEDULER_FACTOR, 
        patience=cfg.LR_SCHEDULER_PATIENCE, verbose=True
    )

    # 4. 训练循环
    best_val_acc = 0.0
    print(f"开始训练 (Epochs: {cfg.NUM_EPOCHS})...")
    
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()
        model.train() # 训练模式
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # 计算 Epoch 级指标
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # --- 验证阶段 ---
        val_loss, val_acc = validate(model, dev_loader, criterion, DEVICE)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved (Acc: {best_val_acc:.4f})")
            
    print("训练结束.")

if __name__ == "__main__":
    train()