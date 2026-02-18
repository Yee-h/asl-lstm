# ASL-LSTM 验证集准确率提升方案

> 当前状态：训练集 ~96%，验证集 ~70%（最高 71.22%），差距 ~26%，严重过拟合
> 目标：验证集准确率提升至 80%+
> 本文档为确定性执行方案，无选择性决策，按Phase顺序执行

---

## 诊断总结

### 训练曲线关键数据点

| 阶段 | Epoch | Train Acc | Val Acc | Train Loss | Val Loss | LR |
|------|-------|-----------|---------|------------|----------|-----|
| 开始收敛 | 122 | 77.88% | 69.73% | 1.09 | 1.33 | 8e-4 |
| 最高验证 | 166 | 86.96% | 71.22% | 0.80 | 1.29 | 4.8e-4 |
| LR衰减后 | 200 | 92.93% | 69.73% | 0.59 | 1.31 | 1.73e-4 |
| 早停触发 | 316 | 96.12% | 68.84% | 0.47 | 1.34 | 3e-6 |

### 5个确诊的过拟合根因

1. **模型输入维度过大**：540维输入（135关键点 × 4通道）直接喂入LSTM，无压缩层，大量无关特征（腿部、部分面部点）成为噪声源
2. **正则化严重不足**：整个模型仅有FC层前一个Dropout(0.35)，无LayerNorm、无输入dropout、无特征mask
3. **数据增强过于保守**：noise_std=0.003极小，time_warp范围[0.9,1.1]极窄，无Mixup/CutMix，无时间裁剪，无关键点级dropout
4. **学习率策略导致过早收敛到尖锐极小值**：ReduceLROnPlateau(factor=0.6, patience=12) 在epoch 147就将LR从8e-4衰减到4.8e-4，之后持续衰减至3e-6，模型困在尖锐极小值无法跳出
5. **优化器选择不当**：Adam + weight_decay=4e-4 并非真正的解耦权重衰减，正则效果打折

---

## Phase 1: 模型架构改造（预期提升 +5~8%）

### 1.1 添加输入投影层（压缩540维 → 256维）

**修改文件**: `src/model/model_lstm.py` 的 `BiLSTMAttention.__init__` 和 `forward`

**在 `__init__` 中，`self.lstm` 定义之前添加**:
```python
# 输入投影层：压缩高维稀疏特征，降低过拟合风险
self.input_projection = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.LayerNorm(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
)
```

**修改 `self.lstm` 的 `input_size` 参数**:
```python
self.lstm = nn.LSTM(
    256,  # 从 input_size 改为 256（投影后维度）
    hidden_size,
    num_layers,
    batch_first=True,
    bidirectional=self.bidirectional,
    dropout=dropout if num_layers > 1 else 0,
)
```

**在 `forward` 方法中，`pack_padded_sequence` 之前添加**:
```python
x = self.input_projection(x)  # (batch, seq_len, 540) -> (batch, seq_len, 256)
```

### 1.2 添加LSTM输出LayerNorm

**在 `__init__` 中，`self.attention` 定义之后添加**:
```python
self.layer_norm = nn.LayerNorm(lstm_output_dim)
```

**在 `forward` 中，attention计算之前添加**:
```python
lstm_output = self.layer_norm(lstm_output)  # 在 pad_packed_sequence 之后
```

### 1.3 扩展分类头

**替换现有的 `self.dropout_fc` 和 `self.fc`**:
```python
self.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(lstm_output_dim, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes),
)
```

**`forward` 中替换**:
```python
# 旧代码:
# out = self.dropout_fc(context)
# out = self.fc(out)
# 新代码:
out = self.classifier(context)
```

### 1.4 增大注意力维度

**修改 `src/config.py`**:
```python
# 旧值:
attention_dim=32,
# 新值:
attention_dim=64,
```

### 1.5 对 forward_with_attention 方法做同样修改

`forward_with_attention` 方法的改动与 `forward` 完全一致，需要同步添加 `input_projection`、`layer_norm` 和 `classifier` 的调用。

---

## Phase 2: 正则化全面加强（预期提升 +3~5%）

### 2.1 切换优化器：Adam → AdamW

**修改文件**: `src/model/train_lstm.py` 的 `train()` 函数

```python
# 旧代码:
optimizer = optim.Adam(
    model.parameters(),
    lr=cfg.TRAINING.learning_rate,
    weight_decay=float(training_profile["weight_decay"]),
)
# 新代码:
optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg.TRAINING.learning_rate,
    weight_decay=float(training_profile["weight_decay"]),
)
```

### 2.2 调整权重衰减

**修改文件**: `src/config.py`

```python
# 旧值:
weight_decay=4e-4,
# 新值:
weight_decay=0.01,
```

### 2.3 增大标签平滑

**修改文件**: `src/config.py`

```python
# 旧值:
label_smoothing=0.03,
# 新值:
label_smoothing=0.1,
```

### 2.4 实现 R-Drop 正则化

R-Drop 让同一个样本经过两次前向（不同 dropout mask），对两次输出施加 KL 散度约束，显著提升小数据集泛化能力。

**修改文件**: `src/model/train_lstm.py` 的训练循环

在训练循环内部，将：
```python
outputs = model(inputs, lengths)
loss = criterion(outputs, labels)
```

替换为：
```python
# R-Drop: 同一输入前向两次（不同dropout mask）
outputs1 = model(inputs, lengths)
outputs2 = model(inputs, lengths)
# 交叉熵损失取平均
ce_loss = 0.5 * (criterion(outputs1, labels) + criterion(outputs2, labels))
# KL散度双向对称
p = F.log_softmax(outputs1, dim=-1)
q = F.log_softmax(outputs2, dim=-1)
kl_loss = 0.5 * (
    F.kl_div(p, q.exp(), reduction='batchmean') +
    F.kl_div(q, p.exp(), reduction='batchmean')
)
rdrop_alpha = 1.0
loss = ce_loss + rdrop_alpha * kl_loss
# 用于统计的outputs取平均
outputs = 0.5 * (outputs1 + outputs2)
```

需要在文件顶部添加:
```python
import torch.nn.functional as F
```

---

## Phase 3: 数据增强大幅强化（预期提升 +3~5%）

### 3.1 实现 Mixup

**修改文件**: `src/model/train_lstm.py` 的训练循环

在训练循环内部（R-Drop 之前），添加 Mixup 逻辑：
```python
# Mixup 增强
mixup_alpha = 0.2
if self_augment_enabled:  # 仅训练时
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size_cur = inputs.size(0)
    index = torch.randperm(batch_size_cur).to(device)
    inputs = lam * inputs + (1 - lam) * inputs[index]
    lengths = torch.max(lengths, lengths[index])  # 取较长的length
    # 标签混合在loss计算中处理
    labels_a, labels_b = labels, labels[index]
```

Loss 计算也需要对应修改（替代原来的 criterion 调用）：
```python
ce_loss = lam * criterion(outputs1, labels_a) + (1 - lam) * criterion(outputs1, labels_b)
ce_loss += lam * criterion(outputs2, labels_a) + (1 - lam) * criterion(outputs2, labels_b)
ce_loss = ce_loss * 0.5
```

需要在文件顶部添加:
```python
import numpy as np
```

### 3.2 实现随机时间裁剪

**修改文件**: `src/model/dataloader.py` 的 `_apply_augmentation` 方法

在 `_apply_augmentation` 方法最前面（几何变换之前）添加：
```python
# 随机时间裁剪：随机截取70%~100%的有效帧
if valid_len > 5:
    crop_ratio = np.random.uniform(0.7, 1.0)
    crop_len = max(3, int(valid_len * crop_ratio))
    max_start = valid_len - crop_len
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    
    cropped = np.zeros_like(data)
    cropped[:crop_len] = data[start:start + crop_len]
    data = cropped
    
    if mask is not None:
        cropped_mask = np.zeros_like(mask)
        cropped_mask[:crop_len] = mask[start:start + crop_len]
        mask = cropped_mask
    
    valid_len = crop_len
```

### 3.3 实现关键点级 Dropout

**修改文件**: `src/model/dataloader.py` 的 `_apply_augmentation` 方法

在几何变换之后、return 之前添加：
```python
# 关键点级 dropout：随机将10%的关键点置零
landmark_dropout_prob = 0.10
num_landmarks = work.shape[2]  # 135
drop_mask = np.random.random(num_landmarks) < landmark_dropout_prob
if drop_mask.any():
    work[:, :, drop_mask] = 0.0
    work_mask[:, drop_mask] = 0
```

### 3.4 调整增强超参数

**修改文件**: `src/config.py` 的 `AUGMENTATION`

```python
AUGMENTATION = AugmentationConfig(
    rotation_range=20.0,       # 旧: 18.0
    scale_min=0.85,            # 旧: 0.88
    scale_max=1.15,            # 旧: 1.12
    translate=0.12,            # 旧: 0.10
    noise_std=0.015,           # 旧: 0.003 [关键提升]
    hflip_prob=0.3,            # 旧: 0.25
    hflip_swap_lr=True,        # 不变
    hflip_zero_centered=True,  # 不变
    time_warp_prob=0.35,       # 旧: 0.16 [关键提升]
    time_warp_min=0.85,        # 旧: 0.90
    time_warp_max=1.15,        # 旧: 1.10
    frame_dropout_prob=0.25,   # 旧: 0.10 [关键提升]
    frame_dropout_max_ratio=0.15,  # 旧: 0.08
)
```

---

## Phase 4: 训练策略优化（预期提升 +2~3%）

### 4.1 替换学习率调度器：ReduceLROnPlateau → CosineAnnealingWarmRestarts + 线性Warmup

**修改文件**: `src/model/train_lstm.py`

替换调度器初始化部分：
```python
# 旧代码:
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)

# 新代码:
warmup_epochs = 5
base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=40,        # 首次重启周期40 epochs
    T_mult=2,      # 后续周期翻倍
    eta_min=1e-5,  # 最低学习率
)
```

在每个 epoch 末尾的调度器更新替换为：
```python
# 旧代码:
# if scheduler is not None:
#     scheduler.step(val_loss)

# 新代码:
if epoch < warmup_epochs:
    # 线性warmup
    warmup_lr = cfg.TRAINING.learning_rate * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
else:
    base_scheduler.step(epoch - warmup_epochs)
```

### 4.2 调整训练超参数

**修改文件**: `src/config.py`

```python
# 旧值 → 新值
batch_size=4,           # 不变
grad_accum_steps=8,     # 旧: 4 → 新: 8（等效batch=32，梯度更稳定）
learning_rate=1e-3,     # 旧: 8e-4 → 新: 1e-3
num_epochs=300,         # 旧: 600 → 新: 300（配合余弦退火更高效）
early_stopping_patience=80,  # 旧: 150 → 新: 80（防止在过拟合区间浪费时间）
```

### 4.3 启用验证TTA

**修改文件**: `src/config.py`

```python
# 旧值:
eval_use_tta_hflip=False,
# 新值:
eval_use_tta_hflip=True,
```

注意：TTA 用于验证集推理，不影响模型选择基准（代码中已有 `_evaluate_no_tta_val_acc` 作为模型晋升门禁）。

---

## 执行顺序与验证检查点

### 严格按以下顺序执行，每个Phase完成后训练验证：

| 步骤 | 修改内容 | 涉及文件 | 验证标准 |
|------|---------|---------|---------|
| 1 | Phase 1 全部 | `model_lstm.py`, `config.py` | 模型能正常前向、参数量合理 |
| 2 | Phase 2.1-2.3 | `train_lstm.py`, `config.py` | 训练能正常启动 |
| 3 | Phase 3.4 | `config.py` | 增强参数修改验证 |
| 4 | Phase 3.2-3.3 | `dataloader.py` | DataLoader 输出形状正确 |
| 5 | Phase 4.1-4.2 | `train_lstm.py`, `config.py` | LR warmup 和余弦退火曲线正确 |
| 6 | Phase 2.4 (R-Drop) | `train_lstm.py` | 训练运行，loss包含KL项 |
| 7 | Phase 3.1 (Mixup) | `train_lstm.py` | 训练运行，mixup生效 |
| 8 | Phase 4.3 | `config.py` | 验证TTA生效 |

### 完整训练一轮后验证：
- val_acc 应在第 80-150 epoch 达到 78%+
- train-val gap 应控制在 10-15% 以内
- val_loss 不应在 epoch 100 之前就开始单调上升

---

## 修改文件汇总

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| `src/config.py` | 参数调整 | attention_dim, weight_decay, label_smoothing, augmentation参数, grad_accum_steps, learning_rate, num_epochs, early_stopping_patience, eval_use_tta_hflip |
| `src/model/model_lstm.py` | 架构改造 | 添加input_projection, layer_norm, 替换classifier; 同步修改forward和forward_with_attention; 同步修改BiLSTM类（如果仍需兼容） |
| `src/model/train_lstm.py` | 训练策略 | Adam→AdamW, R-Drop, Mixup, CosineAnnealingWarmRestarts+Warmup |
| `src/model/dataloader.py` | 增强逻辑 | 随机时间裁剪, 关键点级dropout |

---

## 不可做的事项（红线）

1. **不得移除EMA**：EMA对泛化有正面作用，保留当前EMA配置
2. **不得移除mask机制**：mask对无效关键点的处理是正确的
3. **不得改变预处理流水线**：数据预处理（归一化、肩轴对齐等）保持不变，避免需要重新处理数据集
4. **不得改变数据集分割**：train/val/test分割保持不变
5. **不得改变关键点数量**：保持135点，通过input_projection让模型学习特征选择
6. **不得改变HDF5数据格式**：所有改动仅在模型和训练代码层面
