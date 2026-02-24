# ASL-LSTM 项目优化提升方案

> **核心约束**: 本项目为本科毕业论文选题，模型架构必须基于 **BiLSTM + Attention**，不接受 Transformer 替代方案。所有优化必须在 BiLSTM 框架内进行。

## 一、项目改进路径回顾 (7e7fc84 → d7bd8f4)

### 1.1 基础架构修复阶段 (d66c7cd → 15c6fff)

| 版本 | 主要改进 | 验证精度 | 测试精度 | 结论 |
|------|----------|----------|----------|------|
| d66c7cd | 改进预处理管道，mask-aware增强 | 66% | - | 基础改进 |
| f968dd6 | 稳定E01管道，复现性修复 | 70.62% | - | 验证恢复 |
| 618ec45 | 全量验收，数据契约统一 | - | - | 架构稳定 |
| f5f3858 | 过拟合诊断，优化计划生成 | 71.22% | 69.38% | **基线确立** |

**关键发现**：存在严重过拟合（train 96% vs val 69%），根因分析：
- 输入维度过大（540维 = 135点 × 4通道）
- 正则化不足
- 数据增强保守
- 学习率衰减过快

---

### 1.2 优化实验阶段 (a405260 → 094ae98)

| 实验 | 配置变更 | 验证精度 | 测试精度 | 结论 |
|------|----------|----------|----------|------|
| Phase 1 | attention_dim↑64, AdamW, Mixup α=0.3, TTA | 65.28% | - | ❌ 失败 |
| Phase 2+3 | 回退正则化，禁用Mixup | 71.22% | - | ⚠️ 回归基线 |
| **实验2** | CosineAnnealingWarmRestarts (T0=30, T_mult=2) | **72.70%** | - | ✅ **成功** |

**成功经验**：
- CosineWarmRestarts 周期性重启帮助模型逃离局部最优
- 过拟合缺口从25%缩小到~8%
- 验证精度提升 +1.48%

**失败教训**：
- Mixup 对小数据集（WLASL100）有害，生成样本质量不足
- AdamW + 高 weight_decay 在小数据集上过于激进
- attention_dim=64 导致过拟合加剧

---

### 1.3 高级优化尝试阶段 (cdddf6a → d7bd8f4)

| 实验 | 方法 | 验证精度 | 测试精度 | 结论 |
|------|------|----------|----------|------|
| 实验6 | SWA (start=100, lr=5e-5) | 66.77% | 62.40% | ❌ 失败 |
| 实验8 | Focal Loss (γ=1.0) | - | 68.22% | ❌ 失败 |
| 实验9 | 多种子集成基础设施 | - | 69.38% | 🔄 待验证 |

**失败分析**：
- **SWA**: 过早接管学习率调度，导致性能下降
- **Focal Loss**: 该数据集类别不平衡不严重，Focal Loss无优势

---

## 二、当前最优配置

```python
# 模型架构
hidden_size = 128
num_layers = 2
bidirectional = True
attention_dim = 32
dropout = 0.35

# 训练配置
scheduler_type = "cosine_warm"
cosine_T0 = 30
cosine_T_mult = 2
learning_rate = 8e-4
weight_decay = 4e-4
label_smoothing = 0.03
use_ema = True
ema_decay = 0.999

# 数据增强
rotation_range = 18.0
scale_min/max = 0.88/1.12
translate = 0.10
hflip_prob = 0.25
time_warp_prob = 0.16
```

**当前性能**：
- 验证集最佳：72.70%
- 测试集最佳：69.38%
- 模型参数量：1,115,300

---

## 三、下一步优化提升方案

### 3.1 短期优化（预计提升 1-3%）

#### 方案 A：多种子集成评估（已就绪）
**优先级**: P0 | **预期提升**: +0.5~1.5%

基础设施已完成（ensemble_evaluate.py），建议：

```bash
# 运行多种子训练
python -m src.model.train_lstm --seed 42 --run-tag seed42
python -m src.model.train_lstm --seed 123 --run-tag seed123
python -m src.model.train_lstm --seed 456 --run-tag seed456
python -m src.model.train_lstm --seed 789 --run-tag seed789
python -m src.model.train_lstm --seed 2024 --run-tag seed2024

# 集成评估
python -m src.model.ensemble_evaluate \
    --models checkpoints/seed42/best_model.pth \
              checkpoints/seed123/best_model.pth \
              checkpoints/seed456/best_model.pth \
              checkpoints/seed789/best_model.pth \
              checkpoints/seed2024/best_model.pth
```

#### 方案 B：检查点平均优化
**优先级**: P0 | **预期提升**: +0.5~1%

当前检查点平均策略简单，建议：
1. 在验证精度峰值附近的 epoch 窗口选择检查点
2. 使用 SWA 式的指数加权平均而非简单平均
3. 实现验证集导向的检查点筛选

```python
# 建议实现：智能检查点选择
def select_checkpoints_by_val_acc(checkpoint_dir, top_k=5, window=20):
    """选择验证精度最高的 top_k 个检查点"""
    # 基于验证精度排序，取 top_k
    # 同时考虑 epoch 分布，避免过度集中在某一阶段
    pass
```

#### 方案 C：学习率 Warmup 策略
**优先级**: P1 | **预期提升**: +0.3~0.8%

当前直接使用目标学习率，建议添加 warmup：

```python
# 建议配置
warmup_epochs = 5
warmup_start_lr = 1e-5

# 前5个epoch线性增长到目标学习率
# 可配合 CosineAnnealingWarmRestarts 使用
```

---

### 3.2 中期优化（预计提升 2-4%）

#### 方案 D：模型架构增强
**优先级**: P1 | **预期提升**: +1~2%

##### D1. 多头注意力机制
```python
class MultiHeadAttention(nn.Module):
    """多头注意力替代单一注意力"""
    def __init__(self, hidden_dim, attention_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.attention_heads = nn.ModuleList([
            Attention(hidden_dim, self.head_dim) 
            for _ in range(num_heads)
        ])
        self.fc = nn.Linear(attention_dim, hidden_dim)
```

##### D2. 层归一化增强
```python
# 在 LSTM 输出后添加 LayerNorm
self.layer_norm = nn.LayerNorm(lstm_output_dim)

# 在 forward 中
lstm_output = self.layer_norm(lstm_output)
context, weights = self.attention(lstm_output, mask)
```

##### D3. 残差连接
```python
# 分类头添加残差
self.residual_fc = nn.Linear(input_size, num_classes)
# out = fc_out + residual_fc(x.mean(dim=1))  # 可选
```

#### 方案 E：数据增强升级
**优先级**: P1 | **预期提升**: +0.5~1.5%

##### E1. 时间掩码 (Temporal Masking)
```python
# 随机掩盖一段时间区间，增强时序鲁棒性
def temporal_mask(x, mask_prob=0.1, max_mask_len=10):
    # 类似 SpecAugment 的时序掩码
    pass
```

##### E2. Cutout 变体
```python
# 随机掩盖部分关键点
def keypoint_cutout(x, cutout_prob=0.15, max_cutout_ratio=0.2):
    # 随机选择关键点子集置零
    pass
```

##### E3. MixUp 变体（小数据集友好版）
```python
# 仅在相似类别间进行 MixUp
def class_aware_mixup(x, y, label_similarity_matrix, alpha=0.2):
    # 只混合语义相近的类别
    pass
```

#### 方案 F：训练策略优化
**优先级**: P1 | **预期提升**: +0.5~1%

##### F1. 渐进式训练
```python
# 阶段1: 冻结 LSTM，只训练分类头 (5 epochs)
# 阶段2: 解冻全部，低学习率微调 (20 epochs)
# 阶段3: 全参数训练 (剩余 epochs)
```

##### F2. 梯度惩罚
```python
# 添加梯度范数惩罚，提升泛化能力
def gradient_penalty(model, x, lengths, lambda_gp=1.0):
    # 计算输入梯度，添加 L2 惩罚
    pass
```

---

### 3.3 长期优化（预计提升 2-4%）

#### 方案 G：BiLSTM 架构深度优化
**优先级**: P2 | **预期提升**: +1~2%

##### G1. 增加 LSTM 层数与隐藏维度
```python
# 当前: hidden_size=128, num_layers=2
# 方案: hidden_size=192, num_layers=3
# 注意: 需配合更强的正则化（dropout=0.4）

# 评估不同配置组合:
# - 128x2 (当前基线)
# - 192x2
# - 128x3
# - 192x3
```

##### G2. 双向 LSTM 特征融合优化
```python
class BiLSTMAttentionV2(nn.Module):
    """改进的双向特征融合"""
    def __init__(self):
        # 方案1: 拼接最后时刻的正向和反向隐状态
        # 方案2: 加权融合所有时间步
        # 方案3: 添加 Highway 连接
        
        self.highway_gate = nn.Linear(lstm_output_dim * 2, lstm_output_dim)
        
    def forward(self, x, lengths):
        # Highway 连接: gate * lstm_output + (1-gate) * attention_output
        gate = torch.sigmoid(self.highway_gate(concat_feat))
        output = gate * attention_context + (1 - gate) * lstm_last_hidden
        return output
```

##### G3. 残差 LSTM (Residual LSTM)
```python
class ResidualLSTM(nn.Module):
    """带残差连接的 LSTM 层"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size * 2, 
                    hidden_size, bidirectional=True, batch_first=True)
            for i in range(num_layers)
        ])
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size * 2) 
            if i > 0 else None 
            for i in range(num_layers)
        ])
    
    def forward(self, x, lengths):
        for i, lstm in enumerate(self.lstms):
            out, _ = lstm(x)
            if self.projections[i] is not None:
                x = out + self.projections[i](x)  # 残差连接
            else:
                x = out
        return x
```

#### 方案 H：自监督预训练（基于 BiLSTM）
**优先级**: P2 | **预期提升**: +1~2%

```python
# 利用无标签数据预训练 BiLSTM 编码器
class BiLSTMContrastive(nn.Module):
    """对比学习预训练 - 保持 BiLSTM 架构"""
    def __init__(self):
        self.encoder = BiLSTMAttention(...)  # 核心: 仍是 BiLSTM
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1, x2, lengths1, lengths2):
        # 同一视频的不同增强版本
        z1 = self.projector(self.encoder(x1, lengths1))
        z2 = self.projector(self.encoder(x2, lengths2))
        # NT-Xent loss
        return contrastive_loss(z1, z2)

# 预训练后保留 encoder，替换 projector 为分类头进行微调
```

#### 方案 I：知识蒸馏（BiLSTM 内部）
**优先级**: P2 | **预期提升**: +0.5~1%

```python
# 用多种子训练的大容量 BiLSTM 蒸馏到当前模型
# 或用 hidden_size=256 的教师蒸馏到 hidden_size=128 的学生
class BiLSTMDistillation:
    """BiLSTM 知识蒸馏"""
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher = teacher  # 冻结的 BiLSTM 教师
        self.student = student  # 待训练的 BiLSTM 学生
        self.temperature = temperature
        self.alpha = alpha
    
    def distill_loss(self, student_logits, teacher_logits, labels):
        # 软标签损失 (蒸馏)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

---

### 3.4 数据层面优化

#### 方案 J：特征工程优化
**优先级**: P1 | **预期提升**: +0.5~1%

> **注意**: 特征工程不改变 BiLSTM 核心架构，仅优化输入表示。

##### J1. 二阶动态特征 (ddx, ddy)
当前已预留但关闭，建议重新评估：

```python
# 当前: base_feature_channels = ("x", "y", "dx", "dy")  # 4通道
# 建议: base_feature_channels = ("x", "y", "dx", "dy", "ddx", "ddy")  # 6通道

# 需验证加速度特征是否带来提升
# 注意：会增加输入维度 540 → 810
```

##### J2. 关键点置信度特征
```python
# 添加 MediaPipe 检测的置信度作为额外通道
# confidence_channels = ("left_hand_conf", "right_hand_conf", "pose_conf")
```

##### J3. 关键点分组嵌入
```python
# 为不同身体部位添加可学习的位置嵌入
# 左手、右手、身体、面部各有独立嵌入
# 注意: 这是输入层面的嵌入，不影响 BiLSTM 核心结构
class KeypointEmbedding(nn.Module):
    def __init__(self, num_keypoints=135, embed_dim=16):
        # 身体部位分组嵌入
        self.body_embed = nn.Embedding(1, embed_dim)  # 身体
        self.left_hand_embed = nn.Embedding(1, embed_dim)  # 左手
        self.right_hand_embed = nn.Embedding(1, embed_dim)  # 右手
        self.face_embed = nn.Embedding(1, embed_dim)  # 面部
```

---

## 四、实验优先级排序

| 优先级 | 方案 | 预期提升 | 实现难度 | 风险 |
|--------|------|----------|----------|------|
| **P0** | A: 多种子集成 | +0.5~1.5% | 低 | 低 |
| **P0** | B: 检查点平均优化 | +0.5~1% | 低 | 低 |
| P1 | C: LR Warmup | +0.3~0.8% | 低 | 低 |
| P1 | D2: LayerNorm | +0.5~1% | 低 | 低 |
| P1 | E1: 时间掩码 | +0.5~1% | 中 | 低 |
| P1 | D1: 多头注意力 | +1~2% | 中 | 中 |
| P2 | G: BiLSTM深度优化 | +1~2% | 中 | 中 |
| P2 | H: 自监督预训练 | +1~2% | 高 | 高 |
| P2 | I: 知识蒸馏 | +0.5~1% | 中 | 低 |

> **架构约束**: 所有方案均在 BiLSTM + Attention 框架内，不涉及 Transformer 替代。

---

## 五、推荐实验路线图

### 第一周：快速迭代
1. ✅ 运行多种子集成（方案A）
2. ✅ 优化检查点平均策略（方案B）
3. ✅ 添加 LR Warmup（方案C）

**目标**: 测试精度 69.38% → 71%+

### 第二周：架构微调
1. 添加 LayerNorm（方案D2）
2. 实现时间掩码增强（方案E1）
3. 测试多头注意力（方案D1）

**目标**: 测试精度 71% → 73%+

### 第三周：深度优化
1. 评估 BiLSTM 深度优化（方案G）
2. 测试特征工程优化（方案J）
3. 探索残差 LSTM（方案G3）

**目标**: 测试精度 73% → 75%+

---

## 六、风险与注意事项

### 6.1 已知失败方法（避免重复）
- ❌ Mixup（小数据集有害）
- ❌ AdamW + 高 weight_decay（过于激进）
- ❌ SWA（学习率调度冲突）
- ❌ Focal Loss（类别不平衡不严重）
- ❌ attention_dim=64（过拟合）

### 6.2 实验原则
1. **单变量控制**: 每次实验只改一个配置
2. **完整记录**: 所有实验结果记录在 `docs/progress.md`
3. **三振出局**: 同一方向失败3次后放弃
4. **回退机制**: 保持基线可随时恢复

### 6.3 过拟合监控
- 持续监控 train-val gap
- 若 gap > 20%，需增加正则化
- 若 gap < 5%，可适当增加模型容量

---

## 七、总结

**核心约束**: 本项目为本科毕业论文选题，模型架构必须基于 **BiLSTM + Attention**。

**当前状态**: 验证集 72.70%，测试集 69.38%

**短期目标（1-2周）**: 测试集 71-73%
**中期目标（1个月）**: 测试集 73-75%
**长期目标（3个月）**: 测试集 75%+

**核心策略**: 
1. 优先执行低风险、高回报方案（P0）
2. 所有架构改进保持在 BiLSTM 框架内
3. 架构改进走"小步快跑"路线
4. 保持实验可复现性
5. 持续监控过拟合风险

---

*文档版本: v2.0*
*更新日期: 2026-02-24*
*基于 commit: d7bd8f49c3466ecff352cbcf5a5828bdf8af4432*
