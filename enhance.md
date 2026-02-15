# WLASL100 下一步优化计划（66% 验证准确率阶段）

## 1. 当前状态与结论

## 1.1 现状（基于最近训练）

- 验证集准确率稳定在 **66% 左右**，峰值约 **68%**。
- 训练集准确率可达 **99%**。
- 训练/验证曲线出现明显分叉：训练持续上升，验证较早进入平台。

## 1.2 关键判断

- 当前核心瓶颈不是“提不出特征”，而是“**泛化能力不足（过拟合）**”。
- 因此下一步优先顺序应为：
  1. **先优化训练策略与模型泛化能力**（高收益、低风险）
  2. **再做针对性预处理消融**（避免大改预处理导致结果不可归因）

---

## 2. 总体目标（下一阶段）

## 2.1 指标目标

- Top-1 验证准确率：从 66% 提升到 **70%+**（第一目标）
- Macro-F1：提升并与 Top-1 同步改善
- 训练-验证差距：从 ~30%+ 收敛到 **20% 内**

## 2.2 工程目标

- 训练配置可复现：固定 seed，记录完整配置与日志
- 每次实验“单变量改动”，保证可解释
- 保持现有 `dx/dy` 主线，不推翻现有有效成果

---

## 3. 决策：先优化哪里？

## 3.1 优先级结论

**优先优化神经网络训练策略/结构，不优先大改预处理。**

理由：

1. 训练 99% vs 验证 66%，典型过拟合范式。
2. 预处理已把准确率从 55% 拉升到 66%，说明信息提取已明显改善。
3. 再大改预处理会引入额外变量，短期不利于定位增益来源。

---

## 4. 分阶段实施计划（建议 3 周）

## Phase A（第 1 周）：训练泛化优先

目标：不改主结构，先把泛化做好。

### A1. 训练流程改造（必须）

1. 早停（Early Stopping）
   - 监控 `val_acc` 或 `val_loss`
   - patience 建议 20~30
   - 达到早停即结束，不再固定跑满 500 epoch

2. 最佳模型保存与恢复
   - 严格使用 best checkpoint 做最终测试
   - 禁止用最后 epoch 结果代表模型

3. 梯度裁剪
   - `clip_grad_norm_` 建议 `1.0`

4. 梯度累积（提升等效 batch）
   - 目标等效 batch 16 或 32
   - 减少梯度噪声，提升验证稳定性

### A2. 正则化与损失（单变量实验）

1. Label Smoothing：`0.10 -> 0.03/0.05` 对照
2. Dropout：`0.40 -> 0.30/0.35` 对照
3. Weight Decay：`1e-3 -> 5e-4 / 2e-3` 对照
4. 尝试 Focal Loss（可选）

### A3. 类别平衡策略

1. `WeightedRandomSampler`（训练集）
2. 或 Class-Balanced Loss（择一优先，不要同时上）

---

## Phase B（第 2 周）：轻量结构升级

目标：在 Phase A 基础上，小步提升模型表达能力。

### B1. 轻量时序前端

1. 在 LSTM 前增加 1D Temporal Conv（2~3 层）
2. 核大小建议 3/5，保持参数量可控

### B2. Attention 改造（保守）

1. 先验证 single-head temporal attention
2. 再尝试 multi-head（仅 2~4 头，避免过拟合）

### B3. 消融策略

1. `BiLSTM`（基线）
2. `TCN + BiLSTM`
3. `BiLSTM + Improved Attention`

每个结构跑 3 个 seed，报告均值和方差。

---

## Phase C（第 3 周）：定点预处理优化

目标：只做“高价值、低扰动”的预处理消融。

### C1. 预处理消融（建议顺序）

1. `body+hands+face` vs `body+hands`（face 噪声评估）
2. 质量阈值 `MIN_VALID_RATIO_PER_SAMPLE`：0.35 / 0.40 / 0.45
3. 平滑强度 `SMOOTH_EMA_ALPHA`：0.25 / 0.35 / 0.45

### C2. 保持约束

- 不改动数据划分原则
- 不引入不可解释的大规模数据清洗
- **不得使用 `src/data_process/transfer_hdf5_data.py`**

---

## 5. 实验矩阵（可直接执行）

## 5.1 第一批（建议先跑）

| 实验ID | 改动 | 预期效果 | 优先级 |
|---|---|---|---|
| E7 | EarlyStopping + GradClip + BestCkpt | 降低过拟合、训练稳定 | P0 |
| E8 | LabelSmoothing=0.03 | 提升泛化 | P0 |
| E9 | Dropout=0.30 | 缩小 train-val gap | P0 |
| E10 | WeightedRandomSampler | 提升弱类召回 | P1 |
| E11 | TCN + BiLSTM | 提升时序建模能力 | P1 |

## 5.2 第二批（按结果再开）

| 实验ID | 改动 | 预期效果 | 优先级 |
|---|---|---|---|
| E12 | body+hands 消融 | 评估 face 噪声 | P1 |
| E13 | 质量阈值 0.40/0.45 | 提升数据纯度 | P1 |
| E14 | EMA alpha 消融 | 优化动态细节保留 | P2 |
| E15 | Multi-head attention(2~4) | 提升复杂动作区分 | P2 |

---

## 6. 代码改动建议（按文件）

1. `src/model/train_lstm.py`
   - 增加 EarlyStopping
   - 增加 Grad Clip
   - 增加梯度累积
   - 完善 best model 保存逻辑

2. `src/config.py`
   - 新增训练策略开关与参数（patience、clip、accum steps）
   - 增加实验 ID 记录项

3. `src/model/model_lstm.py`
   - 逐步加入轻量 TCN 前端（可配置开关）

4. `src/model/evaluate_lstm.py`
   - 增加 macro-F1、per-class recall、confusion matrix 输出

5. `src/model/dataloader.py`
   - 增加 WeightedRandomSampler 入口（可开关）

---

## 7. 验收标准（Definition of Done）

满足以下全部条件才算完成一轮优化：

1. 验证准确率均值 >= 68%，且至少一次 >= 70%
2. train-val gap 明显下降（目标 <= 20%）
3. 3-seed 结果方差可接受（避免偶然提升）
4. 输出完整实验报告（配置、曲线、指标、结论）

---

## 8. 执行纪律

1. 每次只改一个核心变量。
2. 每个实验都保存 `terminal-output.md` 与训练曲线图。
3. 若实验失败，回退到最近稳定配置，不叠加继续试。
4. 优先保证可复现性，再追求更高峰值。
