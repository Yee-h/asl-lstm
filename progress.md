# ASL-LSTM 开发进度追踪

## 项目状态概览
- **当前阶段**: 优化与稳定 (Optimization & Stabilization)
- **最近基线**: Val Acc ~70.62% (Commit `f968dd6`)
- **关键里程碑**: 
  - [x] 配置中心化重构 (Config Unification)
  - [x] Mask-aware 数据预处理 (E01)
  - [ ] 过拟合抑制 (Phase A)
  - [ ] 模型结构微调 (Phase B)

## 功能开发日志

### F001: 架构与数据一致性验证 (P0)
- **状态**: 待验证 (Pending Verification)
- **说明**: 代码已合并 (`f968dd6`)，需由 Agent 运行测试确认环境与逻辑一致性，建立可信基线。

### F002: 训练策略优化 (P1)
- **状态**: 待开始
- **计划**: 
  - 调整 Dropout 和 Weight Decay。
  - 引入更激进的 Label Smoothing。
  - 实验不同采样策略解决长尾问题。

### F003: 模型结构微调 (P2)
- **状态**: 待开始
- **计划**: LayerNorm, MLP Head, Hidden Size Tuning.

## 实验记录 (Experiment Log)

| ID | 描述 | Val Acc | Train Acc | Gap | 结论 |
|---|---|---|---|---|---|
| E01 | Mask-aware Preprocessing + Config Refactor | 70.62% | ~99% | ~29% | 基线建立，准确率回升，但仍有过拟合 |
