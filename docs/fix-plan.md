# ASL-LSTM 高准确度实验计划

## 1. 目标与约束
- **目标指标**：训练集 `>=95%`、验证集 `>=80%`、测试集 `>=75%`。
- **执行约束**：
  - 每次实验结束后必须本地提交一次 Git 记录。
  - 代码改动遵循“高内聚、低耦合”，优先模块化变更，避免跨文件散改。
  - 每次改动同步更新文档（至少更新 `docs/progress.md` 与本计划；涉及接口时更新 `docs/API.md` / `README.md`）。

## 2. 当前项目状态（基线）

### 2.1 Git 与历史实验结论
- 最近稳定实验主线：`实验2(CosineWarmRestarts)` 提升验证精度，`实验8(FocalLoss)` 已验证无效并回退。
- 历史最佳验证精度（日志复盘）：`val_acc=73.29%`（`seed123_training.log`，epoch 394，train 95.42%）。
- 历史最佳测试精度（评估报告）：`69.38%`（`logs/evaluation_report_20260220_180145.txt`，模型 `averaged_260_340.pth`）。

### 2.2 目标差距
- 验证集还需提升：`80.00 - 73.29 = 6.71` 个百分点。
- 测试集还需提升：`75.00 - 69.38 = 5.62` 个百分点。

## 3. 实验执行协议（统一）
每轮实验按以下顺序执行：
1. 定义单变量改动（仅 1 个主因子，必要时最多 1 个配套因子）。
2. 执行训练与评估（默认 `--no-tta` 口径），记录 train/val/test。
3. 更新文档：`docs/progress.md` + 本文件“实验记录”。
4. 提交本地 Git（提交信息含实验编号）。

推荐提交格式：
- `exp(train): <实验结论简述> [E0xx]`

## 4. 分阶段实验路线图

### E00（已完成）历史基线审计
- 内容：读取 git 提交链与历史日志，确定当前最优可复现基线。
- 结论：当前主瓶颈在泛化上限（val/test）。

### E01（已完成）实验基础设施解耦
- 目标：降低实验切换成本，支持命令行运行时覆盖，不频繁改 `src/config.py`。
- 代码改动：
  - 新增 `src/model/runtime_overrides.py`（集中处理运行时参数覆盖与校验）。
  - `src/model/train_lstm.py` 接入 `--epochs --learning-rate --weight-decay --dropout --label-smoothing --mixup-alpha`。
  - 新增测试 `src/test/test_runtime_overrides.py`。
- 验证：全量门禁通过（compileall + unittest + ruff）。

### E01-SMOKE（已完成）短训链路验证（非性能实验）
- 命令：
  - `uv run python src/model/train_lstm.py --run-tag exp_e01_smoke --seed 42 --epochs 5`
  - `uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/exp_e01_smoke/best_model.pth`
- 结果：
  - 训练集：`13.04%`
  - 验证集：`14.84%`
  - 测试集：`13.57%`（`logs/evaluation_report_20260221_182641.txt`）
- 结论：链路正常，可用于后续系统化参数实验；该轮不用于精度判断。

### E02（进行中）学习率周期与收敛窗口重定位
- 主因子：`cosine_T0` / `cosine_T_mult` 与 `num_epochs` 组合。
- 计划：固定其余参数，优先比较两组长周期配置（例如 `T0=40/T_mult=2` 与 `T0=50/T_mult=2`）。
- 成功标准：验证峰值突破 `74.5%`，测试达到 `>=70.5%`。
- 已执行子实验：
  - `E02-T1`（`exp_e02_t1_d030_ls001_wd3e4_lr7e4`，50 epoch）
  - 训练结果：训练 `67.48%`、验证 `55.79%`（`tool_c7fc7ae16001YX1bmqyBwqLcVf`）。
  - 测试结果：`47.29%`（`logs/evaluation_report_20260221_184028.txt`）。
  - 结论：显著低于历史基线（val 73.29% / test 69.38%），该参数组合判定为失败，不进入复用候选。

### E03（待执行）正则化平衡实验
- 主因子：`dropout`、`label_smoothing`、`weight_decay`（采用单变量+小步长）。
- 目标：在保持 `train>=95%` 的前提下抬高验证上限。

### E04（待执行）数据质量阈值与采样策略实验
- 主因子：`min_valid_ratio_per_sample` 与 `use_weighted_sampler/sampler_power`。
- 目标：提升长尾类别召回，优先提高测试集准确率。

### E05（待执行）多种子+区间平均+软投票集成
- 主因子：多种子独立训练后做 checkpoint 区间平均，再做 softmax 概率集成。
- 目标：在不改模型结构前提下提升测试稳定性与上限。

### E06（冲刺）结构小改动（仅在 E02~E05 不达标时启用）
- 候选：轻量调整 `hidden_size` / `attention_dim`，禁止引入大规模耦合重构。
- 启动条件：E02~E05 全部完成但测试仍 `<75%`。

## 5. 实验记录（滚动更新）
- [x] E00：基线审计完成。
- [x] E01：运行时覆盖基础设施完成，测试通过。
- [x] E01-SMOKE：短训链路验证完成。
- [ ] E02：进行中（`E02-T1` 已完成，待执行 `E02-T2/T3`）。
- [ ] E03：待执行。
- [ ] E04：待执行。
- [ ] E05：待执行。
- [ ] E06：待执行。
