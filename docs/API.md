# ASL-LSTM API 文档

## 1. 概述

本文档描述 ASL-LSTM 项目所有脚本入口的接口参数、输入输出格式与数据契约。

---

## 2. 脚本入口

### 2.1 系统主入口

- **脚本**: `main.py`
- **作用**: 启动推理模式选择界面（启动器），支持实时推理与离线推理两种模式。
- **参数**: 无
- **运行**:
  ```bash
  uv run python main.py
  ```

### 2.2 预处理入口

- **脚本**: `src/data_process/preprocess_wlasl.py`
- **作用**: 从原始视频提取 MediaPipe 关键点，经过完整预处理管道后生成 HDF5 特征数据。
- **参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--json` | str | `cfg.PATHS.default_json_path` | WLASL 元数据 JSON 路径 |
| `--video-dir` | str | `cfg.PATHS.default_video_dir` | 视频文件目录 |
| `--output-dir` | str | `cfg.PATHS.default_output_dir` | 输出根目录 |
| `--prefix` | str | `cfg.PATHS.default_output_prefix` | 输出文件名前缀 |
| `--limit` | int | `cfg.PATHS.dataset_scale` | 类别数量上限 |

- **输出文件**:
  - `{prefix}{scale}_135-Train.hdf5` — 训练集特征
  - `{prefix}{scale}_135-Val.hdf5` — 验证集特征
  - `{prefix}{scale}_135-Test.hdf5` — 测试集特征
  - `wlasl_{scale}_maplabels.json` — 标签映射文件
  - `{prefix}{scale}_train_stats.json` — Z-Score 统计量（均值、标准差）

- **预处理管道版本**: v3

### 2.3 训练入口

- **脚本**: `src/model/train_lstm.py`
- **作用**: 基于 HDF5 特征数据训练 BiLSTMAttention 模型。
- **参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--overfit-debug` | flag | False | 过拟合诊断模式（关闭增强、Dropout、标签平滑、权重衰减、早停） |

- **训练策略**: Adam (lr=8e-4, wd=4e-4) + CosineAnnealingWarmRestarts (T0=30, T_mult=2) + EMA (decay=0.999, start=epoch 6) + 梯度累积 (4步) + 梯度裁剪 (norm=1.0) + 标签平滑 (0.03) + 早停 (patience=150, metric=val_acc)
- **输出**:
  - `src/checkpoints/best_model.pth` — 验证集最佳模型（EMA 版本）
  - `src/checkpoints/epoch_N.pth` — 定期保存（每 5 epoch）
  - `logs/` — 训练日志

### 2.4 评估入口

- **脚本**: `src/model/evaluate_lstm.py`
- **作用**: 加载模型 checkpoint，在测试集上输出分类报告与准确率。
- **参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | `cfg.EVALUATION.model_path` | 待评估 checkpoint 路径 |
| `--no-tta` | flag | True | 关闭水平翻转 TTA |

- **运行**:
  ```bash
  uv run python src/model/evaluate_lstm.py --model-path src/checkpoints/best_model.pth --no-tta
  ```

### 2.5 集成评估入口

- **脚本**: `src/model/ensemble_evaluate.py`
- **作用**: 加载 4 个不同随机种子训练的模型，Softmax 概率平均后评估测试集。
- **集成模型路径**: 由 `cfg.EVALUATION.ensemble_model_paths` 配置。
- **集成策略**: 对各模型输出的 logits 做 Softmax，取 4 组概率向量的算术平均，再 argmax。
- **运行**:
  ```bash
  uv run python src/model/ensemble_evaluate.py
  ```

### 2.6 Checkpoint 平均入口

- **脚本**: `src/model/average_checkpoints.py`
- **作用**: 对多个 epoch 的 checkpoint 做参数平均，导出新模型。
- **参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start-epoch` | int | 必填 | 参与平均的起始 epoch |
| `--end-epoch` | int | 必填 | 参与平均的结束 epoch |
| `--step` | int | 5 | Checkpoint 步长 |
| `--output` | str | 自动生成 | 输出 checkpoint 路径 |

- **运行**:
  ```bash
  uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260
  ```

### 2.7 实时推理入口

- **脚本**: `src/model/realtime_inference.py`
- **作用**: 摄像头实时手语识别，4 模型集成推理 + GUI 渲染。
- **参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--camera` | int | `cfg.INFERENCE.camera_index` (0) | 摄像头索引 |
| `--video` | str | 无 | 视频文件路径（替代摄像头） |

- **实时优化**: 
  - **多线程并行关键点提取**：使用 `ParallelKeypointExtractor` 类，自动检测 CPU 核心数（`cpu_count - 1`），以流水线模式（`submit_frame()` + `collect_completed()`）并行提取 MediaPipe 关键点
  - 手部优先检测、每 3 帧推理一次
  - IMAGE 模式 MediaPipe（无状态，可并行）

### 2.8 离线推理入口

- **脚本**: `src/model/offline_inference.py`
- **作用**: 导入视频文件进行离线手语识别推理。
- **调用方式**: 通过 `main.py` 启动器界面选择"离线推理"模式，或在代码中直接调用。
- **并行优化**: 使用 `ParallelKeypointExtractor.extract_batch()` 多线程并行提取关键点，与实时推理共享同一实现。

### 2.9 数据检查与统计工具

| 脚本 | 作用 | 主要参数 |
|------|------|---------|
| `src/data_process/count_dataset_samples.py` | 统计各 split 样本数与类别分布 | `--dataset-scale`, `--split`, `--data-root` |
| `src/test/processed_data_test.py` | 预处理后数据质量检查 | `--dataset-scale`, `--split`, `--data-root`, `--sample-limit` |
| `src/test/check_json.py` | JSON 元数据校验 | `--dataset-scale`, `--split`, `--data-root` |
| `src/test/check_coords.py` | 关键点坐标范围检查 | `--dataset-scale`, `--split`, `--data-root` |
| `src/data_process/read_data_struct.py` | HDF5 内部结构查看 | 直接运行 |
| `src/data_process/compare_hdf5_detail.py` | 两个 HDF5 文件逐字段对比 | 文件路径参数 |
| `src/data_process/verify_consistency.py` | 跨 split 数据一致性验证 | 数据路径参数 |

---

## 3. 核心类 API

### 3.1 ParallelKeypointExtractor

**模块**: `src/data_process/preprocess_wlasl.py`

多线程并行关键点提取器，供实时推理与离线推理共享使用。

**初始化参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_workers` | int | 0 | 线程池工作线程数，0 表示自动检测（`cpu_count - 1`） |

**核心方法**:

| 方法 | 签名 | 用途 |
|------|------|------|
| `extract_batch()` | `(frames, progress_callback=None) → List[Tuple]` | 离线批量提取：提交所有帧，按顺序返回结果 |
| `submit_frame()` | `(frame) → Future` | 实时流水线：提交单帧到线程池，非阻塞 |
| `collect_completed()` | `() → List[Tuple]` | 实时流水线：收集已完成结果（按提交顺序） |
| `close()` | `() → None` | 关闭线程池，释放所有 per-thread KeypointExtractor 实例 |

**返回值结构**（`extract_batch` / `collect_completed`）:

```python
Tuple[np.ndarray, np.ndarray, bool]  # (keypoints, valid_mask, has_hands)
# keypoints: (135, 2) float32，关键点坐标
# valid_mask: (135,) bool，各关键点有效性
# has_hands: bool，是否检测到手部
```

**内部实现**:
- 使用 `ThreadPoolExecutor` 管理 worker 线程
- 每个 worker 通过 `threading.local()` 持有独立的 `KeypointExtractor(use_video_mode=False)` 实例
- MediaPipe 使用 IMAGE 模式（无状态），支持帧级并行
- 延迟初始化：线程池在首次 `submit_frame` 时创建

---

## 3. 数据契约

### 3.1 HDF5 样本结构

每个 HDF5 文件内部按样本索引（数字字符串）组织：

```
/                           # 根
├── "0"/                    # 第 0 个样本
│   ├── data                # float32, shape (T, C, 135)
│   ├── length              # int64, 有效帧长度（未 padding 前）
│   ├── label               # string, gloss 标签（如 "book"）
│   ├── video_name          # string, 原始视频路径
│   ├── width               # int64, 视频宽度
│   ├── height              # int64, 视频高度
│   ├── mask (可选)          # uint8, 关键点有效掩码
│   └── quality (可选)       # float32, 样本质量分数
├── "1"/                    # 第 1 个样本
│   └── ...
└── ...
```

**字段详解**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `data` | float32 | `(T, 4, 135)` | T=max_frames(90)，4通道=(x, y, dx, dy)，135关键点 |
| `length` | int64 | 标量 | 有效帧数（重采样/填充前的原始长度，用于 pack_padded_sequence） |
| `label` | string | — | gloss 标签文本 |
| `video_name` | string | — | 来源视频文件名或路径 |
| `width` | int64 | 标量 | 原始视频宽度（像素） |
| `height` | int64 | 标量 | 原始视频高度（像素） |
| `mask` | uint8 | 可变 | （可选）关键点有效性掩码 |
| `quality` | float32 | 标量 | （可选）样本质量评分 |

### 3.2 标签映射文件

- **路径**: `cfg.PATHS.label_map_path`
- **文件名示例**: `wlasl_100_maplabels.json`

**格式契约**（强制互逆）:

```json
{
  "id_to_label": {
    "0": "book",
    "1": "drink",
    "2": "computer"
  },
  "label_to_id": {
    "book": 0,
    "drink": 1,
    "computer": 2
  }
}
```

**校验规则**:
- `id_to_label` 键必须是数字字符串，值必须是标签字符串。
- `label_to_id` 键必须是标签字符串，值必须是整数 ID。
- 两个映射必须**完全互逆**，否则视为非法契约。
- 旧格式（如 `id_to_label: {"book": 0}`）会被解析器拒绝并抛出明确异常。
- 统一解析入口: `src/core/labels.py`。

### 3.3 Z-Score 统计量文件

- **路径**: `cfg.PREPROCESS.feature_stats_path`
- **文件名示例**: `WLASL100_train_stats.json`

**格式**:

```json
{
  "mean": [...],
  "std": [...]
}
```

- 数组长度 = `input_size`（当前为 540 = 4 通道 × 135 关键点）。
- 在训练和推理前，对每个样本的特征执行: `(x - mean) / (std + eps)`，其中 `eps = 1e-6`。

---

## 4. 模型输入输出契约

### 4.1 模型输入

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `x` | Tensor (float32) | `(batch, 90, 540)` | Z-Score 标准化后的特征序列 |
| `lengths` | Tensor (int64) | `(batch,)` | 各样本有效帧长度 |

### 4.2 模型输出

| 输出 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `logits` | Tensor (float32) | `(batch, 100)` | 各类别原始得分（未 Softmax） |

`forward_with_attention()` 额外返回注意力权重 `(batch, seq_len)`。

---

## 5. 配置 API

所有配置通过 `import src.config as cfg` 访问。配置使用 `@dataclass(frozen=True)` 定义，运行时不可修改。

| 分组 | 访问方式 | 关键字段 |
|------|---------|---------|
| 路径 | `cfg.PATHS` | `project_root`, `processed_data_dir`, `label_map_path`, `train_data_path`, `model_save_dir` |
| 序列 | `cfg.SEQUENCE` | `max_frames=90`, `num_landmarks=135`, `input_size=540`, `num_classes=100` |
| 预处理 | `cfg.PREPROCESS` | `smooth_ema_alpha=0.35`, `scale_mode="shoulder_torso_fusion"`, `interp_max_gap=8` |
| MediaPipe | `cfg.MEDIAPIPE` | `pose_model_path`, `hand_model_path`, `face_model_path`, 所有阈值=0.5 |
| 模型 | `cfg.MODEL` | `hidden_size=128`, `num_layers=2`, `dropout=0.35`, `attention_dim=32`, `num_heads=1` |
| 增强 | `cfg.AUGMENTATION` | `rotation_range=18.0`, `hflip_prob=0.25`, `temporal_mask_prob=0.30` |
| 训练 | `cfg.TRAINING` | `batch_size=4`, `learning_rate=8e-4`, `grad_accum_steps=4`, `ema_decay=0.999` |
| 评估 | `cfg.EVALUATION` | `ensemble_model_paths` (4条路径), `use_tta_hflip=False` |
| 推理 | `cfg.INFERENCE` | `camera_index=0`, `camera_width=640`, `inference_interval=3` |
| UI | `cfg.UI` | `font_size=32`, `exit_button_text="退出"`, `skeleton_button_text_on="隐藏骨骼"` |

---

## 6. 统一质量门禁

每个开发任务结束后按顺序执行：

```bash
# 1. 编译检查（确保无语法错误）
uv run python -m compileall src

# 2. 单元测试
uv run python -m unittest discover -s src/test -p "test_*.py"

# 3. Lint 检查（Ruff, line-length=100, target=py310）
uv run ruff check src
```

---

## 7. 推荐评估流程

完整的训练→评估→集成流程：

```bash
# 1. 标准训练
uv run python src/model/train_lstm.py

# 2. 单模型评估
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/best_model.pth

# 3. Checkpoint 参数平均
uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260

# 4. 平均模型评估
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/averaged_160_260.pth

# 5. 4 模型集成评估
uv run python src/model/ensemble_evaluate.py
```

建议记录：最佳 `val_acc` 与对应 epoch、`best_model` 与 `averaged` 模型的测试集准确率。
