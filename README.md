# ASL-LSTM: 美国手语识别系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **BiLSTM + Additive Attention** 的词级美国手语 (ASL) 识别系统。采用 **MediaPipe** 提取 135 个人体骨骼关键点，经过严格的空间几何归一化与动态特征工程后，输入双向 LSTM 进行时序建模，最终通过 4 模型集成推理达到 **75.97% 测试准确率**（WLASL100，100 类）。

系统提供温馨美观的 OpenCV GUI 界面，支持实时摄像头推理与离线视频推理两种模式。

---

## 目录

- [四层架构总览](#四层架构总览)
- [项目目录结构](#项目目录结构)
- [环境要求与安装](#环境要求与安装)
- [数据准备](#数据准备)
- [数据预处理流程](#数据预处理流程)
- [并行关键点提取架构](#并行关键点提取架构)
- [Z-Score 标准化](#z-score-标准化)
- [基础设施层详解](#基础设施层详解)
- [数据处理工具箱](#数据处理工具箱)
- [模型架构](#模型架构)
- [Attention 机制详解](#attention-机制详解)
- [数据加载与增强](#数据加载与增强)
- [训练工具箱](#训练工具箱)
- [模型训练](#模型训练)
- [Checkpoint 工具链](#checkpoint-工具链)
- [模型评估](#模型评估)
- [批量评估系统](#批量评估系统)
- [推理使用](#推理使用)
- [Android 移植](#android-移植)
- [实验结果](#实验结果)
- [配置中心](#配置中心)
- [单元测试体系](#单元测试体系)
- [质量门禁](#质量门禁)

---

## 四层架构总览

系统采用严格分层的 **4 层架构**，层间依赖方向单向约束，确保高内聚低耦合：

```
+------------------------------------------------------------------+
|  Interface Layer (CLI 入口)                                        |
|  main.py / train_lstm.py / evaluate_lstm.py / batch_evaluate.py   |
+------------------------------------------------------------------+
          | 调用
          v
+------------------------------------------------------------------+
|  Application Layer (业务逻辑)                                       |
|  src/model/*  ←→  src/data_process/*                               |
|  训练 / 评估 / 推理       预处理 / 迁移 / 校验                        |
+------------------------------------------------------------------+
          | 调用
          v
+------------------------------------------------------------------+
|  Domain Layer (核心算法)                                            |
|  BiLSTMAttention / Attention / 数据增强 / EMA / EarlyStopping      |
|  预处理 Pipeline / Z-Score 标准化 / TTA                             |
+------------------------------------------------------------------+
          | 调用
          v
+------------------------------------------------------------------+
|  Infrastructure Layer (公共契约)                                    |
|  src/core/paths.py     —— 统一路径构建                              |
|  src/core/labels.py    —— 标签映射加载与双向校验                     |
|  src/core/hdf5_schema.py —— HDF5 必填字段常量与校验                 |
|  src/config.py         —— 全局配置中心（所有超参数集中管理）          |
+------------------------------------------------------------------+
```

**依赖约束**：`src/core/*` **禁止依赖** `src/model/*` 或 `src/data_process/*`，反向允许。标签解析、路径构建、HDF5 Schema 验证必须且仅存在于 `src/core/*` 中。

---

## 项目目录结构

```text
asl-lstm/
├── main.py                            # 系统主入口（启动推理模式选择界面）
├── pyproject.toml                     # UV 依赖配置（Python 3.10, PyTorch cu121）
├── README.md                          # 本文档
├── init.sh                            # 环境初始化脚本（Linux/macOS）
│
├── src/                               # 核心源代码
│   ├── config.py                      # 全局配置中心（10 组 dataclass）
│   ├── core/                          # 基础设施层
│   │   ├── paths.py                   #   统一路径构建函数
│   │   ├── labels.py                  #   标签映射加载与双向契约校验
│   │   └── hdf5_schema.py            #   HDF5 必填字段常量与校验
│   ├── data_process/                  # 数据处理管道
│   │   ├── preprocess_wlasl.py        #   核心预处理：视频 → HDF5 特征（1679 行）
│   │   ├── count_dataset_samples.py   #   数据集统计工具
│   │   ├── transfer_hdf5_data.py      #   HDF5 数据迁移（rebalance）
│   │   ├── verify_consistency.py      #   数据一致性验证
│   │   ├── compare_hdf5_detail.py     #   HDF5 差异对比
│   │   └── read_data_struct.py        #   HDF5 结构查看器
│   ├── model/                         # 模型与推理引擎
│   │   ├── model_lstm.py              #   网络结构（BiLSTMAttention / BiLSTM）
│   │   ├── train_lstm.py              #   训练主循环
│   │   ├── validate_lstm.py           #   验证循环
│   │   ├── evaluate_lstm.py           #   单模型评估
│   │   ├── ensemble_evaluate.py       #   多模型集成评估（Softmax 平均）
│   │   ├── batch_evaluate.py          #   批量评估全部 checkpoint
│   │   ├── dataloader.py              #   数据加载器（增强 + Z-Score）
│   │   ├── training_utils.py          #   训练辅助（EMA / EarlyStopping）
│   │   ├── tta.py                     #   测试时增强（hflip TTA）
│   │   ├── checkpoint_utils.py        #   Checkpoint 参数平均函数
│   │   ├── average_checkpoints.py     #   Checkpoint 平均 CLI
│   │   ├── checkpoint_avg_experiment.py # E02：Checkpoint 平均实验
│   │   ├── realtime_inference.py      #   实时摄像头推理 + GUI（1502 行）
│   │   ├── offline_inference.py       #   离线视频推理 + GUI（1016 行）
│   │   ├── export_lite.py             #   导出 PyTorch Lite (.ptl) 供 Android
│   │   └── export_table.py            #   导出 Markdown 结果对比表
│   ├── test/                          # 单元测试（21 个测试文件）
│   └── mediapipe_models/              # MediaPipe 模型文件（自动下载）
│
├── dataset/
│   ├── raw/                           # 原始 WLASL 视频 + 元数据
│   └── processed/                     # 预处理后 HDF5 特征 + 统计量
│
├── src/checkpoints/                   # 模型权重
│   ├── best_model.pth
│   ├── seed42_temporal_mask/
│   ├── seed123_temporal_mask/
│   ├── seed456_temporal_mask/
│   └── seed789_temporal_mask/
│
├── results/                           # 评估结果
│   ├── summary.json                   # 全部模型评估汇总
│   └── comparison_table.md            # Markdown 对比表
│
├── picture/                           # 训练指标图
├── logs/                              # 训练日志与评估报告
├── docs/                              # 项目文档
└── Android/                           # Android 移植（Gradle 项目）
```

---

## 环境要求与安装

（同现有版本，保持不变）

### 前置条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（推荐）、Linux、macOS |
| Python | **3.10**（MediaPipe 0.10.9 强制要求） |
| GPU | NVIDIA GPU + CUDA 12.1（可选，CPU 亦可运行推理） |
| 包管理器 | [uv](https://github.com/astral-sh/uv)（强制使用） |

### 安装步骤

```bash
# 1. 安装 uv（如尚未安装）
pip install uv

# 2. 克隆项目
git clone <repo-url> && cd asl-lstm

# 3. 安装全部依赖（uv 自动创建 .venv 并锁定版本）
uv sync
```

主要依赖：`torch>=2.5.1` (cu121), `mediapipe==0.10.9`, `opencv-python>=4.12`, `h5py>=3.15`, `scikit-learn>=1.7`, `numpy>=1.24`, `pillow>=10.0`, `matplotlib>=3.10`, `protobuf==3.20.3`。

---

## 数据准备

（同现有版本，保持不变）

本项目默认使用 [WLASL (Word-Level American Sign Language)](https://dxli94.github.io/WLASL/) 数据集。

1. 从 WLASL 官方仓库下载视频文件，放置到 `dataset/raw/WLASL100/` 目录。
2. 下载 `WLASL_v0.3.json` 元数据文件到 `dataset/raw/`。
3. 准备 `nslt_100.json`（子集划分文件）到 `dataset/raw/WLASL100/`。

通过修改 `src/config.py` 中的 `_DATASET_SCALE` 常量（100/300/1000/2000），可以切换数据集规模。

---

## 数据预处理流程

预处理脚本 `src/data_process/preprocess_wlasl.py`（1679 行）实现了完整的特征提取管道，将原始视频转换为模型可用的时空特征张量。

### 总体流程

```text
原始视频 (.mp4)
  |
  v
[MediaPipe 关键点提取] → 135 点 × (x, y) 坐标
  |  KeypointExtractor / ParallelKeypointExtractor
  v
[缺失关键点插值] → 线性插值，最大间隔 8 帧
  |  _interpolate_missing_short_gaps()
  v
[EMA 坐标平滑] → 指数移动平均，α=0.35
  |  _smooth_xy()
  v
[肩轴旋转对齐] → 旋转使双肩连线水平
  |  _align_shoulder_axis()
  v
[肩躯融合尺度归一化] → 中位肩宽作为全局尺度因子
  |  _normalize_with_mask()
  v
[时域线性重采样] → 超 90 帧者重采样至 90 帧
  |  _resample_if_long()
  v
[速度特征计算] → dx, dy 一阶差分
  |  _add_velocity_features()
  v
[零填充对齐] → 统一 shape: (90, 4, 135)
  |  _pad_sequence()
  v
[HDF5 存储] → 按 Train / Val / Test 分文件存储
```

### 步骤详解

#### 1. 骨骼关键点提取

使用 Google MediaPipe Tasks API（`PoseLandmarker` + `HandLandmarker` + `FaceLandmarker`）提取 **135 个关键点**：

| 部位 | 关键点数 | 说明 |
|------|---------|------|
| Body (Pose) | 25 | OpenPose Body 25 拓扑对齐 |
| Left Hand | 21 | MediaPipe 手部 landmarks |
| Right Hand | 21 | MediaPipe 手部 landmarks |
| Face | 68 | 从 MediaPipe 478 面部点中选取 |
| **合计** | **135** | 每个点提取 (x, y) 坐标 |

MediaPipe 使用 `pose_landmarker_heavy` 模型，所有检测器置信度阈值设为 0.5。支持**手部优先检测优化**（`extract_frame_optimized()`）：若未检测到双手，跳过 Pose 和 Face 检测以加速。模型文件首次运行时自动从 Google Storage 下载。

#### 2. 缺失关键点插值

对于因遮挡或检测失败导致的短段缺失（连续 ≤8 帧），使用线性插值自动补全。超出阈值的长段缺失保持为零。

#### 3. EMA 坐标平滑

对 (x, y) 坐标施加指数移动平均（Exponential Moving Average）平滑：

$$x_t' = \alpha \cdot x_t + (1-\alpha) \cdot x_{t-1}'$$

平滑系数 α=0.35，在消除抖动噪声的同时保留手势运动的动态特征。

#### 4. 肩轴旋转对齐 (Shoulder Axis Alignment)

计算每帧双肩连线与水平线的夹角，旋转全部关键点使肩膀始终水平，消除人体倾斜带来的干扰。

#### 5. 尺度归一化 (Scale Normalization)

采用 **shoulder_torso_fusion** 策略：

- 计算全视频序列中肩宽的中位数作为全局尺度因子
- 所有坐标除以该尺度因子（加 ε=1e-6 防除零）
- 零中心化：以双肩中点为中心平移
- 实现平移不变性 + 尺度不变性（消除体型和距离差异）

#### 6. 时域线性重采样

对超过 90 帧的长序列，使用 `cv2.INTER_LINEAR` 线性插值进行等距重采样，统一到 `max_frames=90` 帧。不足 90 帧的序列保留原始长度，后续零填充对齐。

#### 7. 速度特征计算

计算相邻帧间的一阶差分：

$$dx_t = x_t - x_{t-1}, \quad dy_t = y_t - y_{t-1}$$

最终每个关键点的特征通道为 **(x, y, dx, dy)**，共 4 个通道。二阶加速度特征（ddx, ddy，E08 实验）已证明无效，保持关闭。

#### 8. 输出格式

每个样本输出形状为 `(90, 4, 135)`，存储为 HDF5 文件：

- **训练集**: `WLASL100_135-Train.hdf5`
- **验证集**: `WLASL100_135-Val.hdf5`
- **测试集**: `WLASL100_135-Test.hdf5`

同时生成训练集的 Z-Score 统计量文件 `WLASL100_train_stats.json`（4 通道的 mean/std）。

### 运行预处理

```bash
uv run python src/data_process/preprocess_wlasl.py \
    --json dataset/raw/WLASL_v0.3.json \
    --video-dir dataset/raw/WLASL100 \
    --output-dir dataset/processed \
    --prefix WLASL \
    --limit 100
```

---

## 并行关键点提取架构

`preprocess_wlasl.py` 实现了两个层级的提取器：

### KeypointExtractor（单帧提取器）

- 封装 MediaPipe Tasks API，支持 **IMAGE**（无状态，并行安全）和 **VIDEO**（带跟踪）两种模式
- `extract_frame()` — 从单帧提取全部 135 关键点
- `extract_frame_optimized()` — 手部优先优化：先检测手部，若未检测到则跳过 Pose/Face
- `_map_to_135_keypoints()` — 将 MediaPipe 原始 478 面 + 33 躯干 + 21×2 手映射为 WLASL 135 格式（Body 25 + Left Hand 21 + Right Hand 21 + Face 68）

### ParallelKeypointExtractor（并行提取器）

基于 `ThreadPoolExecutor` 的线程池并行方案，设计为流水线模式：

- 每个 worker 线程持有独立的 `KeypointExtractor(IMAGE 模式)`，避免线程安全问题
- 两种使用模式：
  - **Batch 模式** (`extract_batch()`)：一次性提交所有帧，阻塞等待全部完成。用于离线推理。
  - **Streaming 模式** (`submit_frame()` + `collect_completed()`)：逐帧提交，按序收集。用于实时推理流水线。
- 自动检测 `cpu_count - 1` 个 worker，懒初始化

---

## Z-Score 标准化

预处理完成后，`compute_train_feature_stats()` 在训练集上计算每通道的均值与标准差：

```python
# 训练集统计量示例（4 通道）
"mean": [-0.0246, -0.1196, -0.0008, 0.0025]
"std":  [0.2463, 0.6918, 0.0380, 0.0618]
```

推理 / 训练时调用 `_apply_standardization()`：

$$\hat{x} = \frac{x - \mu}{\sigma + \epsilon}$$

写入前确保均值/标准差的通道维度与数据匹配，不匹配时跳过标准化。统计数据导出为 `WLASL100_train_stats.json`，同时被 Android 端使用。

---

## 基础设施层详解

### `src/core/paths.py` — 统一路径构建

消除散落在各脚本中的路径拼接，统一入口：

| 函数 | 返回 | 说明 |
|------|------|------|
| `project_root()` | Path | 项目根目录 |
| `processed_data_root()` | Path | `dataset/processed` |
| `dataset_subset_dir(scale)` | Path | `dataset/processed/WLASL{scale}/` |
| `hdf5_file_path(scale, split)` | Path | `WLASL{scale}_135-{split}.hdf5` |
| `label_map_file_path(scale)` | Path | `wlasl_{scale}_maplabels.json` |

### `src/core/labels.py` — 标签映射契约加载

WLASL 标签映射文件存储为 JSON，格式为双向映射：

```json
{
  "id_to_label": {"0": "book", "1": "drink", ...},
  "label_to_id": {"book": 0, "drink": 1, ...}
}
```

- `load_label_to_id_map(path)` — 加载并校验 id_to_label ↔ label_to_id 为完美互逆关系
- `load_id_to_label_map(path)` — 反向加载（基于前者反转）
- 严格校验：id_to_label 的键必须是数字字符串、值必须是字符串；label_to_id 的值必须是整数
- 自动拒绝旧格式（id_to_label 键非数字字符串）

### `src/core/hdf5_schema.py` — HDF5 字段契约

定义 HDF5 数据文件的必填字段规范：

```python
REQUIRED_SAMPLE_FIELDS = ("data", "length", "label", "video_name", "width", "height")
OPTIONAL_SAMPLE_FIELDS = ("mask", "quality")
```

- `missing_required_fields(group)` — 返回缺失字段列表
- `validate_required_fields(group)` — 校验必填字段，缺失则抛 ValueError

---

## 数据处理工具箱

`src/data_process/` 下提供多个 CLI 数据管理工具：

### `count_dataset_samples.py` — 数据集统计

遍历 HDF5 分片文件，统计每个文件及总计的样本数。

### `transfer_hdf5_data.py` — HDF5 数据迁移

用于数据集 rebalance：从源 HDF5（如 Val）按比例随机抽取样本迁移到目标 HDF5（如 Train）。

- 支持随机种子、源端删除选项、迁移后校验
- 适用于样本分布不均衡时调整训练/验证集比例

### `verify_consistency.py` — 数据一致性验证

将预处理后数据的以下维度与官方 WLASL 元数据对比：

- 标签映射（maplabels）完整性
- 各分片样本计数匹配
- 标签分布合理性
- video ID 集合一致性

### `compare_hdf5_detail.py` — HDF5 差异对比

对两个 HDF5 文件逐样本对比：数据结构、数据形状、统计量（min/max/mean/std）、标签映射、首帧关键点值。

### `read_data_struct.py` — HDF5 结构查看器

CLI 交互式查看 HDF5 文件：列出所有样本键、字段存在性、数据形状、quality 统计。

---

## 模型架构

### BiLSTMAttention（主模型）

```text
输入: (batch, 90, 540)                   # 90帧 × (4通道 × 135关键点)
  ↓
Pack Padded Sequence                      # 变长序列打包（pack_padded_sequence）
  ↓
BiLSTM (2层, hidden=128, bidirectional)   # 输出维度: 128×2 = 256
  ↓                                        # 层间 Dropout = 0.35
Pad Packed Sequence                        # 解包（pad_packed_sequence）
  ↓
LayerNorm (256)                            # E04：层归一化，稳定特征分布
  ↓
Additive Attention (Bahdanau, dim=32)      # 自适应加权各帧重要性
  ↓                                        # 输出: (batch, 256)
Dropout (p=0.35)
  ↓
Linear (256 → 100)                         # 100 类分类头
  ↓
输出: (batch, 100)                         # Logits
```

### BiLSTM（Baseline，无 Attention）

不使用 Attention，直接使用最后一层双向 LSTM 的 **final hidden state** `h_n` 拼接作为分类特征：

- 前向最后一层 state: `h_n[-1, 0, :, :]`  →  `(batch, 128)`
- 反向最后一层 state: `h_n[-1, 1, :, :]`  →  `(batch, 128)`
- 拼接后 → `(batch, 256)` → Dropout → Linear(256 → 100)

#### 模型参数量

| 变体 | hidden_size | num_heads | 参数量 |
|------|-------------|-----------|--------|
| 标准 | 128 | 1 | ~1,115,812 |
| 加宽 | 192 | 1 | ~2,066,852 |

标准模型约 **1.1M 参数量**，轻量高效。

### 核心组件

| 组件 | 说明 |
|------|------|
| **BiLSTM** | 2 层双向 LSTM，hidden_size=128，双向拼接后输出维度 256。使用 `pack_padded_sequence` 处理变长序列，零填充位置不参与计算。 |
| **LayerNorm** | E04 添加：对 LSTM 各时间步输出做层归一化，消除尺度漂移，稳定 Attention 前的特征分布。 |
| **Additive Attention** | Bahdanau 注意力：`Linear(256→32)` → `tanh` → `Linear(32→1)` → Softmax → 加权求和。支持 padding mask。 |
| **MultiHeadAttention** | E06 实验：多头缩放点积注意力。`num_heads=4` 时 head_dim=64。实验证明对小数据集有害（100 类过少），回退至单头。 |
| **分类头** | `Dropout(0.35)` → `Linear(256→100)`。 |

---

## Attention 机制详解

### 单头加性注意力 (Additive / Bahdanau Attention)

`src/model/model_lstm.py:Attention`

```python
energy = tanh(Linear(lstm_output))                    # (batch, seq, hidden) → (batch, seq, attn_dim)
scores = Linear(energy).squeeze(-1)                   # (batch, seq, attn_dim) → (batch, seq)
scores = masked_fill(~mask, -inf)                     # 填充位置 → 负无穷
weights = softmax(scores, dim=1)                      # (batch, seq)
context = sum(weights.unsqueeze(-1) * lstm_output)    # (batch, hidden_dim)
```

- 输出：context vector `(batch, 256)` + attention weights `(batch, seq_len)`
- `forward_with_attention()` — 同时返回注意力权重用于可视化分析

### 多头缩放点积注意力 (Multi-Head, E06)

`src/model/model_lstm.py:MultiHeadAttention`

```text
Q, K, V = Linear(lstm_output)                                 # (batch, seq, 256)
Q, K, V = split_heads() → (batch, num_heads, seq, head_dim)   # head_dim = 256/num_heads
scores = Q @ K^T / sqrt(head_dim)                              # (batch, h, seq, seq)
scores = masked_fill(~key_mask, -inf)                           # padding mask
attn_weights = softmax(scores, dim=-1)
attended = attn_weights @ V                                    # (batch, h, seq, head_dim)
output = merge_heads() + out_proj                               # (batch, seq, 256)
context = mean_pool(output, mask)                               # (batch, 256)
```

实验结果：多头对小数据集（WLASL100）有害，最终采用单头加性注意力。

---

## 数据加载与增强

### CSLDataset

`src/model/dataloader.py:CSLDataset`

训练时 HDF5 数据全部加载到内存（`data_cache`），每样本含：

```python
{
    "feature": np.ndarray,    # (T, C, V) 原始关键点数据
    "label_id": int,          # 类别 ID
    "length": int,            # 真实帧数
    "mask": np.ndarray,       # 每帧关键点有效掩码（可选）
    "quality": float,         # 质量评分（可选，低于 min_valid_ratio=0.35 则过滤）
}
```

### 在线增强 Pipeline

`_apply_augmentation()` 按顺序应用以下增强：

| 增强方式 | 参数 | 实现位置 |
|---------|------|---------|
| 时间扭曲 | prob=16%, speed∈[0.90,1.10] | `_random_time_warp()` — `np.interp` 重采样 |
| 随机丢帧 | prob=10%, max_drop=8% | `_random_frame_dropout()` — 替换为前一帧 |
| 时间掩码 (E05) | prob=30%, max_mask=15% | `_temporal_mask()` — 整段置零 |
| 随机旋转 | ±18° | 旋转矩阵作用于 (x, y) 和 (dx, dy) |
| 随机缩放 | [0.88, 1.12] | 统一缩放系数 |
| 随机平移 | ±0.10 | translation offset |
| 高斯噪声 | σ=0.003 | 逐元素加噪 |
| 水平翻转 | prob=25% | x → -x（零中心）/ 1-x（非零中心）；dx → -dx；左右关键点交换 |

**左右语义交换**：`swap_left_right_keypoints()` 交换 Left Hand(25:46) ↔ Right Hand(46:67) 及 11 组身体左右对称点。

### Z-Score 标准化

`preprocess_keypoints()` → `_ensure_feature_channels()` → `_apply_standardization()`：

- 确保通道数：2 通道→自动补速度特征→4 通道；6 通道→按需保留
- 训练集统计量标准化
- 裁剪/零填充至 `max_frames=90`
- 展平为 `(90, 540)` ← `reshape(90, -1)`

### preprocess_keypoints 流程

```text
输入: (T, C, V) np.ndarray
  → _ensure_feature_channels()     # 确保 4/6 通道
  → 裁剪或零填充至 90 帧
  → _apply_standardization()       # Z-Score
  → reshape(90, -1)                # (90, 540)
  → 返回 torch.Tensor + valid_len
```

### get_dataloaders

工厂函数 `get_dataloaders()` 创建三个 DataLoader（train/val/test），支持可选 `WeightedRandomSampler`（实验证明无效，默认关闭）。

---

## 训练工具箱

### `set_global_seed(seed)`

统一设置 Python random + numpy + torch + CUDA 随机种子，配置 cuDNN 确定性 / benchmark 标志，确保实验可复现。

### ModelEMA

`src/model/training_utils.py:ModelEMA`

指数滑动平均（Exponential Moving Average）：

```python
ema_param = decay * ema_param + (1 - decay) * model_param
```

- decay=0.999，epoch 6 后开始更新
- 维护一份深拷贝的 shadow model，仅在 optimizer.step() 后更新
- 保存 checkpoint 时使用 EMA 参数（验证集精度更高）

### EarlyStopping

`src/model/training_utils.py:EarlyStopping`

- 监控 `val_acc`（max 模式）
- patience=150，min_delta=0.0
- 连续 150 个 epoch 验证精度未创新高则停止训练

---

## 模型训练

### 训练策略

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | lr=8e-4, weight_decay=4e-4 |
| 学习率调度 | CosineAnnealingWarmRestarts | T0=30, T_mult=2（周期：30→60→120→...） |
| 最小学习率 | 3e-6 | 学习率下限 |
| Batch Size | 4 (实际) / 16 (等效) | 梯度累积 4 步 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| EMA | decay=0.999, start_epoch=6 | 指数滑动平均，提升泛化 |
| 标签平滑 | 0.03 | CrossEntropyLoss label_smoothing |
| 早停 | patience=150, metric=val_acc | 长时间无提升自动停止 |
| 最大轮数 | 600 | |
| 随机种子 | 42（默认，支持 --seed 覆盖） | |

### 运行训练

```bash
# 标准训练
uv run python src/model/train_lstm.py

# 指定随机种子 + 运行标签
uv run python src/model/train_lstm.py --seed 456 --run-tag seed456_temporal_mask

# 过拟合诊断模式（关闭增强、正则、早停、EMA、标签平滑等）
uv run python src/model/train_lstm.py --overfit-debug
```

训练过程中自动保存：
- `best_model.pth`：验证集最佳模型（EMA 版本）
- `epoch_N.pth`：每 5 个 epoch 定期保存

---

## Checkpoint 工具链

### `checkpoint_utils.py` — 参数平均函数

`average_state_dicts(state_dicts)` — 对多个 PyTorch state_dict 逐 key 做算术平均（均在 CPU 上进行）：

```python
avg[key] = (dict1[key] + dict2[key] + ... + dictN[key]) / N
```

自动校验所有 state_dict 的 key 集合一致，不一致则报错。

### `average_checkpoints.py` — Checkpoint 平均 CLI

```bash
uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260
```

对指定 epoch 范围的 checkpoint 做参数平均，输出平均后的模型权重。

### `checkpoint_avg_experiment.py` — E02 实验

评估所有 epoch checkpoint → 按 val 精度排序 → Top-K 参数平均 → 测试集评估：

1. `discover_checkpoints()` — 列出全部 epoch checkpoint
2. `evaluate_checkpoints_on_val()` — 逐个在验证集上评估
3. `average_top_k_checkpoints()` — 取 Top-K 平均
4. `ensemble_evaluate_averaged_models()` — 多种子集成评估

### `export_lite.py` — PyTorch Lite 导出

将训练好的模型导出为 `.ptl` 格式供 Android 使用：

```bash
# 标准导出
uv run python src/model/export_lite.py

# INT8 动态量化导出（体积缩小约 4 倍，推理加速 2-3 倍）
uv run python src/model/export_lite.py --quantize
```

导出流程：
1. 加载 checkpoint → `BiLSTMAttention()` → `eval()`
2. 可选：INT8 动态量化（`torch.ao.quantization.quantize_dynamic`，量化 nn.LSTM + nn.Linear）
3. `torch.jit.trace()` 或 `torch.jit.script()`
4. `optimize_for_mobile()` 优化
5. `_save_for_lite_interpreter()` 导出 `.ptl`
6. 验证输出形状 + 量化前后余弦相似度
7. 导出 Z-Score 统计量 JSON（供 Android 端使用）

### `export_table.py` — 结果表格导出

从 `results/summary.json` 读取全部评估结果，自动生成 Markdown 对比表格：

- 核心性能指标表（Top-1 / Top-5 / Macro P/R/F1）
- 模型架构参数表（参数量 / Input / Hidden / Heads / LayerNorm）
- 统计汇总（基线均值 / TM 均值 / 集成结果）

---

## 模型评估

### 单模型评估

```bash
uv run python src/model/evaluate_lstm.py --model-path src/checkpoints/best_model.pth --no-tta
```

生成 `sklearn.metrics.classification_report` 输出每类别 Precision / Recall / F1，保存至 `logs/evaluation_report_*.txt`。

### 多模型集成评估

`src/model/ensemble_evaluate.py:ensemble_evaluate()`

**策略**：Softmax 概率平均（Softmax Averaging）

```text
输入 N 个模型，每个模型对同一样本输出 logits
  → softmax(logits) → probs_i        # 每个模型 → 概率分布
  → avg_probs = mean(probs_1..probs_N) # 概率平均
  → prediction = argmax(avg_probs)     # 取平均后最大概率类
```

支持水平翻转 TTA（`use_tta_hflip=True`）：每模型额外生成翻转后 softmax，共 2N 个分布平均。

```bash
uv run python src/model/ensemble_evaluate.py \
    --models src/checkpoints/seed42_temporal_mask/best_model.pth \
             src/checkpoints/seed123_temporal_mask/best_model.pth \
             src/checkpoints/seed456_temporal_mask/best_model.pth \
             src/checkpoints/seed789_temporal_mask/best_model.pth
```

集成模型路径在 `cfg.EVALUATION.ensemble_model_paths` 中配置。

### TTA (Test-Time Augmentation)

`src/model/tta.py:build_hflip_tta_batch()`

- 对 batch 做水平翻转：x → -x（零中心模式），dx → -dx
- 可选左右关键点交换（`hflip_swap_lr=True`）
- 支持 4ch（x, y, dx, dy）和 6ch（+ddx, ddy）通道

---

## 批量评估系统

`src/model/batch_evaluate.py`（717 行）实现了完整的自动批量评估 pipeline：

### 自动模型参数检测

`_detect_model_params()` — 从 `state_dict` 的 key 中自动推断模型架构参数：

```python
input_size     # 从 lstm.weight_ih_l0 的 shape[1] 推断
hidden_size    # 从 lstm.weight_hh_l0 的 shape[0] 推断
num_layers     # 从 l0, l1, ... 后缀数量推断
bidirectional  # 从 l0 和 l0_reverse 是否存在推断
use_layer_norm # 从 layer_norm.* 是否存在推断
attention_type # 从 attention.* 的 key 判断单头/多头
```

### 实验特殊配置覆盖

不同实验子目录可能需要不同的模型参数/数据配置，通过 `EXPERIMENT_OVERRIDES` 字典处理：

```python
EXPERIMENT_OVERRIDES = {
    "seed456_accel6ch": {"input_size": 810, "enable_accel_feature": True, ...},
    "seed456_multihead": {"num_heads": 4, ...},
    "seed456_bilstm192": {"hidden_size": 192, ...},
}
```

### 评估流程

1. 扫描 `src/checkpoints/` 所有子目录中的 `best_model.pth`
2. 对每个模型：参数检测 → 加载 → 测试集评估 → 保存 `result.json` + `predictions.json`
3. 4 模型集成评估（软编码路径）
4. 汇总写入 `results/summary.json`

---

## 推理使用

### 一键启动

```bash
uv run python main.py
```

运行后打开暖色调启动器界面，可选择两种推理模式。

### 实时推理（Real-time Inference）

`src/model/realtime_inference.py:run_realtime_inference()`

**Pipeline**：

```text
摄像头帧采集 (640×480@30fps, 水平镜像翻转)
  ↓ (逐帧)
submit_frame() → ParallelKeypointExtractor 流水线
  ↓ (非阻塞)
collect_completed() → 按序收集关键点
  ↓ (每 inference_interval=3 帧)
手部有效 → 加入帧缓冲队列 (deque, maxlen=90)
  ↓
prepare_sequence():
    PreprocessHelper.process_video_sequence()    # 插值 → 平滑 → 对齐 → 归一化 → 重采样 → 速度 → 填充
    preprocess_keypoints()                       # Z-Score → reshape → 张量化
  ↓
_ensemble_predict():                             # N 模型 Softmax 平均
  ↓
draw_modern_ui():                                # 暖色调 GUI + FPS + 结果卡片 + 概率条
```

**特性**：
- 多线程并行关键点提取（`ParallelKeypointExtractor`，自动检测 CPU 核数）
- 手部优先检测策略优化速度
- 每 3 帧执行一次集成推理
- 界面显示实时 FPS、骨骼叠加开关、结果卡片含概率条

### 离线推理（Offline Inference）

`src/model/offline_inference.py:run_offline_mode()`

```text
文件对话框（tkinter）选择 .mp4 视频
  ↓
读取全部帧到内存
  ↓
ParallelKeypointExtractor.extract_batch()        # 批量并行提取
  ↓
PreprocessHelper + ensemble inference             # 全量预处理 + 模型推理
  ↓
视频回放 + 骨骼叠加 + 结果卡片显示
```

### UI 操作

| 操作 | 方式 |
|------|------|
| 退出 | 点击「退出」按钮 或 按 `q` 键 |
| 骨骼开关 | 点击「显示骨骼」/「隐藏骨骼」按钮 |
| 导入视频 | 点击「导入视频」按钮（离线模式） |

### UI 设计风格

界面采用温馨简约的暖色调设计，OpenCV 原生渲染（无外部 GUI 框架）：

| 颜色 | RGB | 用途 |
|------|-----|------|
| 米色背景 | (253, 246, 237) | 主背景 |
| 珊瑚色 | (255, 170, 150) | 按钮 |
| 金色 | (245, 210, 140) | 功能按钮 |
| 深棕色 | (80, 65, 55) | 文字 |
| 毛玻璃卡片 | 半透明白 | 结果展示 |

骨骼系统：135 点骨架线 + 关键点渲染，支持一键切换显示/隐藏。

---

## Android 移植

系统提供 Android 端推理能力（开发中），Gradle 项目位于 `Android/`：

```text
Android/
├── app/src/main/java/com/ye/asl_lstm/    # Java 源码
├── app/src/main/assets/                   # 部署资源
│   ├── best_model.ptl                     #   INT8 量化模型（~1MB）
│   ├── zscore_stats.json                  #   Z-Score 标准化统计量
│   └── *.task                             #   MediaPipe 模型文件
├── model/                                 # 模型管理模块
├── UI/                                    # Android UI 模块
└── openspec/                              # OpenSpec 开发规范
```

Android 端部署流程：

1. `export_lite.py --quantize` 导出 INT8 量化模型 + Z-Score 统计量
2. 将 `.ptl` 和 `zscore_stats.json` 放入 Android assets
3. Android 端完成：摄像头采集 → MediaPipe 关键点提取 → 预处理（归一化 + Z-Score）→ LSTM 推理 → 结果展示

---

## 实验结果

> **评估日期**: 2026-05-15 | 测试集 258 样本 | 无 TTA | 当前模型架构: BiLSTM+Attention+LayerNorm (hidden=128, 4ch)
>
> 标注说明：✅ 本次实测 | 🔗 历史记录（模型架构已迭代，当前 config 无法直接加载）

### WLASL100 测试集 Top-1 / Top-5

| 实验分组 | 模型 | Top-1 (%) | Top-5 (%) | 备注 |
|---------|------|-----------|-----------|------|
| **基线模型** | Seed=42 | 66.28 | 90.31 | 🔗 |
| | Seed=123 | 67.05 | 89.53 | 🔗 |
| | Seed=456 | 69.38 | 90.31 | 🔗 |
| | Seed=789 | 65.89 | 90.70 | 🔗 |
| | Seed=2024 | — | — | 🔗 |
| | Seed=4096 | — | — | 🔗 |
| | **基线平均** | **67.38** | — | |
| **时间掩码 E05** | Seed=42 + TM | **73.26** | 92.64 | ✅ |
| | Seed=123 + TM | **74.03** | 92.64 | ✅ |
| | Seed=456 + TM | **74.42** | 91.86 | ✅ 单模型最佳 |
| | Seed=789 + TM | **74.81** | 92.25 | ✅ |
| | **TM 平均** | **74.13 (+6.75%)** | — | |
| **架构变体** | + LayerNorm (E04) | 70.16 | 91.47 | ✅ seed456_layernorm |
| | + MultiHead k=4 (E06) | 72.09 | 89.92 | 🔗 模型键名不兼容 |
| | Hidden=192 (E07a) | 71.71 | 91.47 | 🔗 shape 不匹配 |
| | + Accel 6ch (E08) | 71.71 | 91.47 | 🔗 input_size=810 vs 540 |
| | + LR Warmup (E03) | 65.89 | 89.92 | 🔗 |
| **模型集成** | **4-Model Ensemble (Softmax Avg)** | **75.97** | **94.57** | ✅ seed42/123/456/789 + TM |

### 关键实验记录

| 实验 | 变更 | 结果 |
|------|------|------|
| E02 | Checkpoint 参数平均 | 稳定性提升，单独评估不显著 |
| E03 | LR Warmup | 有害，默认关闭 |
| E04 | LayerNorm | 训练更稳定，纳入基线 |
| E05 | **时间掩码 (Temporal Mask) + 多种子集成** | **75.97%（最佳）** |
| E06 | 多头注意力 num_heads=4 | 对小数据集有害，回退单头 |
| E07a | 加宽 hidden=192 | 过拟合加重，回退 128 |
| E08 | 二阶加速度特征 (ddx, ddy) | 实验失败，回退 4 通道 |

### 关键洞察

- **Temporal Mask 增强（E05）是最大贡献点**：单模型平均提升 +6.75%，将零填充引入训练信号，强迫模型学习上下文依赖
- **LayerNorm（E04）稳定训练**：减小 Attention 层前数值尺度差异
- 多头注意力、宽 hidden 在小数据上过拟合严重
- 4 模型集成（42/123/456/789 + TM）Softmax 平均达到 **75.97%**

---

## 配置中心

所有超参数集中管理于 `src/config.py`，使用 `@dataclass(frozen=True)` 实现 **10 组不可变配置**，通过 `cfg.<分组>.<字段>` 分层访问：

### 完整配置清单

#### PathsConfig — 路径配置
```python
# 项目根目录、数据目录、模型目录、日志目录等 17 个路径
project_root / processed_data_dir / label_map_path
train_data_path / val_data_path / test_data_path
model_save_dir / test_model_path / raw_data_dir
default_json_path / default_video_dir / log_dir  # ...
```

#### SequenceConfig — 序列配置
```python
max_frames=90           # 统一时序长度
num_landmarks=135       # 单帧关键点数量
enable_accel_feature=False  # E08 实验失败，关闭
base_feature_channels=("x", "y", "dx", "dy")  # 4 通道

# 自动计算属性
landmark_dim=4          # len(base_feature_channels)
input_size=540          # landmark_dim * num_landmarks = 4 * 135
num_classes=100         # 等于 _DATASET_SCALE
```

#### PreprocessConfig — 预处理配置
```python
pipeline_version="v3"   # 预处理流水线版本
enable_missing_interp=True   # 短缺失插值
interp_max_gap=8             # 最大插值长度
min_valid_ratio_per_sample=0.35  # 低质量样本过滤阈值
enable_shoulder_axis_align=True   # 肩轴对齐
scale_mode="shoulder_torso_fusion" # 尺度归一化策略
smooth_ema_alpha=0.35       # EMA 平滑系数
enable_standardize=True     # Z-Score 标准化
feature_stats_path=...      # 训练集统计量路径
```

#### MediapipeConfig — MediaPipe 配置
```python
# 模型路径（自动下载） + 检测阈值
pose_min_det_conf=0.5
hand_min_det_conf=0.5
face_min_det_conf=0.5
# 全部 detection / presence / tracking 阈值均为 0.5
```

#### ModelConfig — 模型配置
```python
hidden_size=128         # LSTM 隐层维度
num_layers=2            # LSTM 层数
bidirectional=True      # 双向
dropout=0.35            # Dropout
label_smoothing=0.03   # 标签平滑
use_attention=True      # 使用注意力
attention_dim=32        # 注意力中间维度
use_layer_norm=True     # E04: LayerNorm
num_heads=1             # 单头加性注意力
```

#### AugmentationConfig — 数据增强配置
```python
rotation_range=18.0     # ±18 度
hflip_prob=0.25         # 翻转概率 25%
temporal_mask_prob=0.30 # E05: 时间掩码概率
time_warp_prob=0.16     # 时间扭曲概率
frame_dropout_prob=0.10 # 丢帧概率
noise_std=0.003         # 高斯噪声标准差
```

#### TrainingConfig — 训练配置
```python
batch_size=4
learning_rate=8e-4
weight_decay=4e-4
num_epochs=600
grad_accum_steps=4      # 等效 batch_size=16
grad_clip_max_norm=1.0
scheduler_type="cosine_warm"
cosine_T0=30            # 重启周期
cosine_T_mult=2         # 周期倍增
ema_decay=0.999
ema_start_epoch=6
early_stopping_patience=150
warmup_epochs=0         # E03 失败，关闭
mixup_alpha=0.0         # Mixup 关闭
```

#### EvaluationConfig — 评估配置
```python
ensemble_model_paths=(
    "seed456_temporal_mask/best_model.pth",
    "seed123_temporal_mask/best_model.pth",
    "seed789_temporal_mask/best_model.pth",
    "seed42_temporal_mask/best_model.pth",
)
use_tta_hflip=False
verbose_report=True
save_report=True
```

#### InferenceConfig — 推理配置
```python
camera_index=0
camera_width=640
camera_height=480
camera_fps=60
inference_interval=3    # 每 3 帧推理一次
parallel_workers=0      # 0=自动 (cpu_count-1)
```

#### UIConfig — UI 配置
```python
font_size=32
chinese_font_paths=("msyh.ttc", "simhei.ttf")
exit_button_text="退出"
skeleton_button_text_on="隐藏骨骼"
skeleton_button_text_off="显示骨骼"
skeleton_point_color=(0, 255, 0)    # 绿色
skeleton_line_color=(255, 255, 0)   # 黄色
```

### 数据集规模切换

修改 `_DATASET_SCALE` 常量即可切换：

```python
_DATASET_SCALE = 100  # 支持 100 / 300 / 1000 / 2000
```

所有路径、类别数、nslt JSON 路径自动适配。

---

## 单元测试体系

`src/test/` 包含 **21 个测试文件**，按功能分类：

### 核心契约测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_config_structure.py` | 配置 dataclass 不可变性、必填字段完整性、类型正确性 |
| `test_label_map_contract.py` | 标签映射双向校验、旧格式拒绝、缺失字段异常 |
| `test_model_input_contract.py` | 模型输入输出维度校验、变长序列支持 |
| `test_core_modules.py` | `paths.py` / `labels.py` / `hdf5_schema.py` 函数行为 |
| `test_no_legacy_cfg_access.py` | 禁止旧版全局常量访问模式 |

### 数据处理测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_preprocess_smoke.py` | 预处理 pipeline 冒烟测试 |
| `processed_data_test.py` | 预处理后数据完整性校验 |
| `test_read_data_struct.py` | HDF5 结构查看器功能 |
| `check_coords.py` | 关键点坐标合理性检查 |
| `check_json.py` | JSON 格式与字段完整性 |

### 数据加载与增强测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_dataloader_pipeline.py` | DataLoader 完整加载、Z-Score 标准化 |
| `test_augmentation_simple.py` | 几何/时序增强功能正确性 |
| `test_sampling_strategy.py` | 类别均衡采样权重计算 |
| `test_training_profile.py` | overfit-debug 模式配置正确性 |

### 模型训练测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_training_utils.py` | EMA / EarlyStopping / set_global_seed 行为 |
| `test_checkpoint_utils.py` | checkpoint 平均函数正确性 |
| `test_tta_utils.py` | TTA 翻转 + 左右关键点交换 |
| `test_train_smoke.py` | 训练循环冒烟（小 batch overfit） |

### 推理与 UI 测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_inference_smoke.py` | 推理 pipeline 冒烟 |
| `test_ui.py` | UI 组件渲染与交互 |

### 环境测试

| 测试文件 | 覆盖内容 |
|---------|---------|
| `gpu_cuda_check.py` | GPU / CUDA 可用性检测 |

---

## 质量门禁

```bash
# 编译检查（确保无语法错误）
uv run python -m compileall src

# 单元测试（21 个测试文件）
uv run python -m unittest discover -s src/test -p "test_*.py"

# Lint 检查（ruff，line-length=100，target=py310）
uv run ruff check src
```

---

## 关于数据集

默认采用 **WLASL (Word-Level American Sign Language)** 数据集。请前往 [WLASL 官方仓库](https://dxli94.github.io/WLASL/) 获取视频原文件。支持在 `src/config.py` 中一键切换 WLASL100 / WLASL300 / WLASL1000 / WLASL2000 配置（通过修改 `_DATASET_SCALE` 常量）。
