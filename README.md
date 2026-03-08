# ASL-LSTM: 美国手语识别系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **BiLSTM + Additive Attention** 的词级美国手语 (ASL) 识别系统。采用 **MediaPipe** 提取 135 个人体骨骼关键点，经过严格的空间几何归一化与动态特征工程后，输入双向 LSTM 进行时序建模，最终通过 4 模型集成推理达到 **75.97% 测试准确率**（WLASL100，100 类）。

系统提供温馨美观的 OpenCV GUI 界面，支持实时摄像头推理与离线视频推理两种模式。

---

## 目录

- [项目目录结构](#项目目录结构)
- [环境要求与安装](#环境要求与安装)
- [数据准备](#数据准备)
- [数据预处理流程](#数据预处理流程)
- [模型架构](#模型架构)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [推理使用](#推理使用)
- [实验结果](#实验结果)
- [配置中心](#配置中心)

---

## 项目目录结构

```text
asl-lstm/
├── main.py                            # 系统主入口（启动推理模式选择界面）
├── pyproject.toml                     # UV 依赖配置（Python 3.10, PyTorch cu121）
├── README.md                          # 本文档
├── init.sh                            # 环境初始化脚本
│
├── src/                               # 核心源代码
│   ├── config.py                      # 全局配置中心（所有超参数、路径、UI 设置）
│   ├── core/                          # 基础设施层
│   │   ├── paths.py                   #   统一路径构建函数
│   │   ├── labels.py                  #   标签映射加载与契约校验
│   │   └── hdf5_schema.py             #   HDF5 必填字段常量与校验函数
│   ├── data_process/                  # 数据处理管道
│   │   ├── preprocess_wlasl.py        #   核心预处理脚本（视频 → HDF5 特征）
│   │   ├── count_dataset_samples.py   #   数据集统计工具
│   │   ├── transfer_hdf5_data.py      #   HDF5 数据迁移工具
│   │   ├── verify_consistency.py      #   数据一致性验证
│   │   ├── compare_hdf5_detail.py     #   HDF5 差异对比
│   │   └── read_data_struct.py        #   HDF5 结构查看器
│   ├── model/                         # 模型与推理引擎
│   │   ├── model_lstm.py              #   网络结构定义（BiLSTMAttention / BiLSTM）
│   │   ├── train_lstm.py              #   训练主循环
│   │   ├── evaluate_lstm.py           #   单模型评估
│   │   ├── ensemble_evaluate.py       #   多模型集成评估
│   │   ├── dataloader.py              #   数据加载器（增强、Z-Score 标准化）
│   │   ├── realtime_inference.py      #   实时推理模块（摄像头 + UI）
│   │   ├── offline_inference.py       #   离线推理模块（视频文件 + UI）
│   │   ├── average_checkpoints.py     #   Checkpoint 参数平均
│   │   ├── checkpoint_avg_experiment.py # Checkpoint 平均实验脚本
│   │   ├── checkpoint_utils.py        #   Checkpoint 工具函数
│   │   ├── training_utils.py          #   训练辅助工具（EMA、早停等）
│   │   ├── validate_lstm.py           #   验证循环
│   │   └── tta.py                     #   测试时增强（TTA）
│   ├── test/                          # 单元测试与功能验证（21 个测试文件）
│   └── mediapipe_models/              # MediaPipe 模型文件（自动下载）
│
├── dataset/                           # 数据存放区
│   ├── raw/                           #   原始 WLASL 视频（需自行准备）
│   │   ├── WLASL100/                  #     视频文件 + nslt_100.json
│   │   └── WLASL_v0.3.json            #     WLASL 元数据
│   └── processed/                     #   预处理后输出
│       └── WLASL100/                  #     HDF5 特征 + 标签映射 + 统计量
│
├── src/checkpoints/                   # 模型权重（训练时生成）
│   ├── best_model.pth                 #   当前最佳单模型
│   ├── seed42_temporal_mask/          #   集成模型 Seed 42
│   ├── seed123_temporal_mask/         #   集成模型 Seed 123
│   ├── seed456_temporal_mask/         #   集成模型 Seed 456
│   └── seed789_temporal_mask/         #   集成模型 Seed 789
│
├── logs/                              # 训练日志与评估报告
└── docs/                              # 项目文档
    ├── Product-Spec.md                #   产品需求文档
    ├── Architecture.md                #   架构文档
    ├── API.md                         #   API 接口文档
    ├── feature_list.json              #   任务清单与进度
    └── progress.md                    #   开发进度日志
```

---

## 环境要求与安装

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

主要依赖：`torch>=2.5.1` (cu121), `mediapipe==0.10.9`, `opencv-python>=4.12`, `h5py>=3.15`, `scikit-learn>=1.7`, `numpy>=1.24`, `pillow>=10.0`。

---

## 数据准备

本项目默认使用 [WLASL (Word-Level American Sign Language)](https://dxli94.github.io/WLASL/) 数据集。

1. 从 WLASL 官方仓库下载视频文件，放置到 `dataset/raw/WLASL100/` 目录。
2. 下载 `WLASL_v0.3.json` 元数据文件到 `dataset/raw/`。
3. 准备 `nslt_100.json`（子集划分文件）到 `dataset/raw/WLASL100/`。

通过修改 `src/config.py` 中的 `_DATASET_SCALE` 常量（100/300/1000/2000），可以切换数据集规模。

---

## 数据预处理流程

预处理脚本 `src/data_process/preprocess_wlasl.py` 实现了完整的特征提取管道，将原始视频转换为模型可用的时空特征张量。

### 总体流程

```text
原始视频 (.mp4)
  → MediaPipe 关键点提取 (135 点 × 2D 坐标)
    → 缺失关键点插值 (最大间隔 8 帧)
      → EMA 坐标平滑 (α=0.35)
        → 肩轴旋转对齐
          → 肩躯融合尺度归一化
            → 时域线性重采样 (→ 90 帧)
              → 速度特征计算 (dx, dy)
                → 零填充对齐 (→ 90 帧)
                  → 输出: (90, 4, 135) float32
```

### 步骤详解

#### 1. 骨骼关键点提取

使用 Google MediaPipe 提取 **135 个关键点**，结构如下：

| 部位 | 关键点数 | 说明 |
|------|---------|------|
| Body (Pose) | 25 | OpenPose 25-点拓扑对齐 |
| Left Hand | 21 | 手部 21 关节点 |
| Right Hand | 21 | 手部 21 关节点 |
| Face | 68 | 面部轮廓点（嘴型、表情） |
| **合计** | **135** | 每个点提取 (x, y) 坐标 |

MediaPipe 使用 `pose_landmarker_heavy` 模型，所有检测器置信度阈值设为 0.5。

#### 2. 缺失关键点插值

对于因遮挡或检测失败导致的短段缺失（连续 ≤8 帧），使用线性插值自动补全。超出阈值的长段缺失保持为零。

#### 3. EMA 坐标平滑

对 (x, y) 坐标施加指数移动平均（Exponential Moving Average）平滑：

$$x_t' = \alpha \cdot x_t + (1-\alpha) \cdot x_{t-1}'$$

平滑系数 α=0.35，在消除抖动噪声的同时保留手势运动的动态特征。

#### 4. 肩轴旋转对齐 (Shoulder Axis Alignment)

计算每帧双肩连线与水平线的夹角，旋转全部关键点使肩膀始终水平，消除人体倾斜带来的干扰。

#### 5. 尺度归一化 (Scale Normalization)

采用 **肩躯融合 (shoulder_torso_fusion)** 策略：

- 计算全视频序列中**肩宽的中位数**作为全局尺度因子
- 所有坐标除以该尺度因子
- 实现平移不变性（双肩中点归零）+ 尺度不变性（体型归一）

#### 6. 时域线性重采样

对超过 90 帧的长序列，使用 `cv2` 线性插值进行等距重采样，将变长序列统一到 **max_frames=90** 帧。不足 90 帧的序列保留原始长度，后续零填充对齐。

#### 7. 速度特征计算

计算相邻帧间的一阶差分：

$$dx_t = x_t - x_{t-1}, \quad dy_t = y_t - y_{t-1}$$

最终每个关键点的特征通道为 **(x, y, dx, dy)**，共 4 个通道。

#### 8. 输出格式

每个样本输出形状为 `(90, 4, 135)`，存储为 HDF5 文件：
- **训练集**: `WLASL100_135-Train.hdf5`
- **验证集**: `WLASL100_135-Val.hdf5`
- **测试集**: `WLASL100_135-Test.hdf5`

同时生成训练集的 Z-Score 统计量文件 `WLASL100_train_stats.json`（均值与标准差），供训练和推理时标准化使用。

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

## 模型架构

### BiLSTMAttention 网络

```text
输入: (batch, 90, 540)                      # 90帧 × (4通道 × 135关键点)
  ↓
Pack Padded Sequence                         # 变长序列打包
  ↓
BiLSTM (2层, hidden=128, bidirectional)      # 输出维度: 128×2 = 256
  ↓
Pad Packed Sequence                          # 解包
  ↓
LayerNorm (256)                              # 层归一化，稳定特征分布
  ↓
Additive Attention (Bahdanau, dim=32)        # 自适应加权各帧重要性
  ↓                                          # 输出: (batch, 256)
Dropout (p=0.35)
  ↓
Linear (256 → 100)                           # 100 类分类
  ↓
输出: (batch, 100)                           # Logits
```

### 核心组件详解

| 组件 | 说明 |
|------|------|
| **BiLSTM** | 2 层双向 LSTM，hidden_size=128，双向拼接后输出维度 256。使用 `pack_padded_sequence` 处理变长序列。层间 Dropout=0.35。 |
| **LayerNorm** | 对 LSTM 各时间步输出做层归一化，减少 Attention 前的数值尺度差异。 |
| **Additive Attention** | Bahdanau 注意力机制：Linear(256→32) → tanh → Linear(32→1) → Softmax → 加权求和。支持 padding mask。 |
| **分类头** | Dropout(0.35) → Linear(256→100)。 |

模型参数总量约 **1.1M**，轻量级且训练效率高。

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
| 早停 | patience=150 | 监控 val_acc，长时间无提升自动停止 |
| 最大轮数 | 600 | |
| 随机种子 | 42 | 默认种子 |

### 数据增强（训练时在线增强）

| 增强方式 | 参数 |
|---------|------|
| 随机旋转 | ±18° |
| 随机缩放 | [0.88, 1.12] |
| 随机平移 | ±0.10 |
| 高斯噪声 | σ=0.003 |
| 水平翻转 | 概率 25%，交换左右语义 |
| 时间扭曲 | 概率 16%，速率 [0.90, 1.10] |
| 随机丢帧 | 概率 10%，最大丢弃 8% |
| 时间掩码 | 概率 30%，最大遮蔽 15% |

### 运行训练

```bash
# 标准训练
uv run python src/model/train_lstm.py

# 过拟合诊断模式（关闭增强、正则、早停等）
uv run python src/model/train_lstm.py --overfit-debug
```

训练过程中自动保存：
- `best_model.pth`：验证集最佳模型（EMA 版本）
- `epoch_N.pth`：每 5 个 epoch 定期保存

### Checkpoint 参数平均

对多个 epoch 的 checkpoint 做参数平均，可进一步提升稳定性：

```bash
uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260
```

---

## 模型评估

### 单模型评估

```bash
uv run python src/model/evaluate_lstm.py --model-path src/checkpoints/best_model.pth --no-tta
```

### 多模型集成评估

系统支持 **4 模型集成推理**：使用不同随机种子（42/123/456/789）训练的 4 个模型，对各模型的 Softmax 概率取平均后再取 argmax，大幅提升预测稳定性和准确率。

```bash
uv run python src/model/ensemble_evaluate.py
```

集成模型路径在 `src/config.py` 的 `EVALUATION.ensemble_model_paths` 中配置。

---

## 推理使用

### 一键启动

```bash
uv run python main.py
```

运行后打开暖色调启动器界面，可选择两种推理模式：

### 实时推理（Real-time Inference）

- 调用摄像头（默认索引 0，640×480@30fps）
- **多线程并行关键点提取**：使用 `ParallelKeypointExtractor` 类，自动检测 CPU 核心数（`cpu_count - 1`），以流水线模式并行提取 MediaPipe 关键点
- **手部优先检测**策略优化速度
- 每 3 帧执行一次集成推理
- 界面左上角显示实时 FPS
- 底部温馨毛玻璃卡片显示识别结果与概率条
- 支持 **骨骼叠加开关**：一键显示/隐藏 135 关键点骨架网络

### 离线推理（Offline Inference）

- 通过文件对话框导入 `.mp4` 等格式视频文件
- **多线程并行关键点提取**：与实时推理共享 `ParallelKeypointExtractor` 类，全量读取视频帧后并行提取关键点
- 全量读取视频帧 → 并行关键点提取 → 预处理 → 集成预测
- 支持连续导入新视频重复测试

### UI 操作

| 操作 | 方式 |
|------|------|
| 退出 | 点击「退出」按钮 或 按 `q` 键 |
| 骨骼开关 | 点击「显示骨骼」/「隐藏骨骼」按钮 |
| 导入视频 | 点击「导入视频」按钮（离线模式） |

### UI 设计风格

界面采用温馨简约的暖色调设计：
- 米色背景 RGB(253,246,237)
- 珊瑚色按钮 RGB(255,170,150)
- 金色按钮 RGB(245,210,140)
- 深棕色文字 RGB(80,65,55)
- 柔和阴影与圆角（radius=22-28）

---

## 实验结果

### WLASL100 测试集准确率

| 配置 | 测试准确率 |
|------|-----------|
| 单模型 (best_model, Seed 42) | ~72% |
| **4 模型集成 (Seeds 42/123/456/789)** | **75.97%** |

### 关键实验记录

| 实验 | 变更 | 结果 |
|------|------|------|
| E04 | 添加 LayerNorm | 训练更稳定 |
| E05 | 时间掩码增强 + 多种子集成 | 75.97% (最佳) |
| E06 | 多头注意力 (num_heads=4) | 对小数据集有害，回退 |
| E07a | 加宽 hidden=192 | 过拟合加重，回退 |
| E08 | 二阶加速度特征 (ddx, ddy) | 实验失败，回退至 4 通道 |

---

## 配置中心

所有超参数集中管理于 `src/config.py`，按模块分组访问：

| 配置分组 | 访问方式 | 说明 |
|---------|---------|------|
| 路径 | `cfg.PATHS.*` | 项目路径、数据路径、模型路径 |
| 序列 | `cfg.SEQUENCE.*` | max_frames=90, num_landmarks=135, 通道定义 |
| 预处理 | `cfg.PREPROCESS.*` | 插值、平滑、归一化策略 |
| MediaPipe | `cfg.MEDIAPIPE.*` | 模型路径、检测阈值 |
| 模型 | `cfg.MODEL.*` | hidden_size=128, num_layers=2, dropout=0.35 |
| 增强 | `cfg.AUGMENTATION.*` | 旋转、缩放、翻转、时间扭曲等 |
| 训练 | `cfg.TRAINING.*` | 优化器、调度器、EMA、早停 |
| 评估 | `cfg.EVALUATION.*` | 集成模型路径、TTA 设置 |
| 推理 | `cfg.INFERENCE.*` | 摄像头参数、推理间隔 |
| UI | `cfg.UI.*` | 字体、按钮、颜色 |

---

## 关于数据集

默认采用 **WLASL (Word-Level American Sign Language)** 数据集。请前往 [WLASL 官方仓库](https://dxli94.github.io/WLASL/) 获取视频原文件。支持在 `src/config.py` 中一键切换 WLASL100 / WLASL300 / WLASL1000 / WLASL2000 配置。

---

## 质量门禁

```bash
# 编译检查
uv run python -m compileall src

# 单元测试
uv run python -m unittest discover -s src/test -p "test_*.py"

# Lint 检查
uv run ruff check src
```
