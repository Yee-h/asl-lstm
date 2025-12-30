# ASL-LSTM: 美国手语识别项目 (American Sign Language Recognition)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev/)

本项目实现了基于 **BiLSTM** 的手语识别系统，采用 **双管道架构**（离线训练 + 在线推理），支持从视频数据集训练模型到实时摄像头识别的完整流程。

## 核心特性

### 架构设计

- **管道A（离线训练流水线）**：
  - 批量处理视频数据集，生成高效 **HDF5** 格式特征文件
  - MediaPipe Holistic 提取 **135个关键点** (Body25 + Hand42 + Face68)
  - 改进的 **3D** 坐标归一化算法（鼻尖原点 + 肩宽缩放）
  - 多进程加速 + 断点续传
  - BiLSTM 模型训练

- **管道B（在线推理流水线）**：
  - 实时摄像头输入
  - 严格对齐预处理逻辑（确保推理与训练数据一致）
  - 滑动窗口缓冲（30帧）
  - 防抖动 + 冷却机制
  - 毫秒级响应

### 技术亮点

- ✅ **标准数据格式**：使用 HDF5 存储特征，结构清晰，读取高效
- ✅ **全维度特征**：包含 Body, Hands, Face 共 135 个关键点，保留 Z 轴深度信息
- ✅ **归一化一致性**：训练与推理使用完全相同的归一化逻辑
- ✅ **距离无关**：通过肩宽归一化消除用户与摄像头距离影响
- ✅ **配置集中化**：所有参数统一在 `config.py` 管理
- ✅ **GPU 加速**：支持 CUDA 12.1 训练与推理

##  项目结构

```
asl-lstm/
├── src/                           # 源代码
│   ├── config.py                  # 全局配置（支持 WLASL100/300/2000 切换）
│   ├── data_process/              # 数据处理
│   │   ├── preprocess_wlasl.py    # 预处理主流程 (生成 WLASL 格式 .hdf5)
│   │   ├── transfer_hdf5_data.py  # 数据迁移工具 (Val/Test -> Train)
│   │   ├── analyze_frames.py      # 帧数分析
│   │   ├── count_dataset_samples.py # 样本统计
│   │   └── read_data_struct.py    # HDF5 结构查看
│   ├── model/                     # 模型训练与评估
│   │   ├── model_lstm.py          # BiLSTM 模型定义
│   │   ├── dataloader.py          # DataLoader (支持 HDF5 读取)
│   │   ├── train_lstm.py          # 训练主流程
│   │   ├── validate_lstm.py       # 验证函数
│   │   └── evaluate.py            # 测试集评估
│   ├── checkpoints/               # [生成] 模型权重
│   │   ├── best_model.pth
│   │   └── vocab.json
│   └── test/                      # 测试脚本
│       ├── gpu_cuda_check.py      # GPU 环境检测
│       └── data_shape_test.py     # 数据形状验证
├── dataset/                       # 数据集
│   ├── raw/                       # 原始数据 (WLASL videos & json)
│   ├── processed/                 # 处理后数据
│   │   ├── WLASL100/              # WLASL-100 规模数据
│   │   │   ├── WLASL100_135-Train.hdf5
│   │   │   ├── WLASL100_135-Val.hdf5
│   │   │   ├── WLASL100_135-Test.hdf5
│   │   │   └── wlasl_100_maplabels.json
│   │   ├── WLASL300/              # WLASL-300 规模数据
│   │   └── WLASL2000/             # WLASL-2000 规模数据
├── logs/                          # 日志文件
├── pyproject.toml                 # UV 项目配置
├── uv.lock                        # UV 锁文件
├── AGENTS.md                      # 开发备忘
└── README.md                      # 本文档
```

##  环境配置

### 系统要求

- **Python**: 3.10（必须，mediapipe==0.10.9 不支持 3.11+）
- **GPU**: NVIDIA RTX 系列（推荐，支持 CUDA 12.1）
- **OS**: Windows 10/11, Linux, macOS

### 安装依赖

本项目使用 [UV](https://github.com/astral-sh/uv) 进行依赖管理：

```bash
# 1. 克隆项目
git clone <repository_url>
cd asl-lstm

# 2. 同步依赖
uv sync

# 3. 验证 GPU 可用性
uv run python src/test/gpu_cuda_check.py
```

##  使用指南

### 步骤 1: 数据下载与配置

#### 数据集下载
本项目使用WLASL (World Level American Sign Language) Video数据集。
官方下载地址：[WLASL](https://dxli94.github.io/WLASL/)
Kaggle下载地址：[WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=videos)

#### 数据集配置
在 `src/config.py` 中修改 `DATASET_SCALE` 来选择使用的数据集规模：
```python
# 可选: 100, 300, 2000
DATASET_SCALE = 100 
```

### 步骤 2: 数据预处理（管道A）

从 `dataset/raw` 的原始视频中提取关键点特征并保存为 `.hdf5` 文件。

```bash
# 使用 UV 运行
uv run python src/data_process/preprocess_wlasl.py
```

**预期输出**：
- `dataset/processed/WLASL{SCALE}/WLASL{SCALE}_135-{Split}.hdf5`: 特征数据
- `dataset/processed/WLASL{SCALE}/wlasl_{SCALE}_maplabels.json`: 词汇表映射

### 步骤 3: 模型训练（管道A）

使用提取的特征训练 BiLSTM 模型。程序会自动根据 `DATASET_SCALE` 寻找对应的数据文件。

```bash
# 训练模型
uv run python src/model/train_lstm.py
```

**模型输出**：
- `src/checkpoints/best_model.pth`: 最佳模型权重
- (词汇表直接读取 `dataset/processed/WLASL{SCALE}/wlasl_{SCALE}_maplabels.json`)

### 步骤 4: 实时推理（管道B）

使用摄像头进行实时手语识别。

```bash
uv run python src/model/realtime_inference.py
```

## 🔧 配置参数详解

所有配置集中在 `src/config.py`。


### 数据维度 (Updated)
```python
# 坐标维度 (x, y, z)
LANDMARK_DIM = 3

# 关键点映射
POSE (Body 25) + Left Hand (21) + Right Hand (21) + Face (68)
Total Keypoints = 135

# 总特征维度
TOTAL_FEATURE_DIM = 135 * 3 = 405
MAX_FRAMES = 110 (超长截断)
```

### 归一化逻辑
1. **原点**: 鼻尖 (Nose)
2. **尺度**: 左右肩宽 (Shoulder Width)
3. **映射**: 将 MediaPipe 的 dense output 映射到 OpenPose Body 25 和 simplified Face 68 格式。

## 数据集格式 (HDF5)

**文件**: `dataset/processed/WLASL100/WLASL100_135-Train.hdf5`

**层级结构**:
- `/{Video_ID}` (Group)
  - `data` (Dataset): 形状 `(Frames, 405)`。包含展平后的 normalized 3D 坐标。
    - 内容顺序: [Body(25), LHand(21), RHand(21), Face(68)]
  - Attributes:
    - `label`: Gloss (e.g., "book")
    - `action_id`: Label ID (e.g., 0)
    - `frame_count`: Number of frames


