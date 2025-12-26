# CSL-LSTM: 中国手语识别项目 (Chinese Sign Language Recognition)

本项目旨在构建一个基于 LSTM (Long Short-Term Memory) 的中国手语识别系统。项目包含完整的数据预处理流水线、模型训练（开发中）以及实时推理（开发中）模块。

目前主要实现了基于 MediaPipe 的手部与姿态关键点特征提取，支持多进程加速与断点续传。

## 🚀 功能特性

- **特征提取**: 使用 Google MediaPipe Holistic 提取手部 (Left/Right Hand) 和姿态 (Pose) 关键点。
- **坐标归一化**: 采用以肩膀中心为原点的相对坐标系，消除拍摄距离和位置对识别的影响。
- **多进程加速**: 自动利用多核 CPU 并行处理视频数据，大幅提升预处理速度。
- **断点续传**: 自动跳过已处理的视频文件，防止意外中断后重头开始。
- **数据集管理**: 支持 Train/Test/Dev 数据集划分，自动关联 CSV 标签文件。

## 📂 目录结构

```
csl-lstm/
├── config.py               # 全局配置文件 (路径、参数、开关)
├── data_preprocessing.py   # 数据预处理核心逻辑
├── main.py                 # 项目主入口 (CLI)
├── utils.py                # 通用工具函数 (MediaPipe封装、特征提取)
├── pyproject.toml          # 项目依赖管理
├── README.md               # 项目文档
└── dataset/                # 数据集目录
    ├── label/              # 标签文件 (train.csv, test.csv, dev.csv)
    ├── video/              # 原始视频 (按 split/translator 分类)
    ├── processed/          # [生成] 提取后的特征数据 (.npy)
    └── processed_labels/   # [生成] 处理后的标签索引 (.json)
```

## 🛠️ 安装与环境

推荐使用 Python 3.10+ 环境。

1. **安装依赖**

```bash
pip install mediapipe opencv-python numpy tqdm
```

或者使用 `pyproject.toml` (如果使用 poetry 或 pdm):
```bash
pip install .
```

## ⚙️ 配置

所有可配置项均位于 `config.py` 中，主要包括：

- **路径配置**: 数据集输入/输出路径。
- **MediaPipe 配置**: 检测置信度、模型复杂度。
- **性能配置**: `NUM_WORKERS` (进程数), `USE_MULTIPROCESSING` (多进程开关)。
- **数据维度**: 定义关键点数量和特征维度。

## 💻 使用指南

项目通过 `main.py` 提供统一的命令行接口。

### 1. 数据预处理

从原始视频中提取关键点特征并保存为 `.npy` 文件。

```bash
# 处理所有数据集 (Train/Test/Dev)
python main.py preprocess

# 仅处理训练集
python main.py preprocess --split train

# 禁用多进程 (用于调试)
python main.py preprocess --no-multiprocessing
```

### 2. 查看统计信息

查看数据集的样本数量和分布情况。

```bash
# 查看所有数据集统计
python main.py stats

# 查看训练集统计
python main.py stats --split train
```

## 📊 数据流说明

### 离线训练流水线 (Offline Pipeline)
1. **输入**: `dataset/video/` 下的 MP4 视频。
2. **预处理**: 
   - 遍历视频帧 -> MediaPipe 推理 -> 获取关键点。
   - **归一化**: $P_{new} = P_{raw} - P_{shoulder\_center}$ (消除位移影响)。
   - 拼接左右手特征 (及姿态) -> Flatten。
3. **输出**: `dataset/processed/` 下的 `.npy` 文件 (Shape: `[Frames, Features]`)。

### 在线推理流水线 (Online Pipeline) - *Planned*
- 实时读取摄像头流。
- 使用相同的归一化逻辑处理当前帧。
- 维护滑动窗口 (Sliding Window) 输入模型进行预测。

## 📝 贡献与开发

- **utils.py**: 存放核心算法和工具函数。
- **config.py**: 存放所有硬编码参数，禁止在业务代码中写死参数。

