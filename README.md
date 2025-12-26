# CSL-LSTM: 中国手语识别项目 (Chinese Sign Language Recognition)

本项目旨在构建一个基于 LSTM (Long Short-Term Memory) 的中国手语识别系统。项目包含完整的数据预处理流水线、模型训练（开发中）以及实时推理（开发中）模块。

目前主要实现了基于 MediaPipe 的手部与姿态关键点特征提取，支持多进程加速与断点续传。

## 🚀 功能特性

- **特征提取**: 使用 Google MediaPipe Holistic 提取 **手部 (Left/Right Hand) 和 姿态 (Pose)** 关键点。
- **坐标归一化**: 采用改进的相对坐标系，以 **鼻尖** 为原点，并除以 **肩宽** 进行缩放，同时消除拍摄位置和距离的影响。
- **多进程加速**: 自动利用多核 CPU 并行处理视频数据（Windows 下使用 spawn 模式），大幅提升预处理速度。
- **健壮性**: 
    - **断点续传**: 自动跳过已处理的视频文件。
    - **中文路径检测**: 针对 Windows 下 OpenCV 读取中文路径的问题提供检测和提示。
- **数据集管理**: 支持 Train/Test/Dev 数据集划分，自动关联 CSV 标签文件。

## 📂 目录结构

```
csl-lstm/
├── src/                    # 源代码目录
│   ├── config.py           # 全局配置文件 (路径、维度参数、性能开关)
│   ├── main.py             # 项目主入口 (CLI)
│   ├── utils.py            # 通用工具函数
│   └── data_preprocessing/ # 数据预处理模块
│       ├── data_preprocessing.py # 预处理主流程 (任务调度、多进程)
│       ├── extract_keypoints.py  # 特征提取 (MediaPipe 集成)
│       └── normalize.py          # 归一化核心算法
│   └── test/               # 测试与验证脚本
│       ├── data_shape_test.py    # 验证生成的数据维度
│       └── gpu_cuda_check.py     # 检查 GPU/CUDA 环境
├── dataset/                # 数据集目录
│   ├── label/              # 标签文件 (train.csv, test.csv, dev.csv)
│   ├── video/              # 原始视频 (按 split/translator 分类)
│   ├── processed/          # [生成] 提取后的特征数据 (.npy)
│   └── processed_labels/   # [生成] 处理后的标签索引 (.json)
├── AGENTS.md               # Agent 开发规范与记忆
├── README.md               # 项目文档
├── pyproject.toml          # 项目依赖管理
└── requirements.txt        # 依赖列表
```

## 🛠️ 安装与环境

推荐使用 Python 3.10+ 环境。

1. **安装依赖**

```bash
pip install mediapipe opencv-python numpy tqdm
```

或者使用 `pyproject.toml`:
```bash
pip install .
```

## ⚙️ 配置

所有可配置项均位于 `src/config.py` 中，主要包括：

- **路径配置**: 数据集输入/输出路径。
- **MediaPipe 配置**: 检测置信度、模型复杂度。
- **性能配置**: `NUM_WORKERS` (进程数), `USE_MULTIPROCESSING` (多进程开关)。
- **数据维度**: 
    - **Pose (33点)** + **Left Hand (21点)** + **Right Hand (21点)**
    - 输出总维度: **225** (75个点 × 3维)

## 💻 使用指南

项目通过 `src/main.py` 提供统一的命令行接口。

### 1. 数据预处理

从原始视频中提取关键点特征并保存为 `.npy` 文件。

```bash
cd src
# 处理所有数据集 (Train/Test/Dev)
python main.py preprocess

# 仅处理训练集
python main.py preprocess --split train

# 禁用多进程 (用于调试)
python main.py preprocess --no-multiprocessing
```

> **注意**: 在 Windows 下，如果视频路径包含中文字符，OpenCV 可能会读取失败。程序会检测并提示，建议将视频数据放在全英文路径下。

### 2. 查看统计信息

查看数据集的样本数量和分布情况。

```bash
# 查看所有数据集统计
python main.py stats

# 查看训练集统计
python main.py stats --split train
```

### 3. 测试与验证

项目提供了简单的脚本来验证环境和数据。

```bash
# 检查 CUDA/GPU 是否可用 (需要安装 PyTorch)
python src/test/gpu_cuda_check.py

# 验证生成的 .npy 数据形状 (建议在预处理后运行)
# 作用: 确认输出维度是否为预期的 [frames, 225]
python src/test/data_shape_test.py
```

## 📊 数据流说明

### 离线训练流水线 (Offline Pipeline)
1. **输入**: `dataset/video/` 下的 MP4 视频。
2. **特征提取 (Extract)**: 
   - 遍历视频帧 -> MediaPipe Holistic 推理。
   - 获取 Pose (33), Left Hand (21), Right Hand (21) 关键点。
3. **归一化 (Normalize)**: 
   - **原点修正**: $P' = P - P_{nose}$ (以鼻尖为中心)。
   - **尺度缩放**: $P_{norm} = \frac{P'}{|P_{left\_shoulder} - P_{right\_shoulder}|}$ (消除距离影响)。
4. **输出**: `dataset/processed/` 下的 `.npy` 文件。
   - Shape: `[Frames, 225]`
   - 包含: Pose(0-98) + LH(99-161) + RH(162-224)。

### 在线推理流水线 (Online Pipeline) - *Planned*
- 实时读取摄像头流。
- 复用 `src/data_preprocessing/normalize.py` 中的相同归一化逻辑。
- 维护滑动窗口 (Sliding Window) 输入模型进行预测。

## 📝 开发说明

- **模块化设计**: 
    - `normalize.py`: 纯数学计算，无外部依赖 (除了 numpy)。
    - `extract_keypoints.py`: 负责图像处理和模型推理。
    - `config.py`: 统一管理所有参数。
- **贡献代码**: 请遵循 `AGENTS.md` 中的开发规范。

