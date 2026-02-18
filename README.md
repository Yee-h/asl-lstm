# ASL-LSTM: 美国手语识别系统 (American Sign Language Recognition)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目实现了一个基于 **BiLSTM + Attention** 的高性能手语识别系统，采用 **双管道架构**（离线训练 + 在线推理），支持从视频数据集训练到实时摄像头交互的完整流程。项目经过多轮优化，在训练稳定性和推理鲁棒性上表现优异。

---

## 🚀 核心特性

### 1. 双管道架构 (Dual Pipeline)
- **Pipeline A (离线训练)**: 高效处理 WLASL 视频数据，生成标准化的 HDF5 特征文件，支持断点续传与多进程加速。
- **Pipeline B (在线推理)**: 实时摄像头流处理，严格对齐训练时的预处理逻辑，支持关键点有效性检查与间隔推理，满足实时响应需求。

### 2. 深度优化的数据处理
- **关键点提取**: 使用 MediaPipe 提取 **135个关键点** (Body25 + Hand42 + Face68)。
- **归一化策略**:
  - **坐标原点**: 动态计算肩中点 (Mid-Shoulder)，消除身体位移干扰。
  - **尺度统一**: 基于全序列肩宽中位数 (Global Video-level Scale)，消除拍摄距离影响。
- **特征增强**: 融合坐标 $(x, y)$ 与一阶差分速度 $(dx, dy)$，有效捕捉动作动态。
- **时序对齐**: 采用均匀重采样 (Uniform Resampling) 替代简单的截断/填充，完整保留动作语义。

### 3. 鲁棒的推理引擎
- **输入质量门控**: 未检测到手部关键点或关键点数量不足时暂停预测。
- **推理节流**: 支持按帧间隔执行模型推理，降低实时负载。
- **统一预处理**: 推理端复用训练端的几何归一化与标准化流程。

### 4. 工程化最佳实践
- **UV 依赖管理**: 极速、确定性的 Python 包管理。
- **配置集中化**: 所有超参数通过 `src/config.py` 统一管理。
- **任务追踪**: 通过 `docs/feature_list.json` 与 `docs/progress.md` 追踪开发进度。

---

## 📂 项目结构

```text
asl-lstm/
├── docs/                          # 文档中心
│   ├── Product-Spec.md
│   ├── Product-Spec-CHANGELOG.md
│   ├── feature_list.json
│   ├── progress.md
│   ├── API.md
│   └── Architecture.md
├── src/                           # 源代码核心
│   ├── config.py                  # 全局配置中心 (支持 WLASL100/300/2000 切换)
│   ├── core/                      # 公共基础模块（路径/标签/HDF5 契约）
│   ├── data_process/              # 数据处理管道
│   │   ├── preprocess_wlasl.py    # 核心预处理脚本 (生成 .hdf5)
│   │   └── ...                    # 数据分析与迁移工具
│   ├── model/                     # 模型与推理
│   │   ├── model_lstm.py          # BiLSTM + Attention 模型定义
│   │   ├── train_lstm.py          # 训练主程序
│   │   ├── realtime_inference.py  # 实时推理主程序
│   │   └── ...                    # DataLoader 与 评估脚本
│   └── test/                      # 测试与验证脚本
├── dataset/                       # 数据存放区
│   ├── raw/                       # 原始 WLASL 数据 (需自行下载)
│   └── processed/                 # 处理后的 HDF5 特征数据
├── init.sh                        # 环境初始化脚本
├── pyproject.toml                 # UV 项目配置
├── uv.lock                        # 依赖锁定文件
└── README.md                      # 本文档
```

---

## 🛠️ 快速开始

### 环境要求
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 (必须，MediaPipe 限制)
- **CUDA**: 推荐 12.1+ (用于 GPU 加速)

### 安装步骤

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd asl-lstm
    ```

2.  **初始化环境**
    运行初始化脚本，检查 Python 版本并创建必要目录：
    ```bash
    # Linux/macOS
    bash init.sh

    # Windows (Git Bash)
    sh init.sh
    ```

3.  **安装依赖**
    ```bash
    uv sync
    ```

4.  **验证环境**
    ```bash
    uv run python src/test/gpu_cuda_check.py
    ```

---

## 📖 使用指南

### 1. 数据准备
下载 [WLASL 数据集](https://dxli94.github.io/WLASL/)，并将视频文件放入 `dataset/raw` 目录。

在 `src/config.py` 中配置数据集规模：
```python
# src/config.py
_DATASET_SCALE = 100  # 可选: 100, 300, 1000, 2000
```

### 2. 数据预处理 (Pipeline A)
提取特征并生成 HDF5 文件：
```bash
uv run python src/data_process/preprocess_wlasl.py
```
*输出：`dataset/processed/WLASL100/WLASL100_135-Train.hdf5` 等文件。*

### 3. 模型训练
启动训练流程：
```bash
uv run python src/model/train_lstm.py
```
*模型权重将保存在 `src/checkpoints/` 目录。*
*默认训练已启用 EMA（指数滑动平均）用于验证与 best 模型保存，以提升泛化稳定性。*
*当前回归基线结构为 BiLSTM + Attention + Dropout + Linear 分类头（用于提升小样本稳定性）。*

过拟合能力诊断（固定 `seed=42` 下排查模型是否可拟合训练集）：
```bash
uv run python src/model/train_lstm.py --overfit-debug
```
*该模式会临时关闭训练增强、类别重采样、Dropout、标签平滑与权重衰减。*
*同时会关闭验证集学习率调度与早停，避免诊断过程被验证集波动打断。*

### 3.1 模型评估与 checkpoint 平均
评估默认 best 模型：
```bash
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/best_model.pth
```

对指定 epoch 区间进行 checkpoint 平均（示例：240~260）：
```bash
uv run python src/model/average_checkpoints.py --start-epoch 240 --end-epoch 260
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/averaged_240_260.pth
```
*平均模型可在训练后期降低单点 checkpoint 波动，提升测试稳定性。*
*默认评估口径使用无 TTA 指标，便于跨轮次可比。*

### 3.2 长训交接（当前推荐）
为恢复到历史测试精度区间（约 63%~67%），建议按以下顺序执行：

```bash
# 1) 训练前清空历史 checkpoint（你当前习惯）
# 2) 执行长训（默认使用 src/config.py 当前参数）
uv run python src/model/train_lstm.py

# 3) 评估单点 best_model
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/best_model.pth

# 4) 对后期 checkpoint 做区间平均（示例区间，可按实际最佳段调整）
uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/averaged_160_260.pth
```

建议回传以下信息，便于后续 AI 继续优化：
- 早停轮次（若触发）、最佳验证准确率及对应 epoch。
- `best_model.pth` 的测试准确率与评估报告路径。
- `averaged_*.pth` 的测试准确率与评估报告路径。
- 关键训练日志片段（峰值前后约 20~40 轮）。

### 4. 实时推理 (Pipeline B)
启动摄像头进行实时识别：
```bash
uv run python src/model/realtime_inference.py
```
*按 `q` 退出推理窗口。*

---

## 📊 修复状态

当前项目按 `fix.md` 执行修复流程，任务状态以 `docs/feature_list.json` 为准，执行日志以 `docs/progress.md` 为准。

如需查看接口与架构细节，请参考：
- `docs/API.md`
- `docs/Architecture.md`

---
