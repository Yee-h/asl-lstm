# CSL-LSTM: 中国手语识别项目 (Chinese Sign Language Recognition)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev/)

本项目实现了基于 **BiLSTM** 的中国手语识别系统，采用 **双管道架构**（离线训练 + 在线推理），支持从视频数据集训练模型到实时摄像头识别的完整流程。

## 🎯 核心特性

### 架构设计

- **管道A（离线训练流水线）**：
  - 批量处理视频数据集
  - MediaPipe Holistic 提取手部与姿态关键点
  - 改进的坐标归一化算法（鼻尖原点 + 肩宽缩放）
  - 多进程加速 + 断点续传
  - BiLSTM 模型训练

- **管道B（在线推理流水线）**：
  - 实时摄像头输入
  - 滑动窗口缓冲（30帧）
  - 防抖动 + 冷却机制
  - 毫秒级响应

### 技术亮点

- ✅ **归一化一致性**：训练与推理使用完全相同的归一化逻辑
- ✅ **距离无关**：通过肩宽归一化消除用户与摄像头距离影响
- ✅ **配置集中化**：所有参数统一在 `config.py` 管理
- ✅ **GPU 加速**：支持 CUDA 12.1 训练与推理
- ✅ **健壮性**：中文路径检测、变长序列处理、错误日志

## 📂 项目结构

```
csl-lstm/
├── src/                           # 源代码
│   ├── config.py                  # 🔧 全局配置（路径/超参数/归一化参数）
│   ├── utils.py                   # 工具函数
│   ├── data_preprocessing/        # 数据预处理（管道A - 特征提取）
│   │   ├── extract_keypoints.py   # MediaPipe 特征提取
│   │   ├── normalize.py           # 归一化算法核心
│   │   └── data_preprocessing.py  # 预处理主流程
│   ├── model/                     # 模型训练与评估（管道A - 训练）
│   │   ├── model_lstm.py          # BiLSTM 模型定义
│   │   ├── padding.py             # 数据集与 Collate Function
│   │   ├── train_lstm.py          # 训练主流程
│   │   ├── validate_lstm.py       # 验证函数
│   │   ├── evaluate.py            # 测试集评估
│   │   └── realtime_inference.py  # 实时推理（管道B）
│   ├── checkpoints/               # [生成] 模型权重
│   │   ├── best_model.pth
│   │   └── vocab.json
│   └── test/                      # 测试脚本
│       ├── gpu_cuda_check.py      # GPU 环境检测
│       └── data_shape_test.py     # 数据形状验证
├── dataset/                       # 数据集
│   ├── label/                     # CSV 标签文件
│   ├── video/                     # 原始视频（按 split/translator 组织）
│   ├── processed/videos/          # [生成] 特征 .npy 文件
│   ├── processed/labels/          # [生成] 处理后的标签 JSON
├── logs/                          # 日志文件
├── pyproject.toml                 # UV 项目配置
├── uv.lock                        # UV 锁文件
├── AGENTS.md                      # 开发备忘
└── README.md                      # 本文档
```

## 🛠️ 环境配置

### 系统要求

- **Python**: 3.10（必须，mediapipe==0.10.9 不支持 3.11+）
- **GPU**: NVIDIA RTX 系列（推荐，支持 CUDA 12.1）
- **OS**: Windows 10/11, Linux, macOS

### 安装依赖

本项目使用 [UV](https://github.com/astral-sh/uv) 进行依赖管理：

```bash
# 1. 克隆项目
git clone <repository_url>
cd csl-lstm

# 2. 同步依赖（会自动安装 PyTorch CUDA 12.1 版本）
uv sync

# 3. 验证 GPU 可用性
uv run python src/test/gpu_cuda_check.py
```

**手动安装（如不使用 UV）：**
```bash
pip install mediapipe==0.10.9 opencv-python protobuf==3.20.3 tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```

## 📖 使用指南

### 步骤 1: 数据预处理（管道A - 特征提取）

从原始视频中提取关键点特征并保存为 `.npy` 文件。

```bash
# 使用 UV 运行
uv run python src/data_preprocessing/data_preprocessing.py

# 或激活虚拟环境后运行
python src/data_preprocessing/data_preprocessing.py
```

**配置选项**（在 `src/config.py` 中修改）：
- `NUM_WORKERS`: 并行进程数（默认：CPU核心数-2）
- `DATA_LIMIT`: 测试模式数量限制（`None` 或 `'all'` 表示全部）
- `ENABLE_RESUME`: 断点续传开关
- `MP_MODEL_COMPLEXITY`: MediaPipe 模型复杂度（0/1/2）

**预期输出**：
- `dataset/processed/videos/{split}/{translator}/*.npy`
- `dataset/processed/labels/{split}_labels.json`
- 错误日志（如有）: `logs/preprocessing_errors.log`

### 步骤 2: 模型训练（管道A - 训练）

使用提取的特征训练 BiLSTM 模型。

```bash
# 训练模型
uv run python src/model/train_lstm.py
```

**训练配置**（在 `src/config.py` 中修改）：
- `BATCH_SIZE`: 批次大小（默认：32）
- `LEARNING_RATE`: 初始学习率（默认：1e-4）
- `NUM_EPOCHS`: 训练轮数（默认：50）
- `HIDDEN_DIM`: LSTM 隐藏层维度（默认：256）
- `NUM_LAYERS`: LSTM 层数（默认：2）
- `DROPOUT`: Dropout 比例（默认：0.5）

**模型输出**：
- `src/checkpoints/best_model.pth`: 最佳模型权重
- `src/checkpoints/vocab.json`: 词汇表映射

**训练监控**：
- 实时打印 Train/Val Loss 和 Accuracy
- 自动保存验证集准确率最高的模型
- 学习率自动衰减（ReduceLROnPlateau）

### 步骤 3: 模型评估

在测试集上评估模型性能。

```bash
uv run python src/model/evaluate.py
```

**输出**：
- 测试集准确率
- 错误样本分析（前10个）

### 步骤 4: 实时推理（管道B）

使用摄像头进行实时手语识别。

```bash
uv run python src/model/realtime_inference.py
```

**交互控制**：
- `q`: 退出程序
- `r`: 重置当前句子
- `c`: 清空特征窗口

**实时配置**（在 `src/config.py` 中修改）：
- `SLIDING_WINDOW_SIZE`: 滑动窗口大小（默认：30帧）
- `PREDICTION_THRESHOLD`: 置信度阈值（默认：0.8）
- `DEBOUNCE_FRAMES`: 防抖帧数（默认：5）
- `COOLDOWN_FRAMES`: 冷却帧数（默认：15）
- `CAMERA_INDEX`: 摄像头索引（默认：0）

**识别流程**：
1. 摄像头采集帧 → 提取关键点 → 归一化
2. 添加到滑动窗口（30帧）
3. 窗口满后进行 LSTM 推理
4. 置信度 > 阈值 且 连续N帧一致 → 输出词汇
5. 进入冷却期，避免重复识别

## 🔧 配置参数详解

所有配置集中在 `src/config.py`，禁止在业务代码中硬编码参数。

### 路径配置
```python
PROJECT_ROOT = "项目根目录"
DATASET_PATH = "dataset/"
RAW_VIDEOS_PATH = "dataset/video/"
PROCESSED_DATA_PATH = "dataset/processed/videos/"
MODEL_SAVE_DIR = "src/checkpoints/"

```

### 归一化参数（核心）
```python
# 坐标归一化公式：
# x_new = (x - x_nose) / shoulder_width
# y_new = (y - y_nose) / shoulder_width

HAND_LANDMARKS_NUM = 21  # 单手关键点数
POSE_LANDMARKS_NUM = 33  # 姿态关键点数
LANDMARK_DIM = 3         # 每个点的维度 (x, y, z)
TOTAL_FEATURE_DIM = 225  # 左手(63) + 右手(63) + 姿态(99)
```

### 训练超参数
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
```

### 实时推理参数
```python
SLIDING_WINDOW_SIZE = 30       # 窗口长度
PREDICTION_THRESHOLD = 0.8     # 置信度阈值
DEBOUNCE_FRAMES = 5            # 防抖帧数
COOLDOWN_FRAMES = 15           # 冷却帧数
```

## 🧪 测试与验证

```bash
# GPU 环境检测
uv run python src/test/gpu_cuda_check.py

# 数据形状验证
uv run python src/test/data_shape_test.py
```

## 📊 数据集格式

### 输入格式

**视频文件**：
```
dataset/video/
├── train/
│   ├── A/
│   │   ├── train-00001.mp4
│   │   └── ...
│   ├── B/
│   └── ...
├── test/
└── dev/
```

**标签文件** (`dataset/label/{split}.csv`)：
```csv
Column1,Column2,Column3,Column4,Column5
Number,Translator,Chinese,Gloss,Note
train-00001,A,你好,hello,
train-00002,A,谢谢,thank,
...
```

### 输出格式

**特征文件** (`.npy`)：
```python
# Shape: (Frames, 225)
# 每帧包含：左手(63) + 右手(63) + 姿态(99)
features = np.load('dataset/processed/videos/train/A/train-00001.npy')
```

**标签文件** (`{split}_labels.json`)：
```json
[
  {
    "video_id": "train-00001",
    "npy_path": "dataset/processed/videos/train/A/train-00001.npy",
    "translator": "A",
    "chinese": "你好",
    "gloss": "hello",
    "note": ""
  }
]
```

## 🎯 模型架构

```
输入: (Batch, Seq_Len, 225)
   ↓
全连接层: 225 → 128 (降维 + ReLU + Dropout)
   ↓
BiLSTM: 128 → 256×2 (双向)
   ↓
拼接最后时刻的前向与后向隐藏状态: (Batch, 512)
   ↓
全连接层: 512 → Num_Classes
   ↓
Softmax → 预测概率
```

**关键设计**：
- 使用 `pack_padded_sequence` 处理变长序列
- 双向 LSTM 捕获前后文信息
- Dropout 防止过拟合
- Adam 优化器 + ReduceLROnPlateau 调度器

## ⚠️ 常见问题

### 1. 为什么限制 Python 3.10？

`mediapipe==0.10.9` 不支持 Python 3.11+，会导致依赖解析失败。

### 2. CUDA 版本不匹配怎么办？

项目默认使用 CUDA 12.1。如需其他版本：
```bash
# CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Windows 中文路径报错？

OpenCV 在 Windows 下不支持中文路径。解决方案：
- 将视频移动到纯英文路径
- 或使用 `cv2.imdecode` 配合 `np.fromfile` 读取

### 4. 训练时显存不足？

- 减小 `BATCH_SIZE`（如改为 16）
- 减小 `HIDDEN_DIM`（如改为 128）
- 使用梯度累积

### 5. 实时识别延迟高？

- 降低 `MP_MODEL_COMPLEXITY`（改为 0）
- 增大 `PREDICTION_THRESHOLD`（只在高置信度时输出）
- 使用更快的 GPU

## 🚀 性能优化建议

### 预处理加速
- 启用多进程：`USE_MULTIPROCESSING = True`
- 调整进程数：`NUM_WORKERS = 8`（根据 CPU 核心数调整）
- 使用 SSD 存储视频

### 训练加速
- 使用 GPU：确保 CUDA 可用
- 增大批次：`BATCH_SIZE = 64`（显存允许的情况下）
- 混合精度训练：`torch.cuda.amp`

### 推理加速
- 模型量化：使用 `torch.quantization`
- TensorRT 加速
- 降低摄像头分辨率

## 📝 开发规范

- 所有配置参数必须定义在 `config.py`
- 禁止在业务代码中硬编码路径或参数
- 其他模块通过 `import config as cfg` 导入
- 遵循 PEP 8 代码风格
- 关键函数添加类型注解和文档字符串

## 📄 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 🙏 致谢

- [MediaPipe](https://mediapipe.dev/) - Google 提供的关键点检测框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- CSL 数据集提供者

## 📮 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

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

