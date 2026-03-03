# ASL-LSTM 架构文档

## 1. 架构目标

通过分层解耦实现高内聚、低耦合，减少重复实现，提升可测试性与可维护性。

## 2. 目录结构

```text
asl-lstm/
├── main.py                        # 系统入口（推理模式选择界面）
├── pyproject.toml                 # UV 依赖配置（Python 3.10, PyTorch cu121）
├── src/
│   ├── config.py                  # 全局配置中心（分模块 dataclass）
│   ├── core/                      # 基础设施层
│   │   ├── paths.py               #   统一路径构建
│   │   ├── labels.py              #   标签映射加载与契约校验
│   │   └── hdf5_schema.py         #   HDF5 必填字段常量与校验
│   ├── data_process/              # 数据处理应用层
│   │   ├── preprocess_wlasl.py    #   核心预处理（视频 → HDF5 特征）
│   │   ├── count_dataset_samples.py #  数据集统计
│   │   ├── transfer_hdf5_data.py  #   HDF5 数据迁移
│   │   ├── verify_consistency.py  #   数据一致性验证
│   │   ├── compare_hdf5_detail.py #   HDF5 差异对比
│   │   └── read_data_struct.py    #   HDF5 结构查看
│   ├── model/                     # 模型应用层
│   │   ├── model_lstm.py          #   网络定义（BiLSTMAttention / BiLSTM）
│   │   ├── train_lstm.py          #   训练主循环
│   │   ├── evaluate_lstm.py       #   单模型评估
│   │   ├── ensemble_evaluate.py   #   多模型集成评估
│   │   ├── dataloader.py          #   数据加载（增强 + Z-Score）
│   │   ├── realtime_inference.py  #   实时推理（摄像头 + GUI）
│   │   ├── offline_inference.py   #   离线推理（视频文件 + GUI）
│   │   ├── average_checkpoints.py #   Checkpoint 参数平均
│   │   ├── checkpoint_avg_experiment.py # Checkpoint 平均实验
│   │   ├── checkpoint_utils.py    #   Checkpoint 工具函数
│   │   ├── training_utils.py      #   训练辅助（EMA、早停、Warmup）
│   │   ├── validate_lstm.py       #   验证循环
│   │   └── tta.py                 #   测试时增强（TTA）
│   ├── test/                      # 自动化测试（21 个测试文件）
│   └── mediapipe_models/          # MediaPipe 模型文件
├── dataset/
│   ├── raw/                       # 原始 WLASL 视频与元数据
│   └── processed/                 # 预处理后 HDF5 + 统计量
├── src/checkpoints/               # 模型权重
├── logs/                          # 训练日志与评估报告
└── docs/                          # 项目文档
```

## 3. 四层依赖图

```text
┌─────────────────────────────────────────────────────┐
│  接口层 (CLI 入口)                                    │
│  main.py, train_lstm.py, evaluate_lstm.py, ...       │
├─────────────────────────────────────────────────────┤
│  应用层 (业务逻辑)                                    │
│  src/model/*  ←→  src/data_process/*                 │
│  训练/评估/推理       预处理/迁移/验证                  │
├─────────────────────────────────────────────────────┤
│  领域层 (核心算法)                                    │
│  BiLSTMAttention, Attention, 预处理流水线              │
│  数据增强, EMA, 早停, 学习率调度                       │
├─────────────────────────────────────────────────────┤
│  基础设施层 (公共契约)                                 │
│  src/core/paths.py     统一路径构建                    │
│  src/core/labels.py    标签映射 (id↔label 互逆契约)    │
│  src/core/hdf5_schema.py  HDF5 字段常量与校验          │
│  src/config.py         全局配置中心                    │
└─────────────────────────────────────────────────────┘
```

**数据流向**：

```text
原始视频 → preprocess_wlasl.py → HDF5 特征文件
                                      ↓
                              dataloader.py (增强 + Z-Score)
                                      ↓
                              train_lstm.py → best_model.pth
                                      ↓
                        realtime_inference.py / offline_inference.py
                              (集成推理 + GUI 渲染)
```

## 4. 禁止依赖规则

1. `src/core/*` **不能**依赖 `src/model/*` 与 `src/data_process/*`。
2. `src/model/*` 与 `src/data_process/*` 可以依赖 `src/core/*`，但**不允许**互相复制公共逻辑。
3. 标签映射解析、路径构建、HDF5 字段校验**必须且只能**在 `src/core/*` 维护。
4. 新增脚本不得在应用层重复定义标签或 HDF5 契约校验函数。
5. 所有配置参数**必须**通过 `src/config.py` 的分组 dataclass 访问（如 `cfg.TRAINING.batch_size`），禁止硬编码魔术数字。

## 5. 核心模块职责

### 5.1 基础设施层 (`src/core/`)

| 模块 | 职责 |
|------|------|
| `paths.py` | 统一路径构建函数，消除各脚本中的路径拼接散落 |
| `labels.py` | 标签映射加载与契约校验。强制 `id_to_label` / `label_to_id` 互逆，拒绝旧格式 |
| `hdf5_schema.py` | HDF5 必填字段常量（`data`, `length`, `label`, `video_name`, `width`, `height`）与校验函数 |

### 5.2 数据处理层 (`src/data_process/`)

| 模块 | 职责 |
|------|------|
| `preprocess_wlasl.py` | 核心预处理管道：视频 → MediaPipe 关键点 → 插值/平滑/归一化/重采样/速度特征 → HDF5 |
| `count_dataset_samples.py` | 统计各 split 的样本数与类别分布 |
| `transfer_hdf5_data.py` | HDF5 数据格式迁移工具 |
| `verify_consistency.py` | 跨 split 数据一致性验证 |
| `compare_hdf5_detail.py` | 两个 HDF5 文件的逐字段差异对比 |
| `read_data_struct.py` | HDF5 内部结构可视化查看器 |

### 5.3 模型层 (`src/model/`)

| 模块 | 职责 |
|------|------|
| `model_lstm.py` | 网络定义：`BiLSTMAttention`（主模型）、`BiLSTM`（对比基线）、`Attention`（单头加性）、`MultiHeadAttention`（多头，实验用） |
| `train_lstm.py` | 训练主循环：梯度累积、EMA、早停、学习率调度、定期保存 |
| `validate_lstm.py` | 验证循环：计算 val_loss 与 val_acc |
| `evaluate_lstm.py` | 测试集评估：加载 checkpoint，输出分类报告与混淆矩阵 |
| `ensemble_evaluate.py` | 4 模型集成评估：Softmax 概率平均 |
| `dataloader.py` | `ASLDataset` 数据集类：HDF5 读取、在线增强、Z-Score 标准化 |
| `realtime_inference.py` | 实时推理：摄像头捕获 → 异步关键点提取 → 集成预测 → GUI 渲染（含启动器界面） |
| `offline_inference.py` | 离线推理：视频文件 → 全量提取 → 集成预测 → GUI 渲染 |
| `average_checkpoints.py` | 多 epoch checkpoint 参数平均 |
| `checkpoint_utils.py` | Checkpoint 加载/保存工具函数 |
| `training_utils.py` | EMA 管理器、早停、Warmup 调度器等训练辅助 |
| `tta.py` | 测试时增强（水平翻转 TTA） |

### 5.4 测试层 (`src/test/`)

21 个测试文件覆盖：配置结构、标签契约、模型输入契约、数据加载管道、增强策略、训练工具、推理冒烟测试、预处理冒烟测试等。

## 6. 配置架构

`src/config.py` 使用 `@dataclass(frozen=True)` 定义 10 个不可变配置分组：

```text
PathsConfig         → cfg.PATHS         路径配置
SequenceConfig      → cfg.SEQUENCE      序列与特征维度
PreprocessConfig    → cfg.PREPROCESS    预处理参数
MediapipeConfig     → cfg.MEDIAPIPE     MediaPipe 模型与阈值
ModelConfig         → cfg.MODEL         网络超参数
AugmentationConfig  → cfg.AUGMENTATION  数据增强参数
TrainingConfig      → cfg.TRAINING      训练策略
EvaluationConfig    → cfg.EVALUATION    评估与集成配置
InferenceConfig     → cfg.INFERENCE     推理参数
UIConfig            → cfg.UI            界面配置
```

所有配置通过 `import src.config as cfg` 后以 `cfg.<分组>.<字段>` 形式访问，禁止旧版全局常量别名。

## 7. 集成推理架构

```text
4 个独立训练的模型（Seeds: 42/123/456/789）
         ↓ 各自输出 logits
    Softmax 概率化
         ↓ 4 组概率向量
    算术平均
         ↓
    argmax → 最终预测类别
```

实时推理额外优化：
- **手部优先检测**：先检测手部，有手才触发全身关键点提取
- **每 3 帧推理一次**：平衡精度与实时性
- **异步队列**：关键点提取与模型推理解耦
