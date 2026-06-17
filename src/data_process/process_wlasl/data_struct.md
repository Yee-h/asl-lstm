# WLASL 数据集结构说明

## 1. 文件基本信息
- **文件格式**: HDF5 (`.hdf5`)
- **文件路径**: `pose_action_dataset/ISLR/WLASL/WLASL100/WLASL100_135-Train.hdf5` (以 Train 集为例)
- **总体结构**: HDF5 文件的根目录下包含大量 Group（组），每个组的 Key 为视频的唯一标识符（Video ID，字符串格式，如 `"06328"`）。

## 2. 单个样本结构 (Sample Structure)
每个 Video ID 对应的 Group 包含以下 Dataset：

| 字段名 (Field) | 类型 (Dtype) | 形状 (Shape) | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | `float64` | `(T, 2, 135)` | **骨骼关键点数据**。<br>- `T`: 帧数 (可变)。<br>- `2`: 最大人数 (通常只取 index 0 为主人物)。<br>- `135`: 特征维度 (通常是 Body/Hand/Face 关键点的坐标展平)。 |
| **`label`** | `object` (String) | `()` (Scalar) | **手语单词 (Gloss)**。例如 `"book"`, `"hello"`。 |
| **`video_name`** | `object` (String) | `()` (Scalar) | 原始视频相对路径。例如 `"rgb/WLASL100/train/14675.mp4"`。 |
| **`height`** | `int64` | `()` (Scalar) | 原始视频高度。 |
| **`width`** | `int64` | `()` (Scalar) | 原始视频宽度。 |

## 3. 数据分布统计 (WLASL100 Train)
对 `WLASL100_135-Train.hdf5` 的帧数 (`T`) 进行了统计分析，结果如下：

| 统计项 | 数值 | 建议 |
| :--- | :--- | :--- |
| **样本总数** | 1442 | - |
| **最小帧数** | 13 | - |
| **最大帧数** | 203 | - |
| **平均帧数** | 62.38 | - |
| **中位数** | 60.0 | - |
| **95% 分位点** | **106** | **推荐 MAX_FRAMES 设为 106 或 110** |
| **99% 分位点** | 131 | - |

## 4. 处理建议
1.  **Padding (填充)**: 由于 `T` 是可变的，训练时需填充或截断至固定长度 `MAX_FRAMES` (推荐 106)。
2.  **Flatten (展平)**: 模型输入通常需要将 `(T, 2, 135)` 处理为 `(T, 270)` 或仅取单人 `(T, 135)`。建议保留两人数据 `(T, 270)` 以防关键点误检漂移到第二人索引。
3.  **Masking**: 使用 LSTM/GRU 时，应记录样本的真实长度 `T`，使用 `pack_padded_sequence` 忽略填充部分的计算。
