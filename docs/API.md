# ASL-LSTM API 文档

## 1. 概述

本文档描述 ASL-LSTM 项目主要脚本接口、输入输出与数据契约。后续任务将持续补充参数细节和错误约束。

## 2. 脚本入口

### 2.1 预处理入口
- 脚本: `src/data_process/preprocess_wlasl.py`
- 作用: 从视频提取关键点并生成 HDF5 数据。
- 常用参数: `--json`、`--video-dir`、`--output-dir`、`--prefix`、`--limit`。

### 2.2 训练入口
- 脚本: `src/model/train_lstm.py`
- 作用: 基于 HDF5 数据训练 LSTM 模型。

### 2.3 评估入口
- 脚本: `src/model/evaluate_lstm.py`
- 作用: 加载最佳模型并输出测试集评估指标。

### 2.4 实时推理入口
- 脚本: `src/model/realtime_inference.py`
- 作用: 摄像头/视频流实时识别手语类别。
- 常用参数: `--camera`、`--video`。

### 2.5 数据检查与统计入口
- `src/data_process/count_dataset_samples.py`
  - 参数: `--dataset-scale`、`--split`、`--data-root`。
- `src/test/processed_data_test.py`
  - 参数: `--dataset-scale`、`--split`、`--data-root`、`--sample-limit`。
- `src/test/check_json.py`
  - 参数: `--dataset-scale`、`--split`、`--data-root`。
- `src/test/check_coords.py`
  - 参数: `--dataset-scale`、`--split`、`--data-root`。

## 3. 数据契约（初版）

### 3.1 HDF5 样本结构
- 样本组键: 数字字符串（如 `"0"`）。
- 关键字段: `data`、`length`、`label`、`video_name`、`width`、`height`。
- 可选字段: `mask`、`quality`。

字段定义：
- `data`: `float32`，形状 `(T, C, 135)`，默认 `C=4`（`x,y,dx,dy`）。
- `length`: `int64`，有效帧长度（未 padding 前）。
- `label`: `string`，gloss 标签。
- `video_name`: `string`，原始视频路径或规范化路径。
- `width` / `height`: `int64`，视频宽高。
- `mask`(可选): `uint8`，关键点有效掩码。
- `quality`(可选): `float32`，样本质量分数。

### 3.2 标签映射文件
- 路径: `cfg.PATHS.label_map_path`。
- 文件名示例: `wlasl_100_maplabels.json`。

标签映射采用唯一契约：

```json
{
  "id_to_label": {
    "0": "book",
    "1": "drink"
  },
  "label_to_id": {
    "book": 0,
    "drink": 1
  }
}
```

- `id_to_label` 键必须是数字字符串，值必须是标签字符串。
- `label_to_id` 键必须是标签字符串，值必须是整数 ID。
- 两个映射必须完全互逆，否则视为非法契约。
- 旧格式（如 `id_to_label: {"book": 0}`）会被解析器拒绝并抛出明确异常。
- 统一解析函数: `src/model/dataloader.py` 中的 `load_label_to_id_map`。

## 4. 统一质量门禁

每个任务结束后按顺序执行：

```bash
uv run python -m compileall src
uv run python -m unittest discover -s src/test -p "test_*.py"
```

从 T06 开始额外执行：

```bash
uv run ruff check src
```
