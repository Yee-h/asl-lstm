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
- 默认策略: 启用 EMA（指数滑动平均）进行验证评估与 `best_model.pth` 保存。
- 模型结构（当前回归基线）: BiLSTM + Attention + Dropout + Linear 分类头。
- 常用参数:
  - `--overfit-debug`: 启用过拟合诊断模式（关闭训练增强、类别重采样、Dropout、标签平滑、权重衰减、验证集调度与早停）。

### 2.3 评估入口
- 脚本: `src/model/evaluate_lstm.py`
- 作用: 加载最佳模型并输出测试集评估指标。
- 常用参数:
  - `--model-path`: 指定待评估 checkpoint 路径。
  - `--no-tta`: 关闭水平翻转 TTA（当前默认评估口径为无 TTA，可确保跨轮次可比）。

### 2.3.1 Checkpoint 平均入口
- 脚本: `src/model/average_checkpoints.py`
- 作用: 对多个 epoch 的 checkpoint 做参数平均并导出新模型。
- 常用参数:
  - `--start-epoch` / `--end-epoch`: 参与平均的 epoch 区间。
  - `--step`: checkpoint 步长（默认 5）。
  - `--output`: 输出 checkpoint 路径（可选）。

### 2.3.2 长训恢复建议流程（交接）
- 目标: 在当前回归基线下，验证是否可恢复到历史测试准确率区间（约 63%~67%）。
- 推荐命令:

```bash
uv run python src/model/train_lstm.py
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/best_model.pth
uv run python src/model/average_checkpoints.py --start-epoch 160 --end-epoch 260
uv run python src/model/evaluate_lstm.py --no-tta --model-path src/checkpoints/averaged_160_260.pth
```

- 建议记录并回传:
  - 最佳 `val_acc` 与对应 epoch。
  - `best_model` 与 `averaged` 模型的测试集准确率及报告文件路径。
  - 峰值前后关键训练日志片段。

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
