# model-inference Specification

## Purpose
TBD - created by archiving change asl-android-app. Update Purpose after archive.
## Requirements
### Requirement: 模型加载与初始化
应用 MUST 在启动或进入识别页时，从 assets 目录中加载 PyTorch Lite 格式的预训练模型 `best_model.ptl`。

#### Scenario: 成功加载模型
- **WHEN** 应用启动并初始化模型模块
- **THEN** 模型对象被正确创建在内存中，不应崩溃。

### Requirement: 骨骼数据预处理
应用 MUST 实现从每一帧提取出的原始骨骼点坐标到模型所需的输入特征格式（通常是 Tensor 格式，考虑了时序特征）的转换。

#### Scenario: 累积时序特征
- **WHEN** 收集到最新的一帧骨骼关键点数据
- **THEN** 系统将其追加到时序缓冲区（Buffer）中，形成固定长度的 LSTM 输入序列。

### Requirement: 模型推理
应用 MUST 调用预加载的 PyTorch Mobile 模型进行前向推理，并计算出置信度和耗时。

#### Scenario: 模型推理执行
- **WHEN** 时序缓冲区满一帧或滑动窗口达到推理条件
- **THEN** 调用 `model.forward(tensor)` 并提取输出张量。

### Requirement: 后处理与结果展示
应用 MUST 解析模型输出，找到最大概率对应的类别，并在 UI 上实时展示置信度、延迟时间和对应的分类结果文本（如 "Hello"）。

#### Scenario: 更新 UI 信息
- **WHEN** 模型完成一次推理并返回分类和置信度，记录延迟
- **THEN** 界面的“置信度”、“延迟”和“识别结果”被刷新为新的值。

