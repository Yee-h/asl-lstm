## Context

本项目旨在通过一个 Android 应用，将基于时序骨骼点的 LSTM 手语识别模型 (`seed456_temporal_mask/best_model.ptl`) 落地到移动端。核心需求是实现基于设计图的三大页面：首页、实时识别页和离线推理页，并在实时摄像头画面和离线视频画面上提取人体骨骼点，将其送入 LSTM 模型，并在界面上实时绘制骨骼点、显示置信度、延迟及最终识别结果。

## Goals / Non-Goals

**Goals:**
- 初始化标准 Java Android 项目，支持 Android 7.0 (API 24) 及以上。
- 精确还原 UI 设计图（`UI/首页.png`, `UI/实时.png`, `UI/离线.png`）。
- 集成 PyTorch Mobile 用于在 Android 上加载 `.ptl` 模型并执行推理。
- 解决视频流/摄像头流与骨骼点提取模块（推测为 MediaPipe 或类似方案）的集成。
- 实现骨骼数据到 LSTM 模型的预处理转换（需要收集一定时序长度的帧），并计算推理耗时（延迟）与置信度。

**Non-Goals:**
- 不包括重新训练或修改已有的模型结构。
- 不包括在非 Android 平台上实现（如 iOS 或 Web）。
- 不涉及复杂的账户系统或后台服务器通讯。

## Decisions

1. **框架语言**
   - **决策**：使用 Java，使用传统 XML 布局（或可选 ViewBinding/DataBinding）。
   - **理由**：需求明确指明使用 Java 开发。
2. **摄像头与视频处理方案**
   - **决策**：使用 CameraX 库进行相机预览和帧获取。对于离线视频，使用 `MediaMetadataRetriever` 或 ExoPlayer 的帧提取功能。
   - **理由**：CameraX 提供了生命周期感知的简单 API，并且与各种设备的兼容性较好。
3. **骨骼点提取方案**
   - **决策**：使用 Google MediaPipe Pose (或 Holistic) 解决方案提取实时的关键点坐标。
   - **理由**：由于输入模型是 LSTM，且带有 `temporal_mask` 字样，通常手语识别需要输入一段时序的人体/手部关键点。MediaPipe 在移动端提供了高效的实时关键点提取能力。
4. **模型推理引擎**
   - **决策**：使用 PyTorch Mobile Android (`org.pytorch:pytorch_android_lite`)。
   - **理由**：模型格式为 `.ptl`（PyTorch Lite），原生支持通过 PyTorch Mobile 在端侧直接加载。
5. **UI 实现**
   - **决策**：使用 ConstraintLayout 结合自定义 View（用于绘制骨骼连线与识别框），以及标准按钮与文本组件。
   - **理由**：ConstraintLayout 可以轻松实现类似于设计图中覆盖在深色背景上的悬浮面板、四角对齐边框等效果。

## Risks / Trade-offs

- **Risk: 骨骼点提取性能与延迟** -> Mitigation: 尽量降低摄像头分辨率（如设置为 640x480 或 1280x720），并使用 MediaPipe 的 Lite 模型，确保在低端机型上的帧率。
- **Risk: LSTM 模型的输入格式未知** -> Mitigation: 由于未提供代码中预处理的确切维度信息（如输入帧数、关键点维度数等），可能需要先使用 Dummy 数据测试模型接口，并由模型设计者提供预处理规范，或通过反编译/逆向分析模型输入格式。
- **Risk: UI 中的扫描框或动态效果** -> Mitigation: 优先实现静态的四个圆角边框，确保核心逻辑跑通后再进行 UI 动画的完善。