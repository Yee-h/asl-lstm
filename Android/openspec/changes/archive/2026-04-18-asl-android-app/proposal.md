## Why

开发一款基于Java的Android应用程序，用于将训练好的手语识别模型（LSTM）部署到移动端。这能够让用户在手机上直接进行实时的手语识别（通过摄像头）或离线的手语识别（通过导入视频），提供便捷的无障碍交流辅助工具。

## What Changes

- **全新的 Android 工程**：初始化一个基于 Java 的 Android 项目。
- **UI 界面实现**：严格按照设计图（@UI/ 目录下的首页.png、实时.png、离线.png）实现三个主要页面。
- **实时摄像头推理**：实现调用手机摄像头采集画面，并实时送入模型进行推理，展示识别结果、置信度和延迟。
- **离线视频推理**：实现从手机相册导入视频，提取视频帧进行推理，并展示结果。
- **模型集成**：集成 PyTorch Mobile 依赖，并加载预训练的 `best_model.ptl` 模型文件。
- **骨骼点可视化**：提供“显示骨骼点”的切换开关，在画面上绘制人体关键点信息（推测依赖于 MediaPipe 等前置处理）。

## Capabilities

### New Capabilities

- `ui-implementation`: 实现首页、实时识别页和离线推理页的 UI 布局与交互逻辑。
- `camera-integration`: 实现相机画面获取与实时帧处理。
- `video-processing`: 实现本地视频文件的选择、读取与逐帧解析。
- `model-inference`: 集成 PyTorch Lite 并实现手语识别模型的加载、前处理、推理与后处理（包含置信度与延迟计算）。
- `skeleton-visualization`: 实现人体骨骼关键点的提取与在预览层上的渲染绘制。

### Modified Capabilities

- 无

## Impact

- **依赖项**：需要引入 CameraX 或 Camera2 API 用于相机控制，引入 MediaPipe (可能用于骨骼点提取)，引入 PyTorch Mobile Android 端依赖。
- **权限**：应用需要申请相机权限（`CAMERA`）和读取外部存储权限（`READ_EXTERNAL_STORAGE` / 相册选择权限）。
- **文件系统**：模型文件需打包进 Android assets 目录中。