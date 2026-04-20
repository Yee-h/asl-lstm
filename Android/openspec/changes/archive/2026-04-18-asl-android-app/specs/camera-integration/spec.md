## ADDED Requirements

### Requirement: 相机权限申请
应用在进入实时识别页前 MUST 检查并申请系统相机权限。

#### Scenario: 权限拒绝
- **WHEN** 用户拒绝授予相机权限
- **THEN** 界面提示需开启权限才能使用实时识别功能，且不开启摄像头预览。

### Requirement: 摄像头预览流获取
应用在实时识别页 MUST 调用设备的摄像头获取视频帧，并将其显示在界面中央的预览区域。

#### Scenario: 成功启动摄像头
- **WHEN** 应用具有相机权限并成功初始化 CameraX
- **THEN** 中间的预览界面开始显示设备摄像头捕捉到的实时画面，且长宽比例符合设计要求。

### Requirement: 图像帧回传
应用 MUST 能够从摄像头的数据流中持续获取用于骨骼点提取和模型推理的图像帧。

#### Scenario: 获取 ImageProxy
- **WHEN** 摄像头每一帧数据就绪
- **THEN** 应用的 ImageAnalysis 组件将帧数据以固定分辨率传送到后处理流程。