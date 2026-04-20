## 1. 工程初始化与依赖配置

- [x] 1.1 使用 Android Studio/命令行初始化一个新的基于 Java 的 Android 项目（如包名 `com.example.asllstm`）。
- [x] 1.2 在 `build.gradle` 中添加核心依赖项：AndroidX 核心库、ConstraintLayout、CameraX 组件、PyTorch Mobile Lite 依赖库。
- [x] 1.3 在 `build.gradle` 中添加 MediaPipe Pose/Holistic 相关依赖用于骨骼点提取。
- [x] 1.4 在 `AndroidManifest.xml` 中声明 `CAMERA` 和 `READ_EXTERNAL_STORAGE` 权限。

## 2. UI 界面开发

- [x] 2.1 创建并实现 `activity_main.xml`（首页），添加“实时识别”和“离线推理”按钮及对应逻辑。
- [x] 2.2 创建并实现 `activity_realtime.xml`（实时页面），包含 Camera 预览视图、圆角扫描框、顶部栏、骨骼点开关、及底部的状态展示区（置信度、延迟、识别结果）。
- [x] 2.3 创建并实现 `activity_offline.xml`（离线页面），布局与实时页面相似，但中心带有“导入视频”图标和相关文字。

## 3. 摄像头集成与视频导入

- [x] 3.1 在 `RealtimeActivity.java` 中实现运行时相机权限申请。
- [x] 3.2 使用 CameraX 配置和启动相机预览，绑定到界面上的 `PreviewView`。
- [x] 3.3 在 `OfflineActivity.java` 中实现存储读取权限申请，并通过 Intent 调起系统文件选择器选取本地视频。
- [x] 3.4 实现选定视频的本地路径解析及使用 MediaPlayer/VideoView 或自定义解帧器进行播放。

## 4. 骨骼点提取与渲染

- [x] 4.1 集成 MediaPipe 逻辑，接收 CameraX 提供或视频解析出的图像帧，进行异步/同步骨骼点提取。
- [x] 4.2 创建自定义的 `SkeletonOverlayView.java`，覆盖在预览区域之上，根据 MediaPipe 提取的结果坐标实时绘制圆点与连线。
- [x] 4.3 绑定顶部“显示骨骼点”按钮的状态，控制 `SkeletonOverlayView` 的可见性或绘制逻辑。

## 5. 模型推理集成

- [x] 5.1 将 `D:\Document\Code\Android\asl-lstm\model\seed456_temporal_mask\best_model.ptl` 复制到项目的 `src/main/assets/` 目录下。
- [x] 5.2 编写 `ModelRunner.java` 类，使用 `LiteModuleLoader.load()` 加载 assets 中的 ptl 模型。
- [x] 5.3 实现一个基于时序的缓冲区（Queue 或 List），在每一帧提取出骨骼点后存入缓冲区；当积攒到模型所需的序列长度时，转换为 PyTorch `Tensor` 进行前向推理。
- [x] 5.4 处理模型输出，进行 Softmax 等后处理，获取置信度最高的类别索引和对应的标签文本。
- [x] 5.5 记录从图像送入骨骼点到模型推理完成的耗时（延迟 ms）。

## 6. 整合与优化

- [x] 6.1 将模型推理结果及延迟信息通过 Handler 或 LiveData 回调给主线程，更新实时/离线页面的对应 TextView。
- [x] 6.2 针对离线视频，确保帧提取、骨骼点分析与视频播放同步，不在 UI 线程阻塞。
- [x] 6.3 进行应用级测试：真机安装测试权限获取、视频选取、实时识别和离线识别的流畅度与结果正确性。