# skeleton-visualization Specification

## Purpose
TBD - created by archiving change asl-android-app. Update Purpose after archive.
## Requirements
### Requirement: 骨骼点提取集成
应用 MUST 集成 MediaPipe Pose 或类似引擎，以在实时和离线画面上提取骨骼点坐标。

#### Scenario: 成功提取骨骼点
- **WHEN** 应用启动骨骼点提取并输入图像帧
- **THEN** 每帧均可返回人体关键点的相对坐标/绝对坐标信息。

### Requirement: 骨骼点可视化渲染
应用 MUST 提供一个绘制层（Canvas/SurfaceView），用于在预览画面上覆盖绘制身体骨骼线和节点。

#### Scenario: 绘制骨骼连线
- **WHEN** 用户在“显示骨骼点”开启状态下
- **THEN** 提取到的节点通过点和线段的方式覆盖在当前视频帧/相机画面上。

#### Scenario: 骨骼连线消失
- **WHEN** 用户在“显示骨骼点”关闭状态下
- **THEN** 停止在叠加层上绘制骨骼连线，只保留视频/相机画面。

