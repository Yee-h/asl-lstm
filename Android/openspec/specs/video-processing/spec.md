# video-processing Specification

## Purpose
TBD - created by archiving change asl-android-app. Update Purpose after archive.
## Requirements
### Requirement: 外部存储权限申请
应用在进入离线视频选择前 MUST 检查并申请读取外部存储权限。

#### Scenario: 权限拒绝
- **WHEN** 用户拒绝授予读取外部存储权限
- **THEN** 界面提示需开启权限才能使用离线视频导入功能。

### Requirement: 视频文件选择
应用 MUST 通过 Android 原生文件选择器或相册选择器支持用户选择本地视频文件。

#### Scenario: 成功选择视频
- **WHEN** 用户在相册选择器中选取视频并返回
- **THEN** 应用获取所选视频的 URI 并准备提取帧序列。

### Requirement: 视频帧提取与播放
应用 MUST 能从选取的本地视频文件中提取帧序列用于模型推理。

#### Scenario: 提取视频帧
- **WHEN** 用户选择一个视频进行离线推理
- **THEN** 应用内部解帧（如 MediaMetadataRetriever）提取每一帧并按顺序用于显示在预览区域和送入骨骼点提取/模型。

#### Scenario: 视频播放控制
- **WHEN** 视频正在推理
- **THEN** 界面中间预览区域顺序显示视频帧内容。

