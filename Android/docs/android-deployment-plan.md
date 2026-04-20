# ASL-LSTM 安卓端部署开发计划

> 版本: v1.2 | 创建日期: 2026-04-20 | 更新日期: 2026-04-20
> 源模型: `src/checkpoints/seed456_temporal_mask/best_model.pth`
> 目标: 安卓端实现完整的手语识别流水线
> 测试设备: Xiaomi 15 Pro, Snapdragon 8 Elite

---

## 一、现状与问题诊断

### 1.1 Python 桌面端完整流水线

```
摄像头帧 → [KeypointExtractor] → (2, 135) 关键点
         → [PreprocessHelper]  → (max_frames, 4, 135) 归一化特征
         → [Z-Score 标准化]    → (max_frames, 540) 张量
         → [BiLSTMAttention]   → (100,) logits → softmax → 预测标签
```

关键参数:

- 关键点: 135 点 (Body 25 + Left Hand 21 + Right Hand 21 + Face 68)
- 特征通道: 4 通道 (x, y, dx, dy)
- 序列长度: max_frames = 90
- 输入维度: 540 (= 4 × 135)
- 模型: BiLSTMAttention, hidden=128, layers=2, bidirectional, num_heads=1, use_layer_norm=True
- 标签数: 100 类 (WLASL100)

### 1.2 已完成的工作 (v1.0 → v1.1)

| # | 步骤 | 状态 | 说明 |
|---|---|---|---|
| 1 | 模型导出与验证 | ✅ 完成 | `best_model.ptl` 已验证可用，Z-Score 统计量已硬编码到 Java |
| 2 | MediaPipe 模型文件 | ✅ 完成 | pose_heavy + hand + face 已部署，lite 已删除 |
| 3 | KeypointExtractor.java | ✅ 完成 | Pose+Hand+Face 三模型 135 点提取，无手时跳过 Pose/Face |
| 4 | PreprocessPipeline.java | ✅ 完成 | 7 步流水线完整移植（插值→EMA→肩轴对齐→尺度归一化→速度特征→Z-Score→Padding） |
| 5 | ModelRunner.java 重构 | ✅ 完成 | 使用新提取器+预处理+灵活推理触发 |
| 6 | Activity 层适配 | ✅ 完成 | 使用 KeypointExtractor 替代 PoseLandmarkerHelper，骨骼绘制升级为 135 点 |
| 7 | 竖屏方向修复 | ✅ 完成 | 前置摄像头 Bitmap 旋转+镜像，视频帧旋转，骨骼点正确映射到竖屏 |

### 1.3 已实施的优化 (v1.2)

| # | 优化 | 严重度 | 方案 | 状态 |
|---|---|---|---|---|
| 1 | **三模型推理帧率偏低** | 高 | Face 检测降频到每 3 帧运行一次，中间帧复用 cached 结果 | ✅ 已实施 |
| 2 | **输入分辨率过高** | 高 | MediaPipe 输入降采样到 480px max dim，关键点坐标为归一化值不受影响 | ✅ 已实施 |
| 3 | **尺度归一化预热** | 高 | 前 30 帧使用首帧肩宽作为初始 scale（替代默认值 1.0），更接近真实尺度 | ✅ 已实施 |
| 4 | **EMA 状态跨推理保留** | 中 | `clearBuffer()` 仅清空帧缓冲，保留 EMA/Scale 滑动窗口状态；`resetAll()` 完全重置 | ✅ 已实施 |
| 5 | **置信度阈值过滤** | 中 | 低于 15% 置信度的推理结果显示为 "..." 而非错误标签 | ✅ 已实施 |

### 1.4 待优化项 (v1.2→)

| # | 问题 | 严重度 | 方案 |
|---|---|---|---|
| 1 | **动作结束检测固定阈值** | 中 | 当前固定 15 帧无手触发，应支持自适应阈值（快手势 10 帧，慢手势 20 帧） |
| 2 | **Bitmap 内存管理** | 中 | CameraX Bitmap 和降采样 Bitmap 未显式回收，依赖 GC 可能导致短暂卡顿 |
| 3 | **NNAPI/GPU 推理加速** | 低 | PyTorch Lite 支持 NNAPI delegate，可加速 LSTM 推理 |
| 4 | **APK 体积 ~47MB** | 低 | 三个 MediaPipe 模型合计 42MB，可考虑按需下载模型 |
| 5 | **端到端 Python-Android 一致性验证** | 高 | 尚未编写验证脚本对比两端 PreprocessPipeline 数值输出 |

---

## 二、已实现架构

### 2.1 数据流

```
CameraX ImageProxy
    ↓ rotateAndMirrorBitmap(degrees, mirrorH=true)  [前置摄像头]
    ↓ rotateBitmap(degrees)                          [视频文件]
Bitmap (竖屏方向)
    ↓ KeypointExtractor.detectAsync(bitmap, timestampMs)
    ↓ Hand → (有手?) → Pose → Face → mapTo135Keypoints()
float[2][135] + boolean[135] + hasHands
    ↓ ModelRunner.processFrame(keypoints, validMask, hasHands)
    ↓ PreprocessPipeline.addFrame() → 缓冲区积累
    ↓ (缓冲区满90帧 or 连续15帧无手) → PreprocessPipeline.process()
    ↓ 7步预处理 → float[MAX_FRAMES * 540] + validLength
    ↓ PyTorch Lite forward(x, lengths) → softmax → ASL_LABELS[top1]
推理结果: label + confidence + inferenceTime
```

### 2.2 文件结构

```
Android/app/src/main/java/com/ye/asl_lstm/
├── KeypointExtractor.java    # Pose+Hand+Face 三模型 135点提取
├── PreprocessPipeline.java   # 7步预处理流水线
├── ModelRunner.java          # PyTorch Lite 推理 + 缓冲区管理
├── RealtimeActivity.java     # 实时摄像头界面 (旋转+镜像)
├── OfflineActivity.java      # 视频文件分析界面 (旋转)
├── SkeletonOverlayView.java  # 135点骨骼绘制 (Body+Hand+Face)
└── MainActivity.java         # 主界面入口

Android/app/src/main/assets/
├── best_model.ptl                  (4.3 MB)  LSTM 模型
├── pose_landmarker_heavy.task      (30.7 MB)  Pose 模型
├── hand_landmarker.task            (7.8 MB)   Hand 模型
├── face_landmarker.task            (3.8 MB)   Face 模型
└── zscore_stats.json               (0.1 KB)   Z-Score 统计量（参考用，已硬编码到 Java）
```

### 2.3 已修复的关键 Bug

| Bug | 严重度 | 修复说明 |
|-----|--------|---------|
| 数据展平顺序错误 | 致命 | `padOrTruncate` 中 `v*4+c`（交错）改为 `c*135+v`（分组），匹配 Python `(T, C, V)` 展平顺序 |
| 双重发射 | 致命 | `tryEmit()` 被 Pose/Face 两个回调各触发一次，添加 `frameConsumed` 标志确保每帧只发射一次 |
| Handedness 空列表 | 中等 | `handedness().get(idx).get(0)` 添加空列表保护 `.isEmpty()` 检查 |
| 竖屏方向错误 | 严重 | CameraX 返回横屏 Bitmap，骨骼点坐标系与竖屏 View 不匹配；新增 `rotateAndMirrorBitmap()` 在前置摄像头旋转+镜像 |
| normalizeWithMask 死代码 | 低 | 移除 `if (!masks[t][v])` 在 `continue` 之后的不可达代码块 |

---

## 三、待优化项

### 3.1 性能优化（帧率）

**目标**: 在 Xiaomi 15 Pro (Snapdragon 8 Elite) 上达到 15+ fps

**策略**:

| # | 优化方向 | 预期收益 | 复杂度 |
|---|---|---|---|
| P1 | Face 检测降频: 每 3 帧检测一次 Face，中间帧复用上次结果 | +20-30% fps | 中 |
| P2 | 输入分辨率降采样: 将 Bitmap 缩放到 320x240 后再送 MediaPipe | +30-50% fps | 低 |
| P3 | 使用 IMAGE 模式替代 LIVE_STREAM: 解除 MediaPipe 时序约束，但需自行管理帧顺序 | +15-25% fps | 高 |
| P4 | NNAPI/GPU 推理加速: PyTorch Lite 支持 NNAPI delegate | +20-40% 推理速度 | 中 |
| P5 | Hand 检测降频: 每 2 帧检测一次 Hand，无手时反馈更快 | +10-15% fps | 低 |

**P1 - Face 降频详细方案**:

```java
// KeypointExtractor.detectAsync() 中
handLandmarker.detectAsync(mpImage, timestampMs);  // 每帧都运行
poseLandmarker.detectAsync(mpImage, timestampMs);   // 有手时每帧运行

// Face: 有手时每 3 帧运行一次，其他帧复用上次结果
faceSkipCounter++;
if (hasHandsLastFrame && faceSkipCounter >= 3) {
    faceLandmarker.detectAsync(mpImage, timestampMs);
    faceSkipCounter = 0;
}
// tryEmit() 中: 当 faceResult 为 null 但有上次缓存时使用缓存
```

**P2 - 分辨率降采样详细方案**:

```java
// detectAsync() 中: 将 1920x1080 的 Bitmap 缩放到 320x240
Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 320, 240, true);
MPImage mpImage = new BitmapImageBuilder(scaledBitmap).build();
// 注意: 关键点坐标仍然是归一化的 [0,1]，降采样不影响坐标值
```

### 3.2 精度优化

| # | 优化方向 | 说明 |
|---|---|---|
| A1 | 尺度归一化预热优化 | 前30帧使用滑动窗口中位数（而非默认值1.0），改为使用第一帧的肩宽作为初始值 |
| A2 | EMA 状态跨推理重置 | 推理后清空缓冲区时，EMA 状态应该保留还是重置？当前重置，可能导致下一手势初始帧平滑跳变 |
| A3 | 动作结束检测调优 | 当前固定15帧无手触发，应根据手势速度动态调整（快手势10帧，慢手势20帧） |
| A4 | 置信度阈值过滤 | 推理结果中置信度低于阈值的应忽略，避免噪声误识别 |

### 3.3 端到端验证

| # | 验证项 | 说明 |
|---|---|---|
| V1 | Python-Android 数值一致性 | 对同一视频，对比两端 PreprocessPipeline 每步的中间输出（容差 < 1e-3） |
| V2 | 标签一致性测试 | 同一视频在 Python 和 Android 上应输出相同预测标签 |
| V3 | 实时识别延迟测试 | 测量从手语动作开始到结果输出的端到端延迟 |

---

## 四、依赖与版本

| 依赖 | 版本 | 说明 |
|---|---|---|
| PyTorch Android Lite | 1.13.0 | ptl 模型已验证可用 |
| MediaPipe Tasks Vision | 0.10.14 | 支持 Pose+Hand+Face 同时 LIVE_STREAM |
| CameraX | 1.3.1 | 支持 ImageProxy 旋转信息 |
| compileSdk / minSdk | 36 | Android 16 (Upside Down Cake) |

---

## 五、目标设备测试基准

| 指标 | 当前值 | 目标值 | 说明 |
|---|---|---|---|
| 关键点提取帧率 | ~8-12 fps | 15+ fps | Pose+Hand+Face 三模型同时运行 |
| 推理延迟 | ~20-50 ms | < 30 ms | 单次 LSTM forward |
| 端到端延迟 | ~2-5 秒 | < 3 秒 | 从动作开始到结果输出 |
| 准确率 | 未验证 | > 70% | 与 Python 端对比 |
| 内存占用 | ~200-300 MB | < 500 MB | 三模型 + 推理引擎 |

> 测试设备: Xiaomi 15 Pro, SoC: Snapdragon 8 Elite, 16GB RAM, Android 16

---

## 六、标签映射

ASL 100 类标签 (与 `wlasl_100_maplabels.json` 一致，索引 0-99):

```java
String[] ASL_LABELS = {
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
    "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
    "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
    "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate",
    "hat", "hearing", "kiss", "language", "later", "man", "shirt", "study", "tall", "white",
    "wrong", "accident", "apple", "bird", "change", "color", "corn", "cow", "dance", "dark",
    "doctor", "eat", "enjoy", "forget", "give", "last", "meet", "pink", "pizza", "play",
    "school", "secretary", "short", "time", "want", "work", "africa", "basketball", "birthday", "brown",
    "but", "cheat", "city", "cook", "decide", "full", "how", "jacket", "letter", "medicine",
    "need", "paint", "paper", "pull", "purple", "right", "same", "son", "tell", "thursday"
};
```