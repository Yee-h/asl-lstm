# ASL-LSTM 安卓端部署开发计划

> 版本: v1.0 | 创建日期: 2026-04-20
> 源模型: `src/checkpoints/seed456_temporal_mask/best_model.pth`
> 目标: 安卓端实现完整的手语识别流水线

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

### 1.2 Android 端当前问题

| # | 问题 | 严重度 | 说明 |
|---|---|---|---|
| 1 | **手部 42 点全部为零** | 致命 | 仅用 PoseLandmarker(33点)取前25点，手部索引25-66全零，手语识别失去最核心信息 |
| 2 | **面部 68 点全部为零** | 致命 | 未使用 FaceLandmarker，索引67-134全零，面部表情信息缺失 |
| 3 | **预处理流水线完全缺失** | 致命 | 无肩轴对齐、尺度归一化、EMA平滑、插值，模型输入分布与训练时不匹配 |
| 4 | **Z-Score 统计量硬编码且可能不准确** | 严重 | 坐标值未经预处理就做标准化，即使统计量正确也无效 |
| 5 | **使用 pose_landmarker_lite** | 中等 | lite 版精度低于 heavy，影响关键点提取质量 |
| 6 | **推理触发逻辑粗糙** | 中等 | 仅当缓冲区恰好达到 MAX_FRAMES 时才推理，缺少灵活的触发策略 |

### 1.3 模型文件验证结果

已有 `Android/model/seed456_temporal_mask/best_model.ptl` (4.28MB) 经 Python 验证:

- `forward(x, lengths)` 签名正确，输入 `(1, 90, 540)` + `long[1]` → 输出 `(1, 100)`
- 模型内部逻辑: pack_padded_sequence → LSTM → LayerNorm → Attention(单头加性) → FC
- 可直接使用，无需重新导出

---

## 二、开发步骤

### 步骤 1: 模型导出脚本与验证（Python 端）

**目标**: 编写可复用的导出脚本，确保 ptl 模型文件正确，导出标准化统计量。

**已有成果**: `best_model.ptl` 已验证可用。

**任务清单**:

- [ ] 1.1 编写 `src/model/export_lite.py` 导出脚本，记录完整的导出参数和方法
- [ ] 1.2 将正确的 `best_model.ptl` 复制到 `Android/app/src/main/assets/best_model.ptl`
- [ ] 1.3 导出 Z-Score 标准化统计量到 Java 可读格式（硬编码到 Java 类中）

**Z-Score 统计量** (来自 `WLASL100_train_stats.json`):

```
channels: [x, y, dx, dy]
mean:     [-0.02456032891334202, -0.11962947312034139, -0.0008191781815231185, 0.002500323402822938]
std:      [0.24632292725976973,   0.6917867353827447,   0.03799941547934166,   0.061761387958482336]
eps:      1e-6
```

**产出文件**:
- `src/model/export_lite.py` (新建)
- `Android/app/src/main/assets/best_model.ptl` (确保更新)

---

### 步骤 2: 补充 MediaPipe 模型文件

**目标**: 添加 Hand、Face 检测模型，替换 Pose lite 为 heavy。

**任务清单**:

- [ ] 2.1 将 `src/mediapipe_models/pose_landmarker_heavy.task` (30.7MB) 复制到 `Android/app/src/main/assets/`，替换 lite 版
- [ ] 2.2 将 `src/mediapipe_models/hand_landmarker.task` (7.8MB) 复制到 `Android/app/src/main/assets/`
- [ ] 2.3 将 `src/mediapipe_models/face_landmarker.task` (3.8MB) 复制到 `Android/app/src/main/assets/`
- [ ] 2.4 删除旧的 `pose_landmarker_lite.task`

**APK 体积影响**:

| 模型文件 | 大小 | 说明 |
|---|---|---|
| best_model.ptl | 4.3 MB | LSTM 模型 |
| pose_landmarker_heavy.task | 30.7 MB | Pose (替换 lite) |
| hand_landmarker.task | 7.8 MB | Hand (新增) |
| face_landmarker.task | 3.8 MB | Face (新增) |
| **合计** | **46.6 MB** | |

**产出文件**: `Android/app/src/main/assets/` 下的模型文件

---

### 步骤 3: 新建 KeypointExtractor.java — 完整 135 点提取

**目标**: 替代现有 `PoseLandmarkerHelper.java`，使用 Pose + Hand + Face 三个 MediaPipe 模型提取完整 135 个关键点。

**135 点格式** (与训练时严格一致):

```
[0-24]   Body 25 点    ← Pose Landmarker (33点映射到OpenPose Body 25)
[25-45]  Left Hand 21  ← Hand Landmarker (MediaPipe "Right" → 人体左手)
[46-66]  Right Hand 21 ← Hand Landmarker (MediaPipe "Left" → 人体右手)
[67-134] Face 68 点    ← Face Landmarker (478点映射到dlib 68点)
```

**任务清单**:

- [ ] 3.1 新建 `KeypointExtractor.java`，包含三个 MediaPipe Landmarker 实例
- [ ] 3.2 实现 Body 25 映射 (body_mapping 数组)
- [ ] 3.3 实现 Hand 42 映射 (左右手交换逻辑)
- [ ] 3.4 实现 Face 68 映射 (face_68_indices 数组)
- [ ] 3.5 优化策略: 先检测 Hand，无手则跳过 Pose/Face 检测
- [ ] 3.6 输出格式: `float[2][135]` (xy坐标) + `boolean[135]` (valid mask)

**Body 25 映射表** (OpenPose Body 25 索引 → MediaPipe Pose 索引):

```java
// MP索引 -1 表示需要计算平均值
// 1: Neck = (MP11 + MP12) / 2
// 8: MidHip = (MP23 + MP24) / 2
int[] BODY_MAPPING = {
    0,   // 0:  Nose
    -1,  // 1:  Neck = (MP11 + MP12) / 2
    12,  // 2:  RShoulder
    14,  // 3:  RElbow
    16,  // 4:  RWrist
    11,  // 5:  LShoulder
    13,  // 6:  LElbow
    15,  // 7:  LWrist
    -1,  // 8:  MidHip = (MP23 + MP24) / 2
    24,  // 9:  RHip
    26,  // 10: RKnee
    28,  // 11: RAnkle
    23,  // 12: LHip
    25,  // 13: LKnee
    27,  // 14: LAnkle
    5,   // 15: REye
    2,   // 16: LEye
    8,   // 17: REar
    7,   // 18: LEar
    32,  // 19: LBigToe
    31,  // 20: LSmallToe
    29,  // 21: LHeel
    31,  // 22: RBigToe
    32,  // 23: RSmallToe
    30,  // 24: RHeel
};
```

**Hand 左右手交换逻辑**:

```
MediaPipe 的 "Left"  = 图像左侧 = 人体右侧 → 映射到 OpenPose Right Hand (索引 46-66)
MediaPipe 的 "Right" = 图像右侧 = 人体左侧 → 映射到 OpenPose Left Hand (索引 25-45)
```

**Face 68 映射表** (dlib 68 点索引 → MediaPipe Face 478 点索引):

```java
int[] FACE_68_INDICES = {
    // 面部轮廓 (17 点)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    // 左眉 (5 点)
    70, 63, 105, 66, 107,
    // 右眉 (5 点)
    336, 296, 334, 293, 300,
    // 鼻梁 (4 点)
    168, 6, 197, 195,
    // 鼻尖下方 (5 点)
    5, 4, 1, 19, 94,
    // 左眼 (6 点)
    33, 160, 158, 133, 153, 144,
    // 右眼 (6 点)
    362, 385, 387, 263, 373, 380,
    // 外唇 (12 点)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308,
    // 内唇 (8 点)
    78, 191, 80, 81, 82, 13, 312, 311
};
```

**产出文件**: `KeypointExtractor.java` (新建)

---

### 步骤 4: 新建 PreprocessPipeline.java — 完整预处理流水线

**目标**: 将 Python `PreprocessHelper` 的完整流水线移植为 Java 类。

**流水线顺序** (与训练时严格一致):

```
原始关键点 (T, 2, 135) + mask (T, 135)
    ↓ 1. 短缺失插值 (gap ≤ 8)
    ↓ 2. EMA 平滑 (alpha=0.35)
    ↓ 3. 肩轴对齐 (旋转到水平)
    ↓ 4. 尺度归一化 (shoulder_torso_fusion: 0.7×肩宽 + 0.3×躯干)
    ↓ 5. 速度特征 (dx, dy)
    ↓ 6. Z-Score 标准化 (mean/std 硬编码)
    ↓ 7. 填充/截断到 max_frames=90, mask 无效点清零
输出: float[1][90][540] + int validLength
```

**任务清单**:

- [ ] 4.1 新建 `PreprocessPipeline.java`
- [ ] 4.2 实现 `interpolateMissingShortGaps()`: gap ≤ 8 帧的缺失点线性插值
- [ ] 4.3 实现 `smoothXY()`: EMA 平滑，alpha=0.35，仅对有效点
- [ ] 4.4 实现 `alignShoulderAxis()`: 将肩轴旋转到水平（2D旋转矩阵）
- [ ] 4.5 实现 `normalizeWithMask()`: 平移(肩中点为原点) + 缩放(0.7×肩宽 + 0.3×躯干)
- [ ] 4.6 实现 `addVelocityFeatures()`: dx, dy = frame[t] - frame[t-1]
- [ ] 4.7 实现 `applyStandardization()`: Z-Score 标准化，统计量硬编码
- [ ] 4.8 实现 `padOrTruncate()`: 填充零到 90 帧，或截断；mask 无效点对应特征清零
- [ ] 4.9 处理在线模式特有问题（见下文）

**尺度归一化在线模式适配**:

Python 版使用视频级中位数 (`np.median(shoulder_scales)`)，在实时模式下不可用。

解决方案: 维护最近 300 帧的肩宽/躯干长度的滑动窗口，取中位数作为尺度参考。

初始化前 30 帧使用默认值 1.0，后续使用滑动窗口中位数。

**肩轴对齐算法** (每帧独立):

```
1. 取左肩(LS=idx5)和右肩(RS=idx2)坐标
2. 计算肩轴向量 vec = LS - RS
3. 计算旋转角 angle = atan2(vec.y, vec.x)
4. 构造 2D 旋转矩阵 R(-angle)
5. 以肩中点 root = (LS + RS) / 2 为旋转中心
6. 对所有有效点: pt = R @ (pt - root) + root
```

**尺度归一化算法** (视频级):

```
1. 每帧计算肩宽 shoulder_dist = |LS - RS|
2. 每帧计算躯干长 torso_dist = |root - MidHip|
3. 取所有帧肩宽的中位数 shoulder_scale = median(shoulder_dists)
4. 取所有帧躯干长的中位数 torso_scale = median(torso_dists)
5. video_scale = 0.7 * shoulder_scale + 0.3 * torso_scale
6. 对所有帧: normalized = (data - root) / video_scale
7. mask=0 的点置零
```

**Z-Score 标准化算法**:

```
对每个通道 c ∈ {x, y, dx, dy}:
    standardized[:valid_len, c, :] = (data[:valid_len, c, :] - mean[c]) / (std[c] + eps)

其中:
    mean = [-0.02456, -0.11963, -0.00082, 0.00250]
    std  = [0.24632,  0.69179,  0.03800,  0.06176]
    eps  = 1e-6

标准化后，mask=0 的点清零
```

**产出文件**: `PreprocessPipeline.java` (新建)

---

### 步骤 5: 重构 ModelRunner.java

**目标**: 重构为使用新的 KeypointExtractor + PreprocessPipeline + 正确推理。

**任务清单**:

- [ ] 5.1 移除硬编码的 FEATURE_DIM 和旧标准化常量
- [ ] 5.2 `processFrame()` 接收 `(float[2][135], boolean[135])` 关键点 + mask
- [ ] 5.3 将关键点送入 PreprocessPipeline 维护帧缓冲区
- [ ] 5.4 实现灵活的推理触发策略:
  - 缓冲区达到 90 帧时自动推理
  - 连续 N 帧(≥15帧)无关键点时，使用当前缓冲区内容推理（动作结束检测）
- [ ] 5.5 推理调用: 从 PreprocessPipeline 获取 `float[1][90][540]` + `long[1]` 送入 PyTorch Lite
- [ ] 5.6 后处理: softmax + 取 top1 + ASL_LABELS 标签映射
- [ ] 5.7 推理完成后清空缓冲区，重置速度特征(lastFramePoints)

**产出文件**: `ModelRunner.java` (重写)

---

### 步骤 6: 修改 Activity 层适配

**目标**: RealtimeActivity 和 OfflineActivity 适配新的 KeypointExtractor 接口。

**任务清单**:

- [ ] 6.1 `RealtimeActivity.java`: 替换 PoseLandmarkerHelper → KeypointExtractor
- [ ] 6.2 `OfflineActivity.java`: 同上
- [ ] 6.3 `SkeletonOverlayView.java`: 保持现有绘制逻辑不变（仅使用 Pose 33 点绘制骨架）
- [ ] 6.4 删除 `PoseLandmarkerHelper.java`

**产出文件**: `RealtimeActivity.java`, `OfflineActivity.java` (修改), `PoseLandmarkerHelper.java` (删除)

---

### 步骤 7: 端到端验证

**目标**: 确保 Android 端推理结果与 Python 端一致。

**任务清单**:

- [ ] 7.1 Python 端编写验证脚本: 对同一帧序列，输出 PreprocessHelper 每步中间结果到文件
- [ ] 7.2 Android 端添加调试模式: 对同一输入，输出 PreprocessPipeline 每步中间结果到 Log
- [ ] 7.3 对比两端输出，确保数值一致 (容差 < 1e-3)
- [ ] 7.4 端到端测试: 同一视频，对比 Python 和 Android 的预测标签

**产出**: 验证报告，确认 Android 端流水线与 Python 端一致

---

## 三、文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/model/export_lite.py` | 新建 | 模型导出脚本 |
| `Android/app/src/main/assets/best_model.ptl` | 更新 | 确保 ptl 正确 |
| `Android/app/src/main/assets/pose_landmarker_heavy.task` | 新增 | 替换 lite (30.7MB) |
| `Android/app/src/main/assets/hand_landmarker.task` | 新增 | Hand 关键点模型 (7.8MB) |
| `Android/app/src/main/assets/face_landmarker.task` | 新增 | Face 关键点模型 (3.8MB) |
| `Android/app/src/main/assets/pose_landmarker_lite.task` | 删除 | 被 heavy 替代 |
| `Android/app/src/main/java/.../KeypointExtractor.java` | 新建 | 完整 135 点提取 |
| `Android/app/src/main/java/.../PreprocessPipeline.java` | 新建 | 完整预处理流水线 |
| `Android/app/src/main/java/.../ModelRunner.java` | 重写 | 使用新提取器+预处理+推理 |
| `Android/app/src/main/java/.../RealtimeActivity.java` | 修改 | 适配新接口 |
| `Android/app/src/main/java/.../OfflineActivity.java` | 修改 | 适配新接口 |
| `Android/app/src/main/java/.../PoseLandmarkerHelper.java` | 删除 | 被 KeypointExtractor 替代 |

---

## 四、依赖与版本

| 依赖 | 当前版本 | 是否需更新 |
|---|---|---|
| PyTorch Android Lite | 1.13.0 | 否 (ptl 已验证可用) |
| MediaPipe Tasks Vision | 0.10.14 | 需确认支持 Hand+Face |
| CameraX | 1.3.1 | 否 |
| compileSdk / minSdk | 36 | 否 |

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 3 个 MediaPipe 模型同时运行性能差 | 帧率低于 15fps | 先检测 Hand，无手则跳过 Pose/Face；降低分辨率输入 |
| 尺度归一化在线计算与离线不一致 | 精度下降 | 使用滑动窗口中位数近似，窗口足够大时差异 < 1% |
| PyTorch Lite 对 pack_padded_sequence 支持问题 | 推理崩溃 | 已验证 ptl 可正确执行；回退方案: 固定 lengths=MAX_FRAMES |
| APK 体积过大 (~47MB) | 用户体验 | 已选 heavy Pose；可后续考虑动态下载模型 |
| MediaPipe LIVE_STREAM 三模型时序冲突 | 回调乱序 | 使用统一 timestamp，三个检测器共享同一帧时间戳 |

---

## 六、开发顺序与时间估算

| 步骤 | 预计耗时 | 依赖 |
|---|---|---|
| 1. 模型导出与验证 | 0.5h | 无 |
| 2. 补充 MediaPipe 模型文件 | 0.5h | 无 |
| 3. KeypointExtractor.java | 2h | 步骤 2 |
| 4. PreprocessPipeline.java | 3h | 无 |
| 5. ModelRunner.java 重构 | 1.5h | 步骤 3, 4 |
| 6. Activity 层适配 | 1h | 步骤 5 |
| 7. 端到端验证 | 2h | 步骤 6 |
| **合计** | **~10.5h** | |

---

## 七、标签映射

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
