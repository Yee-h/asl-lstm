from __future__ import annotations

import os
from dataclasses import dataclass


# =============================================================================
# 统一配置中心（分模块）
#
# 设计说明：
# 1) 所有配置统一集中在本文件，按“路径/预处理/模型/训练/推理/UI”等模块分组。
# 2) 业务代码仅通过 cfg.<分组>.<字段> 访问配置，例如：cfg.TRAINING.batch_size。
# 3) 不再提供旧版全局常量别名（如 cfg.BATCH_SIZE / cfg.LEARNING_RATE）。
# =============================================================================


# -----------------------------
# 基础上下文（仅供本文件内部构建使用）
# -----------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_SCALE = 100


@dataclass(frozen=True)
class PathsConfig:
    # 项目根目录绝对路径
    project_root: str
    # 当前实验数据规模（100/300/1000/2000）
    dataset_scale: int
    # 预处理后数据目录（按规模分子目录）
    processed_data_dir: str
    # 标签映射文件路径（maplabels）
    label_map_path: str
    # 训练集 HDF5 路径
    train_data_path: str
    # 验证集 HDF5 路径
    val_data_path: str
    # 测试集 HDF5 路径
    test_data_path: str
    # 模型检查点保存目录
    model_save_dir: str
    # 默认评估模型路径（best_model）
    test_model_path: str
    # 原始数据根目录
    raw_data_dir: str
    # videoId_word 目录（用于 gloss 映射等）
    videoid_word_dir: str
    # 默认元数据 JSON 路径（WLASL_v0.3）
    default_json_path: str
    # gloss 映射 JSON 路径
    wlasl_gloss_json: str
    # 默认视频目录（随数据规模变化）
    default_video_dir: str
    # 默认预处理输出根目录
    default_output_dir: str
    # 默认输出文件名前缀
    default_output_prefix: str
    # 默认日志输出目录
    log_dir: str


@dataclass(frozen=True)
class SequenceConfig:
    # 统一时序长度（不足补零，超出重采样）
    max_frames: int
    # 单帧关键点数量（WLASL 135点）
    num_landmarks: int
    # 是否启用二阶动态特征（ddx, ddy）
    enable_accel_feature: bool
    # 基础特征通道定义（按顺序展平）
    base_feature_channels: tuple[str, ...]

    @property
    def landmark_dim(self) -> int:
        # 每个关键点的通道数（由通道定义自动推导）
        return len(self.base_feature_channels)

    @property
    def input_size(self) -> int:
        # 模型输入维度 = 通道数 × 关键点数
        return self.landmark_dim * self.num_landmarks

    @property
    def num_classes(self) -> int:
        # 类别数与数据规模一致（WLASL100 -> 100 类）
        return _DATASET_SCALE


@dataclass(frozen=True)
class PreprocessConfig:
    # 预处理流水线版本号（用于追踪数据产物来源）
    pipeline_version: str
    # 是否输出预处理调试信息（质量统计、过滤日志等）
    save_debug: bool
    # 预处理多进程 worker 数（None=自动，1=单进程）
    num_workers: int | None
    # 是否启用缺失关键点插值
    enable_missing_interp: bool
    # 允许插值的最大连续缺失长度
    interp_max_gap: int
    # 样本最低有效关键点比例阈值（低于阈值可过滤）
    min_valid_ratio_per_sample: float
    # 单帧最少有效关键点数（实时推理有效帧判定）
    min_valid_keypoints_per_frame: int
    # 是否启用肩轴对齐（旋转归一化）
    enable_shoulder_axis_align: bool
    # 尺度归一化策略（如 shoulder_torso_fusion）
    scale_mode: str
    # 归一化数值稳定项
    normalize_eps: float
    # 是否启用坐标平滑
    enable_xy_smooth: bool
    # 平滑方法（当前支持 ema）
    smooth_method: str
    # EMA 平滑系数（越大越敏感，越小越平滑）
    smooth_ema_alpha: float
    # 是否启用训练集统计标准化
    enable_standardize: bool
    # 训练集统计量文件路径
    feature_stats_path: str
    # 标准化数值稳定项
    standardize_eps: float
    # 不同子集规模对应的 nslt JSON 路径映射
    subset_json_map: dict[int, str]


@dataclass(frozen=True)
class MediapipeConfig:
    # MediaPipe 模型目录
    model_dir: str
    # Pose 模型文件路径
    pose_model_path: str
    # Hand 模型文件路径
    hand_model_path: str
    # Face 模型文件路径
    face_model_path: str
    # Pose 模型下载 URL
    pose_model_url: str
    # Hand 模型下载 URL
    hand_model_url: str
    # Face 模型下载 URL
    face_model_url: str
    # Pose 检测置信度阈值
    pose_min_det_conf: float
    # Pose 存在置信度阈值
    pose_min_presence_conf: float
    # Pose 跟踪置信度阈值
    pose_min_track_conf: float
    # Hand 检测置信度阈值
    hand_min_det_conf: float
    # Hand 存在置信度阈值
    hand_min_presence_conf: float
    # Hand 跟踪置信度阈值
    hand_min_track_conf: float
    # Face 检测置信度阈值
    face_min_det_conf: float
    # Face 存在置信度阈值
    face_min_presence_conf: float
    # Face 跟踪置信度阈值
    face_min_track_conf: float


@dataclass(frozen=True)
class ModelConfig:
    # LSTM 隐藏维度
    hidden_size: int
    # LSTM 层数
    num_layers: int
    # 是否双向 LSTM
    bidirectional: bool
    # 模型级 Dropout
    dropout: float
    # 交叉熵标签平滑系数
    label_smoothing: float
    # 是否启用注意力机制
    use_attention: bool
    # 注意力中间维度
    attention_dim: int
    # 是否在 LSTM 输出后添加 LayerNorm（E04）
    use_layer_norm: bool
    # 注意力头数（E06，1=单头退化为原始加性注意力行为，推荐4）
    num_heads: int


@dataclass(frozen=True)
class AugmentationConfig:
    # 旋转角度范围（度）
    rotation_range: float
    # 随机缩放下限
    scale_min: float
    # 随机缩放上限
    scale_max: float
    # 随机平移范围（归一化坐标）
    translate: float
    # 高斯噪声标准差
    noise_std: float
    # 水平翻转概率
    hflip_prob: float
    # 翻转时是否交换左右关键点语义
    hflip_swap_lr: bool
    # 零中心坐标翻转策略（x -> -x）
    hflip_zero_centered: bool
    # 时间扭曲概率
    time_warp_prob: float
    # 时间扭曲最小倍率
    time_warp_min: float
    # 时间扭曲最大倍率
    time_warp_max: float
    # 随机丢帧概率
    frame_dropout_prob: float
    # 最大丢帧比例
    frame_dropout_max_ratio: float
    # 时间掩码概率（E05）
    temporal_mask_prob: float
    # 时间掩码单段最大长度比例（E05）
    temporal_mask_max_ratio: float


@dataclass(frozen=True)
class TrainingConfig:
    # ==================== 基础训练参数 ====================
    # 单卡实际 batch 大小
    batch_size: int
    # 初始学习率
    learning_rate: float
    # L2 权重衰减
    weight_decay: float
    # 最大训练轮数
    num_epochs: int
    # 训练设备（"cuda" / "cpu"）
    device: str
    # 全局随机种子
    seed: int

    # ==================== 可复现性配置 ====================
    # 是否启用 cudnn 确定性模式（提升可复现性）
    deterministic: bool
    # 是否启用 cudnn benchmark（提升速度但降低可复现性）
    cudnn_benchmark: bool
    # 是否强制使用确定性算子（可能影响性能，必要时可关闭）
    use_deterministic_algorithms: bool

    # ==================== 保存与梯度配置 ====================
    # 模型定期保存间隔
    save_every_n_epochs: int
    # 梯度裁剪阈值（<=0 表示关闭）
    grad_clip_max_norm: float
    # 梯度累积步数（等效 batch 放大）
    grad_accum_steps: int

    # ==================== 学习率调度器 ====================
    # 学习率调度器类型（"plateau" 或 "cosine_warm"）
    scheduler_type: str
    # ReduceLROnPlateau 衰减因子
    scheduler_factor: float
    # ReduceLROnPlateau patience
    scheduler_patience: int
    # 学习率下限
    scheduler_min_lr: float
    # CosineAnnealingWarmRestarts: 初始重启周期（epoch数）
    cosine_T0: int
    # CosineAnnealingWarmRestarts: 周期倍增因子
    cosine_T_mult: int

    # ==================== 学习率 Warmup ====================
    # Warmup 轮数（0 表示关闭）
    warmup_epochs: int
    # Warmup 起始学习率
    warmup_start_lr: float

    # ==================== 早停配置 ====================
    # 是否启用早停
    early_stopping_enabled: bool
    # 早停耐心轮数
    early_stopping_patience: int
    # 早停监控指标（val_acc 或 val_loss）
    early_stopping_metric: str
    # 早停最小改进阈值
    early_stopping_min_delta: float

    # ==================== 采样与加载配置 ====================
    # 是否启用类别均衡采样
    use_weighted_sampler: bool
    # 采样权重指数（1=逆频率）
    sampler_power: float
    # DataLoader worker 数
    dataloader_num_workers: int

    # ==================== EMA 配置 ====================
    # 是否启用参数指数滑动平均（EMA）
    use_ema: bool
    # EMA 衰减系数
    ema_decay: float
    # EMA 开始生效的 epoch（1-based）
    ema_start_epoch: int

    # ==================== 数据增强配置 ====================
    # Sequence-level Mixup alpha（<=0 表示关闭）
    mixup_alpha: float


@dataclass(frozen=True)
class EvaluationConfig:
    """评估配置：统一管理模型评估相关参数"""

    # ==================== 模型路径配置 ====================
    # 当前评估模型路径（可通过修改此值切换不同模型）
    model_path: str
    # 集成评估模型路径列表（为空时使用单模型评估）
    ensemble_model_paths: tuple[str, ...]

    # ==================== TTA 配置 ====================
    # 是否启用水平翻转 TTA
    use_tta_hflip: bool

    # ==================== 评估参数 ====================
    # 评估时是否输出详细分类报告
    verbose_report: bool
    # 是否保存评估报告到日志
    save_report: bool


@dataclass(frozen=True)
class InferenceConfig:
    # 摄像头索引
    camera_index: int
    # 摄像头采集宽度
    camera_width: int
    # 摄像头采集高度
    camera_height: int
    # 摄像头目标帧率
    camera_fps: int
    # 推理间隔（每 N 帧推理一次）
    inference_interval: int


@dataclass(frozen=True)
class UIConfig:
    # 中文字体候选路径（按顺序尝试）
    chinese_font_paths: tuple[str, ...]
    # 主文本字号
    font_size: int
    # 次级文本字号
    font_small_size: int
    # 退出按钮文案
    exit_button_text: str
    # 退出按钮尺寸
    exit_button_size: tuple[int, int]
    # 退出按钮边距（右、上）
    exit_button_margin: tuple[int, int]
    # 骨骼开关文案（显示中）
    skeleton_button_text_on: str
    # 骨骼开关文案（隐藏中）
    skeleton_button_text_off: str
    # 骨骼开关按钮尺寸
    skeleton_button_size: tuple[int, int]
    # 骨骼开关与退出按钮间距
    skeleton_button_gap: int
    # 关键点半径
    skeleton_point_radius: int
    # 关键点颜色（BGR）
    skeleton_point_color: tuple[int, int, int]
    # 骨架线颜色（BGR）
    skeleton_line_color: tuple[int, int, int]
    # 骨架线宽
    skeleton_line_thickness: int


_processed_data_dir = os.path.join(
    _PROJECT_ROOT,  # 项目根目录
    "dataset",  # 数据目录
    "processed",  # 预处理数据目录
    f"WLASL{_DATASET_SCALE}",  # 按数据规模分子目录
)


PATHS = PathsConfig(
    project_root=_PROJECT_ROOT,  # 项目根目录
    dataset_scale=_DATASET_SCALE,  # 当前使用数据规模
    processed_data_dir=_processed_data_dir,  # 预处理后数据根目录
    label_map_path=os.path.join(
        _processed_data_dir, f"wlasl_{_DATASET_SCALE}_maplabels.json"
    ),  # 标签映射路径
    train_data_path=os.path.join(
        _processed_data_dir, f"WLASL{_DATASET_SCALE}_135-Train.hdf5"
    ),  # 训练集 HDF5
    val_data_path=os.path.join(
        _processed_data_dir, f"WLASL{_DATASET_SCALE}_135-Val.hdf5"
    ),  # 验证集 HDF5
    test_data_path=os.path.join(
        _processed_data_dir, f"WLASL{_DATASET_SCALE}_135-Test.hdf5"
    ),  # 测试集 HDF5
    model_save_dir=os.path.join(_PROJECT_ROOT, "src", "checkpoints"),  # 模型保存目录
    test_model_path=os.path.join(
        _PROJECT_ROOT, "src", "checkpoints", "best_model.pth"
    ),  # 默认评估模型路径
    raw_data_dir=os.path.join(_PROJECT_ROOT, "dataset", "raw"),  # 原始数据目录
    videoid_word_dir=os.path.join(_PROJECT_ROOT, "dataset", "videoId_word"),  # videoId_word 目录
    default_json_path=os.path.join(
        _PROJECT_ROOT, "dataset", "raw", "WLASL_v0.3.json"
    ),  # 默认元数据 JSON
    wlasl_gloss_json=os.path.join(
        _PROJECT_ROOT, "dataset", "videoId_word", "WLASL_v0.3.json"
    ),  # gloss 映射 JSON
    default_video_dir=os.path.join(
        _PROJECT_ROOT, "dataset", "raw", f"WLASL{_DATASET_SCALE}"
    ),  # 默认视频目录
    default_output_dir=os.path.join(_PROJECT_ROOT, "dataset", "processed"),  # 默认输出根目录
    default_output_prefix="WLASL",  # 默认文件名前缀
    log_dir=os.path.join(_PROJECT_ROOT, "logs"),  # 日志目录
)


SEQUENCE = SequenceConfig(
    max_frames=90,  # 统一帧长
    num_landmarks=135,  # 关键点数量
    enable_accel_feature=False,  # 先关闭 ddx/ddy，保持稳定
    base_feature_channels=("x", "y", "dx", "dy"),  # 当前基础通道
)


PREPROCESS = PreprocessConfig(
    pipeline_version="v3",  # 预处理版本
    save_debug=True,  # 输出调试信息
    num_workers=None,  # None=自动核数
    enable_missing_interp=True,  # 启用短缺失插值
    interp_max_gap=8,  # 最大插值缺失段长度
    min_valid_ratio_per_sample=0.35,  # 样本有效率阈值
    min_valid_keypoints_per_frame=5,  # 实时推理帧有效点阈值
    enable_shoulder_axis_align=True,  # 启用肩轴对齐
    scale_mode="shoulder_torso_fusion",  # 尺度归一化策略
    normalize_eps=1e-6,  # 归一化稳定项
    enable_xy_smooth=True,  # 启用平滑
    smooth_method="ema",  # 平滑方法
    smooth_ema_alpha=0.35,  # EMA 系数
    enable_standardize=True,  # 启用标准化
    feature_stats_path=os.path.join(
        _processed_data_dir, f"WLASL{_DATASET_SCALE}_train_stats.json"
    ),  # 训练统计量路径
    standardize_eps=1e-6,  # 标准化稳定项
    subset_json_map={
        100: os.path.join(PATHS.raw_data_dir, "WLASL100", "nslt_100.json"),  # WLASL100 子集 JSON
        300: os.path.join(PATHS.raw_data_dir, "WLASL300", "nslt_300.json"),  # WLASL300 子集 JSON
        1000: os.path.join(
            PATHS.raw_data_dir, "WLASL1000", "nslt_1000.json"
        ),  # WLASL1000 子集 JSON
        2000: os.path.join(
            PATHS.raw_data_dir, "WLASL2000", "nslt_2000.json"
        ),  # WLASL2000 子集 JSON
    },
)


MEDIAPIPE = MediapipeConfig(
    model_dir=os.path.join(_PROJECT_ROOT, "src", "mediapipe_models"),  # 模型目录
    pose_model_path=os.path.join(
        _PROJECT_ROOT, "src", "mediapipe_models", "pose_landmarker_heavy.task"
    ),  # Pose 模型路径
    hand_model_path=os.path.join(
        _PROJECT_ROOT, "src", "mediapipe_models", "hand_landmarker.task"
    ),  # Hand 模型路径
    face_model_path=os.path.join(
        _PROJECT_ROOT, "src", "mediapipe_models", "face_landmarker.task"
    ),  # Face 模型路径
    pose_model_url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",  # Pose 下载地址
    hand_model_url="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",  # Hand 下载地址
    face_model_url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",  # Face 下载地址
    pose_min_det_conf=0.5,  # Pose 检测阈值
    pose_min_presence_conf=0.5,  # Pose 存在阈值
    pose_min_track_conf=0.5,  # Pose 跟踪阈值
    hand_min_det_conf=0.5,  # Hand 检测阈值
    hand_min_presence_conf=0.5,  # Hand 存在阈值
    hand_min_track_conf=0.5,  # Hand 跟踪阈值
    face_min_det_conf=0.5,  # Face 检测阈值
    face_min_presence_conf=0.5,  # Face 存在阈值
    face_min_track_conf=0.5,  # Face 跟踪阈值
)


MODEL = ModelConfig(
    hidden_size=128,  # LSTM 隐藏维度（回归到小样本更稳的容量）
    num_layers=2,  # LSTM 层数
    bidirectional=True,  # 双向 LSTM
    dropout=0.35,  # Dropout（Phase 2: 回退至基线值，避免过度正则化）
    label_smoothing=0.03,  # 标签平滑（Phase 2: 回退至基线值）
    use_attention=True,  # 使用注意力
    attention_dim=32,  # 注意力维度（单头加性注意力时使用，多头时忽略）
    use_layer_norm=True,  # E04: 在 LSTM 输出后添加 LayerNorm
    num_heads=1,  # 注意力头数（1=单头加性注意力，E06实验证明多头对小数据有害，回退为1）
)


AUGMENTATION = AugmentationConfig(
    rotation_range=18.0,  # 随机旋转角度范围
    scale_min=0.88,  # 缩放下限
    scale_max=1.12,  # 缩放上限
    translate=0.10,  # 平移幅度
    noise_std=0.0030,  # 噪声强度
    hflip_prob=0.25,  # 水平翻转概率
    hflip_swap_lr=True,  # 翻转后交换左右语义
    hflip_zero_centered=True,  # 零中心翻转策略
    time_warp_prob=0.16,  # 时间扭曲概率
    time_warp_min=0.90,  # 时间扭曲下限
    time_warp_max=1.10,  # 时间扭曲上限
    frame_dropout_prob=0.10,  # 丢帧概率
    frame_dropout_max_ratio=0.08,  # 最大丢帧比例
    temporal_mask_prob=0.30,  # 时间掩码概率（E05）
    temporal_mask_max_ratio=0.15,  # 时间掩码单段最大帧数比例（E05）
)


TRAINING = TrainingConfig(
    # ==================== 基础训练参数 ====================
    batch_size=4,  # 单步 batch
    learning_rate=8e-4,  # 初始学习率
    weight_decay=4e-4,  # L2 正则
    num_epochs=600,  # 最大轮数
    device="cuda",  # 期望设备
    seed=42,  # 随机种子
    # ==================== 可复现性配置 ====================
    deterministic=True,  # 启用确定性模式，便于复现实验
    cudnn_benchmark=False,  # 关闭 benchmark，避免引入非确定性
    use_deterministic_algorithms=False,  # 默认不强制全部算子确定性
    # ==================== 保存与梯度配置 ====================
    save_every_n_epochs=5,  # 定期保存间隔
    grad_clip_max_norm=1.0,  # 梯度裁剪
    grad_accum_steps=4,  # 梯度累积步数
    # ==================== 学习率调度器 ====================
    scheduler_type="cosine_warm",  # CosineAnnealingWarmRestarts（基线为 "plateau"）
    scheduler_factor=0.6,  # 学习率衰减比例（仅 plateau 模式使用）
    scheduler_patience=12,  # 学习率调度耐心（仅 plateau 模式使用）
    scheduler_min_lr=3e-6,  # 最小学习率
    cosine_T0=30,  # 初始重启周期30个epoch
    cosine_T_mult=2,  # 每次重启周期翻倍（30→60→120→...）
    # ==================== 学习率 Warmup ====================
    warmup_epochs=0,  # 前N轮线性预热（0=关闭）; E03实验证明warmup有害，默认关闭
    warmup_start_lr=1e-5,  # 预热起始学习率
    # ==================== 早停配置 ====================
    early_stopping_enabled=True,  # 启用早停
    early_stopping_patience=150,  # 早停耐心
    early_stopping_metric="val_acc",  # 早停监控指标
    early_stopping_min_delta=0.0,  # 最小改进阈值
    # ==================== 采样与加载配置 ====================
    use_weighted_sampler=False,  # 类别均衡采样（实验证明无效，保持关闭）
    sampler_power=0.7,  # 采样权重指数
    dataloader_num_workers=0,  # DataLoader worker
    # ==================== EMA 配置 ====================
    use_ema=True,  # 启用 EMA 提升泛化稳定性
    ema_decay=0.999,  # EMA 衰减系数
    ema_start_epoch=6,  # 前几轮热身后启用 EMA
    # ==================== 数据增强配置 ====================
    mixup_alpha=0.0,  # Mixup 关闭（实验证明对小数据集有害）
)


EVALUATION = EvaluationConfig(
    # ==================== 模型路径配置 ====================
    model_path=os.path.join(_PROJECT_ROOT, "src", "checkpoints", "best_model.pth"),  # 默认评估模型
    ensemble_model_paths=(),  # 集成评估模型列表（为空则使用单模型）
    # ==================== TTA 配置 ====================
    use_tta_hflip=False,  # 关闭 TTA，保持评估稳定性
    # ==================== 评估参数 ====================
    verbose_report=True,  # 输出详细分类报告
    save_report=True,  # 保存评估报告到日志
)


INFERENCE = InferenceConfig(
    camera_index=0,  # 摄像头编号
    camera_width=640,  # 摄像头宽度
    camera_height=480,  # 摄像头高度
    camera_fps=30,  # 摄像头帧率
    inference_interval=3,  # 推理间隔
)


UI = UIConfig(
    chinese_font_paths=(
        r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    ),
    font_size=32,  # 主文本字号
    font_small_size=26,  # 次级文本字号
    exit_button_text="退出",  # 退出按钮文案
    exit_button_size=(90, 40),  # 退出按钮尺寸
    exit_button_margin=(12, 12),  # 退出按钮边距
    skeleton_button_text_on="隐藏骨骼",  # 骨骼按钮文案（当前显示）
    skeleton_button_text_off="显示骨骼",  # 骨骼按钮文案（当前隐藏）
    skeleton_button_size=(110, 40),  # 骨骼按钮尺寸
    skeleton_button_gap=10,  # 按钮间距
    skeleton_point_radius=3,  # 关键点半径
    skeleton_point_color=(0, 255, 0),  # 关键点颜色
    skeleton_line_color=(255, 255, 0),  # 骨架线颜色
    skeleton_line_thickness=1,  # 骨架线宽
)


# 确保模型目录存在，避免训练保存时报错
os.makedirs(PATHS.model_save_dir, exist_ok=True)
