#全局配置文件，存放全局配置信息和参数，
#函数中绝对不允许出现写死参数的情况，只能通过调用config.py来传递参数

import os
import multiprocessing



# ================= 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集根目录
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset')

# 原始视频存放路径 (目录结构为 dataset/video/{split}/{translator}/xxx.mp4)
RAW_VIDEOS_PATH = os.path.join(DATASET_PATH, 'video')
# 标签文件存放路径 (目录结构为 dataset/label/{split}.csv)
RAW_LABELS_PATH = os.path.join(DATASET_PATH, 'label')

# 数据集划分 (train/test/dev)
SPLITS = ['train', 'test', 'dev']

# 翻译者列表 (A-L)
TRANSLATORS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

# 特征数据输出路径 (保持划分结构: dataset/processed/videos/{split}/{translator}/xxx.npy)
PROCESSED_DATA_PATH = os.path.join(DATASET_PATH, 'processed', 'videos')

# 处理后的标签文件输出路径
PROCESSED_LABELS_PATH = os.path.join(DATASET_PATH, 'processed', 'labels')

# ================= MediaPipe 配置 =================
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5
# 模型复杂度: 0, 1, 2. 1是平衡，2精度最高但慢。离线处理建议 1 或 2
MP_MODEL_COMPLEXITY = 1 

# ================= 硬件与性能 =================
# 自动检测CPU核心数，保留2个核给系统，防止死机
# Ryzen 7 8845H 有 16 个逻辑线程，这里建议设为 12-14
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# 是否使用多进程预处理数据
USE_MULTIPROCESSING = True

# 多进程处理的批次大小 (每个进程一次处理的任务数)
CHUNK_SIZE = 1

# 数据处理数量限制 (None 或 'all' 表示处理所有，整数表示只处理前 N 个)
# 用于快速测试代码逻辑
DATA_LIMIT = 1

# ================= 数据维度 =================
# 关键点数量 (Holistic 模式)
# 左手 21 + 右手 21 + 姿态 33 = 75 个点
# 输出特征向量: Pose(33*3) + Left Hand(21*3) + Right Hand(21*3) = 225 维

# 单手关键点数量
HAND_LANDMARKS_NUM = 21
# 姿态关键点数量 (MediaPipe Pose 完整输出)
POSE_LANDMARKS_NUM = 33
# 每个关键点的坐标维度 (x, y, z)
LANDMARK_DIM = 3

# 左手特征维度: 21 * 3 = 63
LEFT_HAND_FEATURE_DIM = HAND_LANDMARKS_NUM * LANDMARK_DIM
# 右手特征维度: 21 * 3 = 63
RIGHT_HAND_FEATURE_DIM = HAND_LANDMARKS_NUM * LANDMARK_DIM
# 姿态特征维度: 33 * 3 = 99
POSE_FEATURE_DIM = POSE_LANDMARKS_NUM * LANDMARK_DIM

# 总特征维度: 左手(63) + 右手(63) + 姿态(99) = 225
TOTAL_FEATURE_DIM = LEFT_HAND_FEATURE_DIM + RIGHT_HAND_FEATURE_DIM + POSE_FEATURE_DIM

# ================= 视频处理配置 =================
# 支持的视频格式
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# 是否启用断点续传 (跳过已处理的文件)
ENABLE_RESUME = True

# ================= 日志配置 =================
# 是否保存错误日志
SAVE_ERROR_LOG = True
ERROR_LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'preprocessing_errors.log')

# ================= 模型训练配置 =================
# 训练超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # 初始学习率
NUM_EPOCHS = 50  # 训练轮数
WEIGHT_DECAY = 1e-5  # L2正则化系数

# 模型架构参数
HIDDEN_DIM = 256  # LSTM 隐藏层维度
NUM_LAYERS = 2  # LSTM 层数
DROPOUT = 0.5  # Dropout 概率

# 学习率调度器配置
LR_SCHEDULER_FACTOR = 0.5  # LR衰减因子
LR_SCHEDULER_PATIENCE = 5  # 验证集Loss不下降的容忍轮数

# 模型保存路径
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'src', 'checkpoints')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
VOCAB_PATH = os.path.join(MODEL_SAVE_DIR, 'vocab.json')

# DataLoader 配置
NUM_DATALOADER_WORKERS = 2  # DataLoader 工作线程数

# ================= 实时推理配置 =================
# 滑动窗口配置
SLIDING_WINDOW_SIZE = 30  # 窗口大小（帧数）
PREDICTION_THRESHOLD = 0.8  # 预测置信度阈值
DEBOUNCE_FRAMES = 5  # 防抖动：连续N帧才确认
COOLDOWN_FRAMES = 15  # 冷却：识别一个词后的等待帧数

# 摄像头配置
CAMERA_INDEX = 0  # 默认摄像头索引
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30