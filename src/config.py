import os

# --- 数据路径配置 ---
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据集规模 (例如: 100, 300, 2000)
DATASET_SCALE = 100
# 已处理数据的存储目录
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed", f"WLASL{DATASET_SCALE}")
# 标签映射文件路径 (JSON格式)
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, f"wlasl_{DATASET_SCALE}_maplabels.json")

# 训练、验证和测试集的 HDF5 文件路径
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, f"WLASL{DATASET_SCALE}_135-Train.hdf5")
VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, f"WLASL{DATASET_SCALE}_135-Val.hdf5")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, f"WLASL{DATASET_SCALE}_135-Test.hdf5")

# 模型检查点保存目录
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "src", "checkpoints")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 测试/评估时使用的模型路径
TEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# --- 数据处理配置 ---
# 序列最大帧数 (超出重采样，不足补零)
MAX_FRAMES = 90
# 关键点维度 (包含速度特征: x, y, dx, dy)
LANDMARK_DIM = 4
# 关键点数量
NUM_LANDMARKS = 135
# 模型输入维度 (135个关键点 * 每个点4维 = 540)
INPUT_SIZE = LANDMARK_DIM * NUM_LANDMARKS
# 类别数量，对应数据集规模
NUM_CLASSES = DATASET_SCALE

# --- 预处理脚本相关路径与参数 ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
VIDEOID_WORD_DIR = os.path.join(PROJECT_ROOT, "dataset", "videoId_word")
DEFAULT_JSON_PATH = os.path.join(RAW_DATA_DIR, "WLASL_v0.3.json")
# WLASL_v0.3.json 用于获取 gloss 标签映射
WLASL_GLOSS_JSON = os.path.join(VIDEOID_WORD_DIR, "WLASL_v0.3.json")
DEFAULT_VIDEO_DIR = os.path.join(RAW_DATA_DIR, f"WLASL{DATASET_SCALE}")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
DEFAULT_OUTPUT_PREFIX = "WLASL"
DEFAULT_LIMIT = None
# 多进程处理配置 (设为 1 禁用多进程，None 自动检测 CPU 核心数)
NUM_WORKERS = None

# 子集 JSON 映射
SUBSET_JSON_MAP = {
	100: os.path.join(RAW_DATA_DIR, "WLASL100", "nslt_100.json"),
	300: os.path.join(RAW_DATA_DIR, "WLASL300", "nslt_300.json"),
	1000: os.path.join(RAW_DATA_DIR, "WLASL1000", "nslt_1000.json"),
	2000: os.path.join(RAW_DATA_DIR, "WLASL2000", "nslt_2000.json"),
}

# Mediapipe 模型与阈值
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "mediapipe_models")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

POSE_MIN_DET_CONF = 0.5
POSE_MIN_PRESENCE_CONF = 0.5
POSE_MIN_TRACK_CONF = 0.5

HAND_MIN_DET_CONF = 0.5
HAND_MIN_PRESENCE_CONF = 0.5
HAND_MIN_TRACK_CONF = 0.5

FACE_MIN_DET_CONF = 0.5
FACE_MIN_PRESENCE_CONF = 0.5
FACE_MIN_TRACK_CONF = 0.5

# --- 模型超参数 ---
# 隐藏层维度
HIDDEN_SIZE = 128
# LSTM 层数
NUM_LAYERS = 2
# 是否使用双向 LSTM
BIDIRECTIONAL = True
# 随机丢弃率 (Dropout)
DROPOUT = 0.4
# 标签平滑 (Label Smoothing)
LABEL_SMOOTHING = 0.1

# --- Attention 机制配置 ---
# 是否启用 Attention 机制
USE_ATTENTION = True
# Attention 隐藏层维度 (用于计算注意力权重)
ATTENTION_DIM = 32

# --- 数据增强配置（适度增强，防止过拟合但不过度破坏分布） ---
# 旋转范围 (度)
AUG_ROTATION_RANGE = 20
# 缩放范围
AUG_SCALE_MIN = 0.85
AUG_SCALE_MAX = 1.15
# 平移范围 (坐标归一化到 0~1)
AUG_TRANSLATE = 0.12
# 高斯噪声标准差
AUG_NOISE_STD = 0.004
# 水平翻转概率
AUG_HFLIP_PROB = 0.30

# --- 训练超参数 ---
# 批处理大小
BATCH_SIZE = 4
# 学习率
LEARNING_RATE = 1e-3
# L2 正则化 (权重衰减)
WEIGHT_DECAY = 1e-3
# 训练轮数
NUM_EPOCHS = 500
# 训练设备 (程序中会自动检查 GPU 可用性)
DEVICE = 'cuda'

# 摄像头索引（实时推理使用）
CAMERA_INDEX = 0
# 摄像头分辨率（降低可提高帧率）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 摄像头目标帧率
CAMERA_FPS = 60
# 推理间隔（每隔多少帧进行一次模型推理，1=每帧都推理，2=隔一帧推理）
INFERENCE_INTERVAL = 1

# --- UI/显示配置 ---
# 中文字体候选路径（按顺序尝试找到可用字体）
CHINESE_FONT_PATHS = [
	r"C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
	r"C:\\Windows\\Fonts\\simhei.ttf", # 黑体
]
# 叠加文本字号
UI_FONT_SIZE = 32
UI_FONT_SMALL_SIZE = 26
# 退出按钮文字与尺寸
EXIT_BUTTON_TEXT = "退出"
EXIT_BUTTON_SIZE = (90, 40)   # (width, height)
EXIT_BUTTON_MARGIN = (12, 12) # (right_margin, top_margin)

# 骨骼显示切换按钮配置
SKELETON_BUTTON_TEXT_ON = "隐藏骨骼"
SKELETON_BUTTON_TEXT_OFF = "显示骨骼"
SKELETON_BUTTON_SIZE = (110, 40)  # (width, height)
SKELETON_BUTTON_GAP = 10  # 与退出按钮的间距

# 骨骼绘制样式
SKELETON_POINT_RADIUS = 3
SKELETON_POINT_COLOR = (0, 255, 0)   # BGR: 绿色
SKELETON_LINE_COLOR = (255, 255, 0)  # BGR: 青色
SKELETON_LINE_THICKNESS = 1
