"""
WLASL 数据集预处理脚本
将手语视频处理为 HDF5 格式，提取 135 个骨骼关键点用于深度学习模型训练。

使用 MediaPipe Tasks API 提取全身姿态（Pose 33 + Hands 21*2 + Face 478 -> 映射到 135 点）

输出格式：
- WLASL_train.hdf5
- WLASL_val.hdf5  
- WLASL_test.hdf5

每个 HDF5 文件结构：
- /{video_id}/data: float64 (T, 2, 135) - 归一化后的关键点坐标
- /{video_id}/label: string - gloss 标签
- /{video_id}/video_name: string - 原始视频路径
- /{video_id}/width: int64 - 视频宽度
- /{video_id}/height: int64 - 视频高度

使用方法：
    uv run python src\\data_process\\preprocess_wlasl.py                     # 处理完整数据集（WLASL2000+）
    uv run python src\\data_process\\preprocess_wlasl.py --subset 100        # 处理 WLASL100 子集
    uv run python src\\data_process\\preprocess_wlasl.py --subset 300        # 处理 WLASL300 子集
    uv run python src\\data_process\\preprocess_wlasl.py --subset 2000       # 处理 WLASL2000 子集
    uv run python src\\data_process\\preprocess_wlasl.py --limit 50          # 只处理前 50 个视频（用于测试）
    uv run python src\\data_process\\preprocess_wlasl.py --json my_data.json # 使用自定义 JSON 文件
"""

import argparse
import json
import os
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cv2
import h5py
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 抑制 MediaPipe 的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============== 用户配置参数（可直接修改）==============
# 数据集规模：设置为 100, 300, 2000 或 None
# - 100: 使用 WLASL100 子集 (100 个类别)
# - 300: 使用 WLASL300 子集 (300 个类别)
# - 2000: 使用 WLASL2000 子集 (2000 个类别)
# - None: 使用完整数据集 (WLASL_v0.3.json)
DEFAULT_SUBSET = 100  # 修改此值选择数据集规模

# 输入输出路径配置
DEFAULT_JSON_PATH = "dataset\\raw\\WLASL_v0.3.json"
DEFAULT_VIDEO_DIR = "dataset\\raw\\video"
DEFAULT_OUTPUT_DIR = "output" #本项目中为processed
DEFAULT_OUTPUT_PREFIX = "WLASL"

# 处理数量限制：设置为 None 处理全部，或设置具体数字限制处理数量
DEFAULT_LIMIT = None  # 例如设置为 100 只处理前 100 个视频

# ============== 以下配置通常无需修改 ==============
# 数据集规模映射 (subset -> JSON 文件)
SUBSET_JSON_MAP = {
    100: "dataset\\raw\\nslt_100.json",
    300: "dataset\\raw\\nslt_300.json",
    2000: "dataset\\raw\\nslt_2000.json",
}

# 目标关键点数量 (WLASL 官方格式)
TARGET_KEYPOINTS = 135

# 模型文件路径
MODEL_DIR = "src\\mediapipe_models"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

# 模型下载 URL
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model(url: str, path: str) -> None:
    """
    检查并下载模型文件
    
    如果本地已存在模型文件则跳过下载
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        print(f"  ✓ 已存在: {os.path.basename(path)}")
    else:
        print(f"  ↓ 下载中: {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"  ✓ 下载完成: {os.path.basename(path)}")


class KeypointExtractor:
    """使用 MediaPipe Tasks API 提取人体骨骼关键点"""
    
    def __init__(self):
        """初始化 MediaPipe 模型"""
        # 下载模型
        print("  检查和下载模型文件...")
        download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
        download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
        
        # 初始化 Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        
        # 初始化 Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        # 初始化 Face Landmarker
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        
        print("  模型加载完成!")
        
    def close(self):
        """释放资源"""
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()
        
    def extract_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从单帧图像中提取关键点
        
        Args:
            frame: BGR 格式的图像 (H, W, 3)
            
        Returns:
            关键点数组 (2, 135) 或 None（如果检测失败）
        """
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 运行检测
        try:
            pose_result = self.pose_detector.detect(mp_image)
            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_detector.detect(mp_image)
        except Exception:
            return None
        
        # 提取关键点并映射到 135 点格式
        keypoints = self._map_to_135_keypoints(pose_result, hand_result, face_result)
        
        return keypoints
    
    def _map_to_135_keypoints(self, pose_result, hand_result, face_result) -> np.ndarray:
        """
        将 MediaPipe 结果映射到 135 个关键点
        
        WLASL 官方 135 点格式：
        - Body: 25 点 (OpenPose Body 25 格式)
        - Left Hand: 21 点
        - Right Hand: 21 点
        - Face: 68 点 (68-point facial landmarks)
        
        Returns:
            形状为 (2, 135) 的数组，包含 (x, y) 坐标
        """
        keypoints = np.zeros((2, TARGET_KEYPOINTS), dtype=np.float64)
        
        # 1. 提取 Body 关键点 (25 点)
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose = pose_result.pose_landmarks[0]
            # OpenPose Body 25 索引到 MediaPipe Pose 索引的映射
            body_mapping = [
                0,   # 0: Nose
                -1,  # 1: Neck (不存在，用 11+12 平均)
                12,  # 2: RShoulder
                14,  # 3: RElbow
                16,  # 4: RWrist
                11,  # 5: LShoulder
                13,  # 6: LElbow
                15,  # 7: LWrist
                -1,  # 8: MidHip (23+24 平均)
                24,  # 9: RHip
                26,  # 10: RKnee
                28,  # 11: RAnkle
                23,  # 12: LHip
                25,  # 13: LKnee
                27,  # 14: LAnkle
                5,   # 15: REye
                2,   # 16: LEye
                8,   # 17: REar
                7,   # 18: LEar
                32,  # 19: LBigToe
                31,  # 20: LSmallToe
                29,  # 21: LHeel
                31,  # 22: RBigToe
                32,  # 23: RSmallToe
                30,  # 24: RHeel
            ]
            
            for i, mp_idx in enumerate(body_mapping):
                if mp_idx >= 0:
                    keypoints[0, i] = pose[mp_idx].x
                    keypoints[1, i] = pose[mp_idx].y
                elif i == 1:  # Neck
                    keypoints[0, i] = (pose[11].x + pose[12].x) / 2
                    keypoints[1, i] = (pose[11].y + pose[12].y) / 2
                elif i == 8:  # MidHip
                    keypoints[0, i] = (pose[23].x + pose[24].x) / 2
                    keypoints[1, i] = (pose[23].y + pose[24].y) / 2
        
        # 2. 提取手部关键点 (21*2 点, 索引 25-66)
        if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_result.hand_landmarks, hand_result.handedness)
            ):
                # 判断是左手还是右手
                hand_label = handedness[0].category_name.lower()
                
                if hand_label == 'left':
                    offset = 25  # 左手: 索引 25-45
                else:
                    offset = 46  # 右手: 索引 46-66
                
                for i, lm in enumerate(hand_landmarks):
                    keypoints[0, offset + i] = lm.x
                    keypoints[1, offset + i] = lm.y
        
        # 3. 提取面部关键点 (68 点, 索引 67-134)
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face = face_result.face_landmarks[0]
            # 68 面部关键点映射 (基于 dlib 68 点标准)
            face_68_indices = [
                # 面部轮廓 (17 点)
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                # 左眉 (5 点)
                70, 63, 105, 66, 107,
                # 右眉 (5 点)
                336, 296, 334, 293, 300,
                # 鼻梁 (4 点)
                168, 6, 197, 195,
                # 鼻尖下方 (5 点)
                5, 4, 1, 19, 94,
                # 左眼 (6 点)
                33, 160, 158, 133, 153, 144,
                # 右眼 (6 点)
                362, 385, 387, 263, 373, 380,
                # 外唇 (12 点)
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308,
                # 内唇 (8 点)
                78, 191, 80, 81, 82, 13, 312, 311,
            ]
            
            for i, face_idx in enumerate(face_68_indices):
                if i < 68:
                    keypoints[0, 67 + i] = face[face_idx].x
                    keypoints[1, 67 + i] = face[face_idx].y
        
        return keypoints


def load_wlasl_json(json_path: str) -> dict:
    """
    加载 WLASL JSON 元数据文件并构建视频信息字典
    
    支持两种格式：
    1. WLASL_v0.3.json 格式：列表形式，包含完整的 gloss 和 instance 信息
    2. nslt_*.json 格式：字典形式 {video_id: {"subset": str, "action": [label_id, ...]}}
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        字典：{video_id: {"gloss": str, "split": str, ...}}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_info = {}
    
    # 判断 JSON 格式
    if isinstance(data, list):
        # WLASL_v0.3.json 格式（列表形式）
        for entry in data:
            gloss = entry['gloss']
            for instance in entry['instances']:
                video_id = instance['video_id']
                split = instance.get('split', 'train')
                
                video_info[video_id] = {
                    'gloss': gloss,
                    'split': split,
                    'bbox': instance.get('bbox'),
                    'fps': instance.get('fps'),
                    'frame_start': instance.get('frame_start', 1),
                    'frame_end': instance.get('frame_end', -1),
                    'variation_id': instance.get('variation_id'),
                    'signer_id': instance.get('signer_id'),
                }
    
    elif isinstance(data, dict):
        # nslt_*.json 格式（字典形式）
        # 需要从 WLASL_v0.3.json 获取 gloss 标签映射
        gloss_map = _build_gloss_map()
        
        for video_id, info in data.items():
            split = info.get('subset', 'train')
            # action 字段包含 [label_id, ...]，取第一个作为主标签
            action = info.get('action', [])
            label_id = action[0] if action else -1
            gloss = gloss_map.get(label_id, f"unknown_{label_id}")
            
            video_info[video_id] = {
                'gloss': gloss,
                'split': split,
                'label_id': label_id,
                'bbox': None,
                'fps': None,
                'frame_start': 1,
                'frame_end': -1,
            }
    
    return video_info


def _build_gloss_map() -> dict:
    """
    从 WLASL_v0.3.json 构建 label_id -> gloss 的映射
    
    Returns:
        字典：{label_id: gloss}
    """
    gloss_map = {}
    
    wlasl_json = os.path.join(os.path.dirname(DEFAULT_JSON_PATH), "WLASL_v0.3.json")
    if not os.path.exists(wlasl_json):
        print(f"  警告: 找不到 {wlasl_json}，无法获取 gloss 标签")
        return gloss_map
    
    with open(wlasl_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 在 WLASL_v0.3.json 中，gloss 按索引顺序排列
    for idx, entry in enumerate(data):
        gloss_map[idx] = entry['gloss']
    
    return gloss_map


def process_video(
    video_path: str,
    extractor: KeypointExtractor,
    frame_start: int = 1,
    frame_end: int = -1,
) -> tuple[Optional[np.ndarray], int, int]:
    """
    处理单个视频，提取所有帧的关键点
    
    Args:
        video_path: 视频文件路径
        extractor: 关键点提取器
        frame_start: 起始帧（1-indexed）
        frame_end: 结束帧（-1 表示到最后）
        
    Returns:
        (keypoints_array, width, height) 或 (None, 0, 0) 如果失败
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, 0, 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 处理帧范围
    start_idx = max(0, frame_start - 1)  # 转换为 0-indexed
    end_idx = total_frames if frame_end == -1 else min(frame_end, total_frames)
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    all_keypoints = []
    frame_idx = start_idx
    
    while frame_idx < end_idx:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 提取关键点
        keypoints = extractor.extract_frame(frame)
        if keypoints is not None:
            all_keypoints.append(keypoints)
        else:
            # 如果某帧检测失败，使用零填充
            all_keypoints.append(np.zeros((2, TARGET_KEYPOINTS), dtype=np.float64))
        
        frame_idx += 1
    
    cap.release()
    
    if len(all_keypoints) == 0:
        return None, width, height
    
    # 堆叠为 (T, 2, 135) 形状
    keypoints_array = np.stack(all_keypoints, axis=0)
    
    return keypoints_array, width, height


def create_hdf5_file(
    output_path: str,
    video_data: dict,
    split: str = "train",
    subset: Optional[int] = None,
) -> None:
    """
    创建 HDF5 文件（匹配官方 WLASL 数据格式）
    
    Args:
        output_path: 输出文件路径
        video_data: {video_id: {"data": array, "label": str, "video_name": str, "width": int, "height": int}}
        split: 数据划分 (train/val/test)
        subset: 数据集规模 (100/300/2000)，用于生成 video_name 路径
    """
    with h5py.File(output_path, 'w') as f:
        # 使用数字索引作为 Group 名称（匹配官方格式）
        for idx, (video_id, data) in enumerate(
            tqdm(video_data.items(), desc=f"Writing {os.path.basename(output_path)}")
        ):
            # 创建 Group，使用数字索引
            grp = f.create_group(str(idx))
            
            # 存储数据
            grp.create_dataset('data', data=data['data'], dtype='float64')
            
            # 存储标签 (字符串需要特殊处理)
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset('label', data=data['label'], dtype=dt)
            
            # 生成官方格式的 video_name 路径
            # 官方格式: rgb/WLASL{subset}/{split}/{video_id}.mp4
            if subset:
                video_name = f"rgb/WLASL{subset}/{split}/{video_id}.mp4"
            else:
                video_name = data['video_name']
            grp.create_dataset('video_name', data=video_name, dtype=dt)
            
            # 存储视频尺寸
            grp.create_dataset('width', data=data['width'], dtype='int64')
            grp.create_dataset('height', data=data['height'], dtype='int64')


def create_maplabels_json(
    output_path: str,
    all_data: dict,
) -> None:
    """
    创建 label 映射文件（匹配官方格式）
    
    官方格式：
    {
        "id_to_label": {
            "book": 0,
            "drink": 1,
            ...
        }
    }
    
    Args:
        output_path: 输出 JSON 文件路径
        all_data: 所有 split 的数据 {split: {video_id: {...}}}
    """
    # 收集所有唯一的 label
    labels = set()
    for split_data in all_data.values():
        for video_data in split_data.values():
            labels.add(video_data['label'])
    
    # 按字母顺序排序并分配 ID
    sorted_labels = sorted(labels)
    id_to_label = {label: idx for idx, label in enumerate(sorted_labels)}
    
    # 写入 JSON 文件
    maplabels = {"id_to_label": id_to_label}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(maplabels, f, indent=4, ensure_ascii=False)
    
    print(f"  已生成标签映射文件: {output_path} ({len(id_to_label)} 个类别)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='WLASL 数据集预处理脚本 - 提取骨骼关键点并保存为 HDF5 格式'
    )
    parser.add_argument(
        '--json', '-j',
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f'JSON 元数据文件路径 (默认: {DEFAULT_JSON_PATH})'
    )
    parser.add_argument(
        '--video-dir', '-v',
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f'视频文件目录 (默认: {DEFAULT_VIDEO_DIR})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f'输出文件名前缀 (默认: {DEFAULT_OUTPUT_PREFIX})'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='限制处理的视频数量 (用于测试)'
    )
    parser.add_argument(
        '--subset', '-s',
        type=int,
        choices=[100, 300, 2000],
        default=None,
        help='选择数据集规模: 100, 300, 或 2000 (自动选择对应的 JSON 文件)'
    )
    
    args = parser.parse_args()
    
    # 优先使用代码中的配置，其次是命令行参数
    # 如果代码中设置了 DEFAULT_SUBSET，优先使用
    subset = args.subset if args.subset is not None else DEFAULT_SUBSET
    
    if subset is not None:
        if subset in SUBSET_JSON_MAP:
            args.json = SUBSET_JSON_MAP[subset]
            args.prefix = f"WLASL{subset}"
            args.subset = subset
        else:
            parser.error(f"不支持的数据集规模: {subset}")
    
    # 如果代码中设置了 DEFAULT_LIMIT，优先使用
    if args.limit is None and DEFAULT_LIMIT is not None:
        args.limit = DEFAULT_LIMIT
    
    return args


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("WLASL 数据集预处理脚本")
    print("=" * 60)
    print(f"  JSON 文件:   {args.json}")
    print(f"  视频目录:    {args.video_dir}")
    print(f"  输出目录:    {args.output_dir}")
    print(f"  输出前缀:    {args.prefix}")
    if args.limit:
        print(f"  处理限制:    {args.limit} 个视频")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载 JSON 元数据
    print(f"\n[1/5] 加载 JSON 元数据: {args.json}")
    if not os.path.exists(args.json):
        print(f"  错误: JSON 文件不存在: {args.json}")
        sys.exit(1)
    video_info = load_wlasl_json(args.json)
    print(f"  找到 {len(video_info)} 个视频条目")
    
    # 2. 扫描可用视频文件
    print(f"\n[2/5] 扫描视频目录: {args.video_dir}")
    if not os.path.exists(args.video_dir):
        print(f"  错误: 视频目录不存在: {args.video_dir}")
        sys.exit(1)
    video_dir = Path(args.video_dir)
    available_videos = set()
    for vf in video_dir.glob("*.mp4"):
        video_id = vf.stem
        available_videos.add(video_id)
    print(f"  找到 {len(available_videos)} 个视频文件")
    
    # 统计各个 split 的数量
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for video_id, info in video_info.items():
        if video_id in available_videos:
            split = info['split']
            if split in split_counts:
                split_counts[split] += 1
    print(f"  分布: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    
    # 3. 初始化关键点提取器
    print("\n[3/5] 初始化 MediaPipe 模型...")
    extractor = KeypointExtractor()
    
    # 4. 处理视频并生成 HDF5
    print("\n[4/5] 处理视频并提取关键点...")
    
    # 按 split 分组
    split_data = {'train': {}, 'val': {}, 'test': {}}
    
    # 处理每个可用的视频
    processed_count = 0
    failed_count = 0
    
    # 获取要处理的视频列表
    videos_to_process = [(vid, video_info[vid]) for vid in available_videos if vid in video_info]
    
    # 应用限制
    if args.limit and args.limit < len(videos_to_process):
        videos_to_process = videos_to_process[:args.limit]
        print(f"  已限制为前 {args.limit} 个视频")
    
    for video_id, info in tqdm(videos_to_process, desc="Processing videos"):
        video_path = str(video_dir / f"{video_id}.mp4")
        
        try:
            # 提取关键点
            keypoints, width, height = process_video(
                video_path,
                extractor,
                frame_start=info.get('frame_start', 1),
                frame_end=info.get('frame_end', -1),
            )
            
            if keypoints is None or keypoints.shape[0] == 0:
                failed_count += 1
                continue
            
            # 确定 split
            split = info.get('split', 'train')
            if split not in split_data:
                split = 'train'  # 未知 split 归入 train
            
            # 存储数据
            split_data[split][video_id] = {
                'data': keypoints,
                'label': info['gloss'],
                'video_name': f"{video_id}.mp4",
                'width': width,
                'height': height,
            }
            
            processed_count += 1
            
        except Exception as e:
            tqdm.write(f"  警告: 处理视频 {video_id} 时出错: {e}")
            failed_count += 1
            continue
    
    # 释放资源
    extractor.close()
    
    # 5. 写入 HDF5 文件
    print("\n[5/5] 写入 HDF5 文件...")
    
    # split 名称映射（匹配官方格式）
    split_name_map = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
    
    for split, data in split_data.items():
        if len(data) > 0:
            # 官方文件命名格式: WLASL100_135-Train.hdf5
            split_name = split_name_map.get(split, split.capitalize())
            output_path = os.path.join(args.output_dir, f"{args.prefix}_135-{split_name}.hdf5")
            create_hdf5_file(output_path, data, split=split, subset=args.subset)
            print(f"  {split}: 已保存 {len(data)} 个样本到 {output_path}")
    
    # 生成 maplabels.json 文件
    if args.subset:
        maplabels_path = os.path.join(args.output_dir, f"wlasl_{args.subset}_maplabels.json")
    else:
        maplabels_path = os.path.join(args.output_dir, "wlasl_maplabels.json")
    create_maplabels_json(maplabels_path, split_data)
    
    # 6. 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"  成功处理: {processed_count} 个视频")
    print(f"  处理失败: {failed_count} 个视频")
    print(f"  输出目录: {args.output_dir}/")
    

if __name__ == "__main__":
    main()
