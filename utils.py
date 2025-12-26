import os
import cv2
import numpy as np
import mediapipe as mp
import config as cfg

# 初始化 MediaPipe Holistic
mp_holistic = mp.solutions.holistic  # type: ignore


def get_mediapipe_model():
    """
    工厂函数：返回一个新的 Holistic 实例
    注意：在多进程中，每个进程必须创建自己的实例，不能共享
    """
    return mp_holistic.Holistic(
        min_detection_confidence=cfg.MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=cfg.MP_TRACKING_CONFIDENCE,
        model_complexity=cfg.MP_MODEL_COMPLEXITY
    )


def get_shoulder_center(pose_landmarks):
    """
    计算肩膀中心坐标 (归一化的参考原点)
    MediaPipe Pose 索引: 11 (左肩), 12 (右肩)
    
    Args:
        pose_landmarks: MediaPipe Pose 结果
        
    Returns:
        np.array: [center_x, center_y] 或 None
    """
    if pose_landmarks is None:
        return None
    
    # 获取左肩 (11) 和 右肩 (12)
    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]
    
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    return np.array([center_x, center_y])


def normalize_landmarks(landmarks, center_point):
    """
    归一化核心逻辑：
    Relative_Point = Point - Shoulder_Center
    
    Args:
        landmarks: 手部关键点
        center_point: 归一化参考点 (肩膀中心)
        
    Returns:
        np.array: 归一化后的特征向量 (63维 = 21点 * 3坐标)
    """
    if landmarks is None or center_point is None:
        # 如果检测不到手，返回全 0 向量 (21个点 * 3维 = 63)
        return np.zeros(cfg.HAND_LANDMARKS_NUM * cfg.LANDMARK_DIM)
    
    flattened_data = []
    for res in landmarks.landmark:
        # 相对坐标计算
        rel_x = res.x - center_point[0]
        rel_y = res.y - center_point[1]
        rel_z = res.z  # z 轴通常保持相对深度即可
        
        flattened_data.extend([rel_x, rel_y, rel_z])
        
    return np.array(flattened_data)

def extract_features_from_video(video_path: str) -> np.ndarray:
    """
    处理单个视频的完整流程，提取手部关键点特征
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        np.ndarray: 形状为 (Frames, Features) 的特征数组
                    其中 Features = 126 (左手63 + 右手63)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames_data = []
    
    # 在函数内部初始化模型 (进程安全)
    with get_mediapipe_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 图像转 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 性能优化
            
            # 2. 推理
            results = holistic.process(image)
            
            # 3. 计算参考点 (肩膀中心)
            # 即使不做动作，只要有人，Pose通常都能检测到
            shoulder_center = get_shoulder_center(results.pose_landmarks)
            
            # 如果没检测到人(肩)，这帧可能无效，用上一帧或者全0填充
            # 这里简单处理：如果没有肩，就假设原点是 (0.5, 0.5) 或者跳过
            if shoulder_center is None:
                shoulder_center = np.array([0.5, 0.5])

            # 4. 提取并归一化左右手
            # 左手
            lh = normalize_landmarks(results.left_hand_landmarks, shoulder_center)
            # 右手
            rh = normalize_landmarks(results.right_hand_landmarks, shoulder_center)
            
            # 5. 拼接特征 (左手 + 右手) -> 维度 63 + 63 = 126
            # 你可以根据需要把 Pose 的关键点也加进来
            frame_features = np.concatenate([lh, rh])
            
            frames_data.append(frame_features)
            
    cap.release()
    
    if len(frames_data) == 0:
        return np.array([])
    
    return np.array(frames_data)


def load_processed_labels(split: str) -> list:
    """
    加载处理后的标签文件
    
    Args:
        split: 数据集划分 ('train', 'test', 'dev')
        
    Returns:
        list: 标签信息列表
    """
    import json
    import os
    
    label_file = os.path.join(cfg.PROCESSED_LABELS_PATH, f'{split}_labels.json')
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"处理后的标签文件不存在: {label_file}")
    
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_npy_feature(npy_path: str) -> np.ndarray:
    """
    加载单个 .npy 特征文件
    
    Args:
        npy_path: .npy 文件路径
        
    Returns:
        np.ndarray: 特征数组
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"特征文件不存在: {npy_path}")
    
    return np.load(npy_path)


def get_dataset_stats(split: str | None = None) -> dict:
    """
    获取数据集统计信息
    
    Args:
        split: 数据集划分，None 表示所有划分
        
    Returns:
        dict: 统计信息
    """
    
    splits = [split] if split else cfg.SPLITS
    stats = {}
    
    for s in splits:
        processed_path = os.path.join(cfg.PROCESSED_DATA_PATH, s)
        if not os.path.exists(processed_path):
            stats[s] = {'count': 0, 'translators': {}}
            continue
        
        translator_stats = {}
        total_count = 0
        
        for translator in cfg.TRANSLATORS:
            translator_path = os.path.join(processed_path, translator)
            if os.path.exists(translator_path):
                count = len([f for f in os.listdir(translator_path) if f.endswith('.npy')])
                translator_stats[translator] = count
                total_count += count
        
        stats[s] = {
            'count': total_count,
            'translators': translator_stats
        }
    
    return stats