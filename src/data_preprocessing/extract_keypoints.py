"""
关键点提取模块

使用 MediaPipe Holistic 从视频中提取:
- 左手关键点 (21 points)
- 右手关键点 (21 points)  
- 姿态关键点 (33 points)

并进行坐标归一化处理
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os

# 将 src 目录添加到 path 中以支持导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

# 使用相对导入或绝对导入 (解决多进程环境下的导入问题)
try:
    from data_preprocessing.normalize import Normalizer
except ImportError:
    # 作为包内模块直接运行时的 fallback
    from normalize import Normalizer


def _has_chinese_chars(path: str) -> bool:
    """检查路径是否包含中文字符"""
    for char in path:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


class KeypointExtractor:
    """关键点提取器"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        
    def get_model(self):
        """返回一个新的 Holistic 实例"""
        return self.mp_holistic.Holistic(
            min_detection_confidence=cfg.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MP_TRACKING_CONFIDENCE,
            model_complexity=cfg.MP_MODEL_COMPLEXITY
        )
        
    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        从视频中提取并归一化特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            np.ndarray: 形状为 (Frames, Features) 的特征数组
                        Features = 左手(63) + 右手(63) + 姿态(99) = 225
        """
        # Windows 中文路径兼容处理
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # 针对中文路径给出更明确的提示
            if _has_chinese_chars(video_path):
                raise ValueError(
                    f"无法打开视频文件: {video_path}\n"
                    f"[提示] 路径包含中文字符，Windows 下 OpenCV 可能无法正确读取。"
                    f"建议将视频移动到不含中文的路径下。"
                )
            else:
                raise ValueError(f"无法打开视频文件: {video_path}")
            
        frames_data = []
        
        with self.get_model() as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 1. 预处理
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 2. 推理
                results = holistic.process(image)
                
                # 3. 归一化处理 (获取左手、右手、姿态三个特征)
                lh, rh, pose = Normalizer.process(
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                    results.pose_landmarks,
                    cfg.HAND_LANDMARKS_NUM,
                    cfg.POSE_LANDMARKS_NUM,
                    cfg.LANDMARK_DIM
                )
                
                # 4. 拼接特征: 左手(63) + 右手(63) + 姿态(99) = 225
                frame_features = np.concatenate([lh, rh, pose])
                frames_data.append(frame_features)
                
        cap.release()
        
        if len(frames_data) == 0:
            return np.array([])
            
        return np.array(frames_data)


def extract_features(video_path: str) -> np.ndarray:
    """
    独立函数供外部调用
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        np.ndarray: 形状为 (Frames, 225) 的特征数组
    """
    extractor = KeypointExtractor()
    return extractor.extract_from_video(video_path)
