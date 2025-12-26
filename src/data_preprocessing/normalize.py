"""
坐标归一化处理模块

按照论文逻辑：
x_new = (x - x_nose) / |x_left_shoulder - x_right_shoulder|
y_new = (y - y_nose) / |x_left_shoulder - x_right_shoulder|

原点: 鼻尖 (比肩膀更稳定，处于画面中心)
分母: 肩宽 (消除距离/缩放影响)
"""

import numpy as np


class Normalizer:
    """坐标归一化处理类"""
    
    # MediaPipe Pose 关键点索引
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    
    @classmethod
    def get_reference_points(cls, pose_landmarks):
        """
        获取参考点：鼻尖和双肩
        
        Returns:
            tuple: (nose, left_shoulder, right_shoulder) 或 (None, None, None)
        """
        if pose_landmarks is None:
            return None, None, None
            
        nose = pose_landmarks.landmark[cls.NOSE_IDX]
        left_shoulder = pose_landmarks.landmark[cls.LEFT_SHOULDER_IDX]
        right_shoulder = pose_landmarks.landmark[cls.RIGHT_SHOULDER_IDX]
        
        return nose, left_shoulder, right_shoulder

    @staticmethod
    def calculate_shoulder_width(left_shoulder, right_shoulder):
        """
        计算肩宽 (水平距离绝对值)
        
        Returns:
            float: 肩宽，若无效则返回 0
        """
        if left_shoulder is None or right_shoulder is None:
            return 0.0
        
        width = abs(left_shoulder.x - right_shoulder.x)
        # 防止除以0
        return width if width > 1e-6 else 1.0

    @staticmethod
    def normalize_hand_landmarks(landmarks, nose, shoulder_width, landmark_count, landmark_dim):
        """
        归一化手部关键点
        
        Args:
            landmarks: MediaPipe 手部关键点
            nose: 鼻尖关键点 (归一化原点)
            shoulder_width: 肩宽 (归一化分母)
            landmark_count: 关键点数量 (21)
            landmark_dim: 坐标维度 (3)
            
        Returns:
            np.array: 归一化后的特征向量
        """
        if landmarks is None or nose is None or shoulder_width <= 0:
            return np.zeros(landmark_count * landmark_dim)
            
        flattened_data = []
        for lm in landmarks.landmark:
            rel_x = (lm.x - nose.x) / shoulder_width
            rel_y = (lm.y - nose.y) / shoulder_width
            rel_z = lm.z  # Z轴保持相对深度
            
            flattened_data.extend([rel_x, rel_y, rel_z])
            
        return np.array(flattened_data)

    @staticmethod
    def normalize_pose_landmarks(pose_landmarks, nose, shoulder_width, landmark_count, landmark_dim):
        """
        归一化姿态关键点
        
        Args:
            pose_landmarks: MediaPipe Pose 关键点
            nose: 鼻尖关键点 (归一化原点)
            shoulder_width: 肩宽 (归一化分母)
            landmark_count: 关键点数量 (33)
            landmark_dim: 坐标维度 (3)
            
        Returns:
            np.array: 归一化后的特征向量
        """
        if pose_landmarks is None or nose is None or shoulder_width <= 0:
            return np.zeros(landmark_count * landmark_dim)
            
        flattened_data = []
        for lm in pose_landmarks.landmark:
            rel_x = (lm.x - nose.x) / shoulder_width
            rel_y = (lm.y - nose.y) / shoulder_width
            rel_z = lm.z  # Z轴保持相对深度
            
            flattened_data.extend([rel_x, rel_y, rel_z])
            
        return np.array(flattened_data)

    @classmethod
    def process(cls, lh_landmarks, rh_landmarks, pose_landmarks, 
                hand_landmarks_num, pose_landmarks_num, landmark_dim):
        """
        执行完整的归一化流程
        
        Args:
            lh_landmarks: 左手关键点
            rh_landmarks: 右手关键点
            pose_landmarks: 姿态关键点
            hand_landmarks_num: 单手关键点数量 (21)
            pose_landmarks_num: 姿态关键点数量 (33)
            landmark_dim: 坐标维度 (3)
            
        Returns:
            tuple: (left_hand, right_hand, pose) 归一化后的特征向量
        """
        # 获取参考点
        nose, left_shoulder, right_shoulder = cls.get_reference_points(pose_landmarks)
        
        # 计算肩宽
        shoulder_width = cls.calculate_shoulder_width(left_shoulder, right_shoulder)
        
        # 归一化左手
        lh = cls.normalize_hand_landmarks(
            lh_landmarks, nose, shoulder_width, 
            hand_landmarks_num, landmark_dim
        )
        
        # 归一化右手
        rh = cls.normalize_hand_landmarks(
            rh_landmarks, nose, shoulder_width, 
            hand_landmarks_num, landmark_dim
        )
        
        # 归一化姿态
        pose = cls.normalize_pose_landmarks(
            pose_landmarks, nose, shoulder_width,
            pose_landmarks_num, landmark_dim
        )
        
        return lh, rh, pose
