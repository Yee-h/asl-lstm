"""
实时手语识别推理模块 (管道B: 在线推理流水线)

功能：
1. 从摄像头获取实时视频流
2. 使用 MediaPipe 提取关键点特征（与训练时相同的归一化逻辑）
3. 维护固定长度的滑动窗口（30帧）
4. 使用训练好的 LSTM 模型进行实时识别
5. 实现防抖动和冷却机制，避免误判
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import json
from collections import deque
import time

import config as cfg
from data_preprocessing.extract_keypoints import KeypointExtractor
from model.model_lstm import CSL_BiLSTM


class RealtimeSignLanguageRecognizer:
    """实时手语识别器"""
    
    def __init__(self, model_path: str = None, vocab_path: str = None):
        """
        初始化识别器
        
        Args:
            model_path: 模型权重文件路径，默认使用 config 中的路径
            vocab_path: 词汇表文件路径，默认使用 config 中的路径
        """
        self.model_path = model_path or cfg.BEST_MODEL_PATH
        self.vocab_path = vocab_path or cfg.VOCAB_PATH
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")
        
        # 加载词汇表
        self._load_vocabulary()
        
        # 加载模型
        self._load_model()
        
        # 特征提取器
        self.extractor = KeypointExtractor()
        
        # 滑动窗口（存储最近 N 帧的特征）
        self.window_size = cfg.SLIDING_WINDOW_SIZE
        self.feature_window = deque(maxlen=self.window_size)
        
        # 状态机参数
        self.prediction_threshold = cfg.PREDICTION_THRESHOLD
        self.debounce_frames = cfg.DEBOUNCE_FRAMES
        self.cooldown_frames = cfg.COOLDOWN_FRAMES
        
        # 当前状态
        self.consecutive_predictions = []  # 连续预测的词汇
        self.cooldown_counter = 0  # 冷却计数器
        self.current_sentence = []  # 已识别的句子
        
        print(f"[INFO] 识别器初始化完成")
        print(f"[INFO] 窗口大小: {self.window_size} 帧")
        print(f"[INFO] 置信度阈值: {self.prediction_threshold}")
        print(f"[INFO] 防抖帧数: {self.debounce_frames}")
    
    def _load_vocabulary(self):
        """加载词汇表"""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"词汇表文件不存在: {self.vocab_path}")
        
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        # 创建索引到词汇的映射
        self.idx_to_label = {idx: word for word, idx in self.label_map.items()}
        print(f"[INFO] 加载词汇表: {len(self.label_map)} 个词汇")
    
    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model = CSL_BiLSTM(
            input_dim=cfg.TOTAL_FEATURE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=len(self.label_map),
            num_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"[INFO] 模型加载成功: {self.model_path}")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        处理单帧图像，提取特征
        
        Args:
            frame: BGR 格式的图像帧
        
        Returns:
            tuple: (特征向量, 是否成功提取)
        """
        # 转换为 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 使用 MediaPipe 处理
        with self.extractor.get_model() as holistic:
            results = holistic.process(image)
        
        # 归一化处理
        from data_preprocessing.normalize import Normalizer
        lh, rh, pose = Normalizer.process(
            results.left_hand_landmarks,
            results.right_hand_landmarks,
            results.pose_landmarks,
            cfg.HAND_LANDMARKS_NUM,
            cfg.POSE_LANDMARKS_NUM,
            cfg.LANDMARK_DIM
        )
        
        # 拼接特征
        features = np.concatenate([lh, rh, pose])
        
        # 检查是否成功提取（非全零）
        success = np.any(features != 0)
        
        return features, success
    
    def predict(self) -> tuple:
        """
        基于当前窗口进行预测
        
        Returns:
            tuple: (预测词汇, 置信度, 是否有效)
        """
        if len(self.feature_window) < self.window_size:
            return None, 0.0, False
        
        # 转换为 Tensor
        features = np.array(list(self.feature_window))  # (window_size, 225)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, window_size, 225)
        
        # 推理
        with torch.no_grad():
            lengths = torch.LongTensor([self.window_size])
            outputs = self.model(features_tensor, lengths)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        # 获取预测结果
        pred_idx = predicted_idx.item()
        conf = confidence.item()
        pred_word = self.idx_to_label[pred_idx]
        
        # 判断是否有效（置信度是否超过阈值）
        valid = conf >= self.prediction_threshold
        
        return pred_word, conf, valid
    
    def update_state(self, pred_word: str, confidence: float, valid: bool):
        """
        更新状态机
        
        Args:
            pred_word: 预测的词汇
            confidence: 置信度
            valid: 是否有效
        """
        # 冷却期间不接受新词
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # 无效预测，清空缓冲
        if not valid:
            self.consecutive_predictions.clear()
            return
        
        # 记录连续预测
        self.consecutive_predictions.append(pred_word)
        
        # 保持缓冲区大小
        if len(self.consecutive_predictions) > self.debounce_frames:
            self.consecutive_predictions.pop(0)
        
        # 检查是否连续 N 帧都是同一个词
        if len(self.consecutive_predictions) == self.debounce_frames:
            # 统计最频繁的词
            from collections import Counter
            most_common = Counter(self.consecutive_predictions).most_common(1)[0]
            word, count = most_common
            
            # 如果大部分帧都是同一个词，则认为识别成功
            if count >= self.debounce_frames * 0.7:  # 至少70%一致
                # 避免重复添加
                if not self.current_sentence or self.current_sentence[-1] != word:
                    self.current_sentence.append(word)
                    print(f"\n[识别] {word} (置信度: {confidence:.2f})")
                    print(f"[句子] {''.join(self.current_sentence)}")
                
                # 进入冷却期
                self.cooldown_counter = self.cooldown_frames
                self.consecutive_predictions.clear()
    
    def run(self):
        """运行实时识别"""
        # 打开摄像头
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
        
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        
        print("\n[INFO] 摄像头已启动")
        print("[提示] 按 'q' 退出, 按 'r' 重置句子, 按 'c' 清空窗口")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 无法读取摄像头画面")
                break
            
            # 镜像翻转（更自然）
            frame = cv2.flip(frame, 1)
            
            # 提取特征
            features, success = self.process_frame(frame)
            
            # 添加到窗口
            self.feature_window.append(features)
            
            # 预测
            pred_word, confidence, valid = self.predict()
            
            # 更新状态
            if pred_word:
                self.update_state(pred_word, confidence, valid)
            
            # 可视化
            display_frame = frame.copy()
            
            # 显示信息
            info_y = 30
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            info_y += 30
            window_status = f"Window: {len(self.feature_window)}/{self.window_size}"
            cv2.putText(display_frame, window_status, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if pred_word and valid:
                info_y += 30
                pred_text = f"Pred: {pred_word} ({confidence:.2f})"
                color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
                cv2.putText(display_frame, pred_text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示句子
            if self.current_sentence:
                sentence_text = ''.join(self.current_sentence)
                cv2.putText(display_frame, f"Sentence: {sentence_text}", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 冷却提示
            if self.cooldown_counter > 0:
                cv2.putText(display_frame, f"Cooldown: {self.cooldown_counter}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('CSL Real-time Recognition', display_frame)
            
            # FPS 计算
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                current_fps = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_sentence.clear()
                print("[INFO] 句子已重置")
            elif key == ord('c'):
                self.feature_window.clear()
                self.consecutive_predictions.clear()
                print("[INFO] 窗口已清空")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] 识别已停止")


def main():
    """主函数"""
    print("="*60)
    print("CSL-LSTM 实时手语识别")
    print("="*60)
    
    try:
        recognizer = RealtimeSignLanguageRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
