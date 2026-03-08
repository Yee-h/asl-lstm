"""
WLASL 数据集预处理脚本
将手语视频处理为 HDF5 格式，提取 135 个骨骼关键点用于深度学习模型训练。

使用 MediaPipe Tasks API 提取全身姿态（Pose 33 + Hands 21*2 + Face 478 -> 映射到 135 点）

优化处理流程：
- 坐标归一化：以肩中点为原点，使用全局肩宽中位数作为缩放尺度
- 时序对齐：长序列均匀重采样，短序列零填充，统一到 MAX_FRAMES 帧
- 速度特征：添加一阶差分 (dx, dy)，捕捉动作方向和速度

输出格式：
- WLASL_train.hdf5
- WLASL_val.hdf5
- WLASL_test.hdf5

每个 HDF5 文件结构：
- /{video_id}/data: float32 (MAX_FRAMES, 4, 135) - 归一化后的关键点坐标+速度
- /{video_id}/length: int64 - 有效帧数 (未 Padding 前的长度)
- /{video_id}/label: string - gloss 标签
- /{video_id}/video_name: string - 原始视频路径
- /{video_id}/width: int64 - 视频宽度
- /{video_id}/height: int64 - 视频高度

使用方法：
    uv run python src\\data_process\\preprocess_wlasl.py                     # 处理由 config.py 的 DATASET_SCALE 指定的规模
    uv run python src\\data_process\\preprocess_wlasl.py --limit 50          # 只处理前 50 个视频（用于测试）
    uv run python src\\data_process\\preprocess_wlasl.py --json my_data.json # 使用自定义 JSON 文件
"""

import argparse
import json
import multiprocessing as mp_process
import os
import sys
import threading
import urllib.request
import warnings
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import h5py
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 允许从项目根目录导入配置（config.py）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg

# 抑制 MediaPipe 的警告信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============== 用户配置参数（集中于 config.py）==============
# 数据集规模：设置为 100, 300, 1000, 2000 或 None
# - 100: 使用 WLASL100 子集 (100 个类别)
# - 300: 使用 WLASL300 子集 (300 个类别)
# - 1000: 使用 WLASL1000 子集 (1000 个类别)
# - 2000: 使用 WLASL2000 子集 (2000 个类别)
# - None: 使用完整数据集 (WLASL_v0.3.json)
DEFAULT_SUBSET = cfg.PATHS.dataset_scale  # 修改此值选择数据集规模

# 输入输出路径配置（从 config 导入）
DEFAULT_JSON_PATH = cfg.PATHS.default_json_path
DEFAULT_VIDEO_DIR = cfg.PATHS.default_video_dir
DEFAULT_OUTPUT_DIR = cfg.PATHS.default_output_dir
DEFAULT_OUTPUT_PREFIX = cfg.PATHS.default_output_prefix

# 处理数量限制：设置为 None 处理全部，或设置具体数字限制处理数量
DEFAULT_LIMIT = None

# ============== 以下配置通常无需修改（集中于 config.py） ==============
# 数据集规模映射 (subset -> JSON 文件)
SUBSET_JSON_MAP = cfg.PREPROCESS.subset_json_map

# 目标关键点数量 (WLASL 官方格式)
TARGET_KEYPOINTS = cfg.SEQUENCE.num_landmarks

# 模型文件路径
MODEL_DIR = cfg.MEDIAPIPE.model_dir
POSE_MODEL_PATH = cfg.MEDIAPIPE.pose_model_path
HAND_MODEL_PATH = cfg.MEDIAPIPE.hand_model_path
FACE_MODEL_PATH = cfg.MEDIAPIPE.face_model_path

# 模型下载 URL
POSE_MODEL_URL = cfg.MEDIAPIPE.pose_model_url
HAND_MODEL_URL = cfg.MEDIAPIPE.hand_model_url
FACE_MODEL_URL = cfg.MEDIAPIPE.face_model_url


def download_model(url: str, path: str) -> None:
    """
    检查并下载模型文件

    如果本地已存在模型文件则跳过下载
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"  [OK] 已存在: {os.path.basename(path)}")
    else:
        print(f"  [DL] 下载中: {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"  [OK] 下载完成: {os.path.basename(path)}")


class KeypointExtractor:
    """使用 MediaPipe Tasks API 提取人体骨骼关键点"""

    def __init__(self, use_video_mode: bool = False):
        """
        初始化 MediaPipe 模型

        Args:
            use_video_mode: 是否使用 VIDEO 模式（适用于连续视频帧处理，速度更快）
                            默认为 False (IMAGE 模式)，以保持与现有预处理脚本的兼容性
        """
        self.use_video_mode = use_video_mode
        self.running_mode = vision.RunningMode.VIDEO if use_video_mode else vision.RunningMode.IMAGE

        # 下载模型
        print("  检查和下载模型文件...")
        download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
        download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        download_model(FACE_MODEL_URL, FACE_MODEL_PATH)

        # 初始化 Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=cfg.MEDIAPIPE.pose_min_det_conf,
            min_pose_presence_confidence=cfg.MEDIAPIPE.pose_min_presence_conf,
            min_tracking_confidence=cfg.MEDIAPIPE.pose_min_track_conf,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # 初始化 Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=self.running_mode,
            num_hands=2,
            min_hand_detection_confidence=cfg.MEDIAPIPE.hand_min_det_conf,
            min_hand_presence_confidence=cfg.MEDIAPIPE.hand_min_presence_conf,
            min_tracking_confidence=cfg.MEDIAPIPE.hand_min_track_conf,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # 初始化 Face Landmarker
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=self.running_mode,
            num_faces=1,
            min_face_detection_confidence=cfg.MEDIAPIPE.face_min_det_conf,
            min_face_presence_confidence=cfg.MEDIAPIPE.face_min_presence_conf,
            min_tracking_confidence=cfg.MEDIAPIPE.face_min_track_conf,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        mode_str = "VIDEO" if use_video_mode else "IMAGE"
        print(f"  模型加载完成! (模式: {mode_str})")

    def close(self):
        """释放资源"""
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()

    def extract_frame(
        self, frame: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从单帧图像中提取关键点

        Args:
            frame: BGR 格式的图像 (H, W, 3)
            timestamp_ms: 时间戳（毫秒），VIDEO 模式下必须提供

        Returns:
            (keypoints, valid_mask)
            - keypoints: 关键点数组 (2, 135)
            - valid_mask: 每个关键点是否有效，形状 (135,)
        """
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 运行检测
        try:
            if self.use_video_mode:
                if timestamp_ms is None:
                    raise ValueError("VIDEO 模式下必须提供 timestamp_ms")
                pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
                hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
                face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
            else:
                pose_result = self.pose_detector.detect(mp_image)
                hand_result = self.hand_detector.detect(mp_image)
                face_result = self.face_detector.detect(mp_image)
        except Exception:
            return None, None

        # 提取关键点并映射到 135 点格式
        keypoints, valid_mask = self._map_to_135_keypoints(
            pose_result,
            hand_result,
            face_result,
        )

        return keypoints, valid_mask

    def extract_frame_optimized(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
        face_skip_ratio: int = 3,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        优化版的提取方法：
        1. 优先检测手部，如果没有检测到手部，直接返回，不进行后续检测。
        2. 面部检测降频处理 (TODO: 需要缓存机制，目前简单起见暂不实现缓存复用，仅降频)
           (注: VIDEO模式下MediaPipe内部有跟踪，跳帧可能会影响跟踪质量，但为了性能...)
           实际上，MediaPipe VIDEO模式要求连续帧。如果跳过面部检测，会导致跟踪丢失。
           所以这里主要优化点是：手部检测不到 -> 终止。

        Args:
            frame: BGR 图像
            timestamp_ms: 时间戳
            face_skip_ratio: (未使用，保留接口)

        Returns:
            (keypoints, valid_mask, hands_detected)
            - hands_detected: bool, 是否检测到了手部
        """
        if not self.use_video_mode or timestamp_ms is None:
            # 回退到普通模式
            kp, mask = self.extract_frame(frame, timestamp_ms)
            # 简单判断是否检测到手部 (索引 25-66 是手部)
            has_hands = False
            if mask is not None:
                has_hands = np.sum(mask[25:67]) > 0
            return kp, mask, has_hands

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            # 1. 检测 Pose (身体姿态通常比较稳定且对 Crop 手部有用，但在 MP Tasks API 中 Hand 独立运行)
            # 策略调整：先检测 Hand。如果没手，直接退出。
            # Hand Detector
            hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

            has_hands = False
            if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                has_hands = True

            if not has_hands:
                # 没手，为了保持 Pose 跟踪 (如果需要)，可以运行 Pose，但面部绝对跳过
                # 如果完全不运行 Pose，下一次 Pose 跟踪可能会重新初始化 (耗时)。
                # 为了极致性能，没手直接返回。MP 内部会在下一帧尝试重新检测。
                return None, None, False

            # 2. 检测 Pose
            pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)

            # 3. 检测 Face (全速运行以保持跟踪，或者由调用者控制频率？)
            # 由于 VIDEO 模式依赖时序，这里必须每帧运行才能保持最佳跟踪效果。
            # 性能瓶颈主要在于 Face Mesh (478点)。
            face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

            # 映射
            keypoints, valid_mask = self._map_to_135_keypoints(
                pose_result,
                hand_result,
                face_result,
            )
            return keypoints, valid_mask, True

        except Exception:
            return None, None, False

    def _map_to_135_keypoints(
        self, pose_result, hand_result, face_result
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将 MediaPipe 结果映射到 135 个关键点

        WLASL 官方 135 点格式：
        - Body: 25 点 (OpenPose Body 25 格式)
        - Left Hand: 21 点
        - Right Hand: 21 点
        - Face: 68 点 (68-point facial landmarks)

        Returns:
            (keypoints, valid_mask)
            - keypoints: 形状为 (2, 135) 的数组，包含 (x, y) 坐标
            - valid_mask: 形状为 (135,) 的数组，1 表示该关键点有效
        """
        keypoints = np.zeros((2, TARGET_KEYPOINTS), dtype=np.float64)
        valid_mask = np.zeros((TARGET_KEYPOINTS,), dtype=np.uint8)

        # 1. 提取 Body 关键点 (25 点)
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose = pose_result.pose_landmarks[0]
            # OpenPose Body 25 索引到 MediaPipe Pose 索引的映射
            body_mapping = [
                0,  # 0: Nose
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
                5,  # 15: REye
                2,  # 16: LEye
                8,  # 17: REar
                7,  # 18: LEar
                32,  # 19: LBigToe
                31,  # 20: LSmallToe
                29,  # 21: LHeel
                31,  # 22: RBigToe
                32,  # 23: RSmallToe
                30,  # 24: RHeel
            ]

            for i, mp_idx in enumerate(body_mapping):
                if mp_idx >= 0:
                    keypoints[0, i] = np.clip(pose[mp_idx].x, 0.0, 1.0)
                    keypoints[1, i] = np.clip(pose[mp_idx].y, 0.0, 1.0)
                    valid_mask[i] = 1
                elif i == 1:  # Neck
                    keypoints[0, i] = np.clip((pose[11].x + pose[12].x) / 2, 0.0, 1.0)
                    keypoints[1, i] = np.clip((pose[11].y + pose[12].y) / 2, 0.0, 1.0)
                    valid_mask[i] = 1
                elif i == 8:  # MidHip
                    keypoints[0, i] = np.clip((pose[23].x + pose[24].x) / 2, 0.0, 1.0)
                    keypoints[1, i] = np.clip((pose[23].y + pose[24].y) / 2, 0.0, 1.0)
                    valid_mask[i] = 1

        # 2. 提取手部关键点 (21*2 点, 索引 25-66)
        # 注意：MediaPipe 的左右手标签是相对于摄像头视角的（镜像）
        # 需要交换左右手以匹配 OpenPose 格式（相对于人体自身）
        if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_result.hand_landmarks, hand_result.handedness)
            ):
                # 判断是左手还是右手（需要交换）
                hand_label = handedness[0].category_name.lower()

                # MediaPipe 的 'left' 是图像中的左侧，对应人体的右手
                # MediaPipe 的 'right' 是图像中的右侧，对应人体的左手
                if hand_label == "left":
                    offset = 46  # MediaPipe left -> OpenPose 右手: 索引 46-66
                else:
                    offset = 25  # MediaPipe right -> OpenPose 左手: 索引 25-45

                for i, lm in enumerate(hand_landmarks):
                    # 裁剪坐标到 [0, 1] 范围
                    keypoints[0, offset + i] = np.clip(lm.x, 0.0, 1.0)
                    keypoints[1, offset + i] = np.clip(lm.y, 0.0, 1.0)
                    valid_mask[offset + i] = 1

        # 3. 提取面部关键点 (68 点, 索引 67-134)
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face = face_result.face_landmarks[0]
            # 68 面部关键点映射 (基于 dlib 68 点标准)
            face_68_indices = [
                # 面部轮廓 (17 点)
                10,
                338,
                297,
                332,
                284,
                251,
                389,
                356,
                454,
                323,
                361,
                288,
                397,
                365,
                379,
                378,
                400,
                # 左眉 (5 点)
                70,
                63,
                105,
                66,
                107,
                # 右眉 (5 点)
                336,
                296,
                334,
                293,
                300,
                # 鼻梁 (4 点)
                168,
                6,
                197,
                195,
                # 鼻尖下方 (5 点)
                5,
                4,
                1,
                19,
                94,
                # 左眼 (6 点)
                33,
                160,
                158,
                133,
                153,
                144,
                # 右眼 (6 点)
                362,
                385,
                387,
                263,
                373,
                380,
                # 外唇 (12 点)
                61,
                185,
                40,
                39,
                37,
                0,
                267,
                269,
                270,
                409,
                291,
                308,
                # 内唇 (8 点)
                78,
                191,
                80,
                81,
                82,
                13,
                312,
                311,
            ]

            for i, face_idx in enumerate(face_68_indices):
                if i < 68:
                    keypoints[0, 67 + i] = np.clip(face[face_idx].x, 0.0, 1.0)
                    keypoints[1, 67 + i] = np.clip(face[face_idx].y, 0.0, 1.0)
                    valid_mask[67 + i] = 1

        return keypoints, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# 并行关键点提取器（线程池 + IMAGE 模式）
# ─────────────────────────────────────────────────────────────────────────────


class ParallelKeypointExtractor:
    """
    可复用的并行关键点提取器。

    使用 ThreadPoolExecutor + 线程本地 IMAGE 模式 KeypointExtractor 实例，
    在多个 worker 线程上并行提取帧的骨骼关键点。

    提供两种使用模式：
    - **批量模式** (``extract_batch``): 提交所有帧，阻塞等待全部完成。适合离线推理。
    - **流式模式** (``submit_frame`` / ``collect_completed``): 逐帧提交、
      按提交顺序取回已完成结果。适合实时推理的流水线场景。

    内部每个 worker 线程通过 ``threading.local()`` 持有独立的
    ``KeypointExtractor(use_video_mode=False)`` 实例，IMAGE 模式无状态，
    可安全并行。

    Args:
        num_workers: 线程池 worker 数量。0 表示自动（``cpu_count - 1``），
                     1 表示禁用并行（仍使用线程池但只有 1 个 worker）。
    """

    # ── 类型别名 ────────────────────────────────────────────────────────
    FrameResult = Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]
    """(keypoints, valid_mask, has_hands)"""

    def __init__(self, num_workers: int = 0) -> None:
        self._num_workers = self._resolve_num_workers(num_workers)
        self._tls = threading.local()
        self._extractors: List[KeypointExtractor] = []
        self._extractors_lock = threading.Lock()

        # 流式模式的待处理 Future 队列（先入先出，保持帧顺序）
        self._pending: deque[Future] = deque()

        # 懒初始化线程池（首次提交时创建）
        self._pool: Optional[ThreadPoolExecutor] = None
        self._closed = False

    # ── 公共接口 ────────────────────────────────────────────────────────

    @property
    def num_workers(self) -> int:
        """返回实际使用的 worker 数量。"""
        return self._num_workers

    def extract_batch(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List["ParallelKeypointExtractor.FrameResult"]:
        """
        批量提取所有帧的关键点（阻塞）。

        按帧的原始顺序返回结果列表。适合离线推理一次性处理所有帧。

        Args:
            frames: BGR numpy 数组列表。
            progress_callback: 可选回调 ``(completed, total)``，每完成一帧调用一次。

        Returns:
            与 *frames* 等长的列表，每个元素为
            ``(keypoints, valid_mask, has_hands)``。
        """
        if self._closed:
            raise RuntimeError("ParallelKeypointExtractor 已关闭")

        pool = self._ensure_pool()
        total = len(frames)
        results: List[ParallelKeypointExtractor.FrameResult] = [(None, None, False)] * total

        future_to_idx = {}
        for idx, frame in enumerate(frames):
            future = pool.submit(self._extract_one_frame, frame)
            future_to_idx[future] = idx

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = (None, None, False)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)

        return results

    def submit_frame(self, frame: np.ndarray) -> Future:
        """
        提交单帧进行异步提取（非阻塞）。

        返回的 ``Future`` 会被自动追加到内部的有序待处理队列中，
        可通过 ``collect_completed()`` 按提交顺序取回结果。

        Args:
            frame: BGR numpy 数组。

        Returns:
            ``concurrent.futures.Future`` 对象。
        """
        if self._closed:
            raise RuntimeError("ParallelKeypointExtractor 已关闭")

        pool = self._ensure_pool()
        future = pool.submit(self._extract_one_frame, frame)
        self._pending.append(future)
        return future

    def collect_completed(self) -> List["ParallelKeypointExtractor.FrameResult"]:
        """
        从待处理队列的头部取回连续已完成的结果。

        按 **提交顺序** 返回：从队列头部开始，只要 Future 已完成就弹出并
        收集结果；遇到第一个未完成的 Future 则停止（保证顺序性）。

        Returns:
            已完成的 ``(keypoints, valid_mask, has_hands)`` 结果列表（可能为空）。
        """
        results: List[ParallelKeypointExtractor.FrameResult] = []
        while self._pending:
            front = self._pending[0]
            if not front.done():
                break
            self._pending.popleft()
            try:
                results.append(front.result())
            except Exception:
                results.append((None, None, False))
        return results

    def close(self) -> None:
        """关闭线程池并释放所有 worker 创建的 KeypointExtractor 实例。"""
        if self._closed:
            return
        self._closed = True

        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

        with self._extractors_lock:
            for ext in self._extractors:
                try:
                    ext.close()
                except Exception:
                    pass
            self._extractors.clear()

        self._pending.clear()

    # ── 内部方法 ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_num_workers(n: int) -> int:
        """解析实际 worker 数量。0 = 自动（cpu_count - 1）。"""
        if n <= 0:
            import multiprocessing

            n = max(1, multiprocessing.cpu_count() - 1)
        return max(1, n)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        """懒初始化线程池。"""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._num_workers,
                initializer=self._pool_initializer,
            )
        return self._pool

    def _pool_initializer(self) -> None:
        """ThreadPoolExecutor initializer：在每个 worker 线程中创建 IMAGE 模式的 KeypointExtractor。"""
        ext = KeypointExtractor(use_video_mode=False)
        self._tls.extractor = ext
        with self._extractors_lock:
            self._extractors.append(ext)

    def _extract_one_frame(self, frame: np.ndarray) -> "ParallelKeypointExtractor.FrameResult":
        """
        worker 函数：使用线程本地的 KeypointExtractor 提取单帧关键点。

        Returns:
            (keypoints, valid_mask, has_hands)
        """
        ext: KeypointExtractor = self._tls.extractor
        kp, mask = ext.extract_frame(frame, timestamp_ms=None)
        has_hands = False
        if mask is not None:
            has_hands = bool(np.sum(mask[25:67]) > 0)
        return kp, mask, has_hands


class PreprocessHelper:
    """
    骨骼关键点后处理器 (V2 优化版)

    改进：
    1. 弹弓效应消除：先计算速度再 Padding，避免边界速度噪声
    2. 有效长度保留：返回 valid_length 支持 Masking
    3. 线性插值：使用 cv2.resize 替代 FFT 重采样，更适合非周期手语动作
    """

    def __init__(self, max_frames: int = 90):
        self.max_frames = max_frames
        self.LEFT_SHOULDER_IDX = 5
        self.RIGHT_SHOULDER_IDX = 2
        self.MID_HIP_IDX = 8

    def process_video_sequence(
        self,
        raw_frames: np.ndarray,
        raw_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int, np.ndarray, float]:
        """
        处理单个视频的所有帧数据。

        Args:
            raw_frames: 原始关键点数据 (T, 2, 135)
            raw_masks: 关键点有效掩码 (T, 135)

        Returns:
            (final_data, valid_len, final_mask, quality_score)
            - final_data: (max_frames, C, 135)，默认 C=4 (x,y,dx,dy)
            - valid_len: 有效帧数
            - final_mask: (max_frames, 135)
            - quality_score: 有效关键点比例
        """
        data = raw_frames.transpose(0, 2, 1).astype(np.float32)  # (T, 135, 2)

        if raw_masks is None:
            masks = ((raw_frames[:, 0, :] != 0) | (raw_frames[:, 1, :] != 0)).astype(np.uint8)
        else:
            masks = raw_masks.astype(np.uint8)

        if cfg.PREPROCESS.enable_missing_interp:
            data, masks = self._interpolate_missing_short_gaps(data, masks)

        if cfg.PREPROCESS.enable_xy_smooth:
            data = self._smooth_xy(data, masks)

        if cfg.PREPROCESS.enable_shoulder_axis_align:
            data = self._align_shoulder_axis(data, masks)

        norm_data = self._normalize_with_mask(data, masks)

        norm_data, masks, valid_len = self._resample_if_long(norm_data, masks)

        feature_data = self._add_velocity_features(norm_data, valid_len)
        if cfg.SEQUENCE.enable_accel_feature:
            feature_data = self._add_acceleration_features(feature_data, valid_len)

        final_data, final_mask = self._pad_sequence(feature_data, masks)
        quality_score = float(final_mask[:valid_len].mean()) if valid_len > 0 else 0.0

        final_data = final_data.transpose(0, 2, 1)  # (T, C, V)
        return final_data, valid_len, final_mask, quality_score

    def _interpolate_missing_short_gaps(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对短缺失段进行线性插值，并更新掩码。"""
        filled = data.copy()
        updated_mask = mask.copy()
        max_gap = max(1, int(cfg.PREPROCESS.interp_max_gap))
        T, V, _ = data.shape

        for v in range(V):
            t = 0
            while t < T:
                if updated_mask[t, v] == 1:
                    t += 1
                    continue

                start = t
                while t < T and updated_mask[t, v] == 0:
                    t += 1
                end = t - 1

                gap_len = end - start + 1
                left = start - 1
                right = t

                if gap_len > max_gap:
                    continue
                if left < 0 or right >= T:
                    continue
                if updated_mask[left, v] == 0 or updated_mask[right, v] == 0:
                    continue

                for c in range(2):
                    left_val = filled[left, v, c]
                    right_val = filled[right, v, c]
                    step = (right_val - left_val) / (gap_len + 1)
                    for k in range(gap_len):
                        filled[start + k, v, c] = left_val + step * (k + 1)

                updated_mask[start : end + 1, v] = 1

        return filled, updated_mask

    def _smooth_xy(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """仅对有效关键点做 EMA 平滑。"""
        if cfg.PREPROCESS.smooth_method.lower() != "ema":
            return data

        smoothed = data.copy()
        alpha = float(cfg.PREPROCESS.smooth_ema_alpha)
        T, V, _ = data.shape

        for v in range(V):
            valid_idx = np.where(mask[:, v] == 1)[0]
            if valid_idx.size < 2:
                continue

            for c in range(2):
                prev = smoothed[valid_idx[0], v, c]
                for idx in valid_idx[1:]:
                    prev = alpha * smoothed[idx, v, c] + (1.0 - alpha) * prev
                    smoothed[idx, v, c] = prev

        return smoothed

    def _align_shoulder_axis(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """将每帧肩轴旋转到水平，减小拍摄角度偏差。"""
        aligned = data.copy()
        eps = float(cfg.PREPROCESS.normalize_eps)

        for t in range(aligned.shape[0]):
            if mask[t, self.LEFT_SHOULDER_IDX] == 0 or mask[t, self.RIGHT_SHOULDER_IDX] == 0:
                continue

            left = aligned[t, self.LEFT_SHOULDER_IDX, :2]
            right = aligned[t, self.RIGHT_SHOULDER_IDX, :2]
            vec = left - right
            norm = np.linalg.norm(vec)
            if norm <= eps:
                continue

            angle = np.arctan2(vec[1], vec[0])
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

            root = (left + right) / 2.0
            valid_points = np.where(mask[t] == 1)[0]
            for v in valid_points:
                pt = aligned[t, v, :2] - root
                aligned[t, v, :2] = rot @ pt + root

        return aligned

    def _normalize_with_mask(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """基于掩码做平移与尺度归一化。"""
        T = data.shape[0]
        roots = np.zeros((T, 2), dtype=np.float32)
        shoulder_scales = []
        torso_scales = []

        for t in range(T):
            has_left = mask[t, self.LEFT_SHOULDER_IDX] == 1
            has_right = mask[t, self.RIGHT_SHOULDER_IDX] == 1

            if has_left and has_right:
                left = data[t, self.LEFT_SHOULDER_IDX, :2]
                right = data[t, self.RIGHT_SHOULDER_IDX, :2]
                roots[t] = (left + right) / 2.0
                shoulder_dist = np.linalg.norm(left - right)
                if shoulder_dist > cfg.PREPROCESS.normalize_eps:
                    shoulder_scales.append(float(shoulder_dist))
            elif t > 0:
                roots[t] = roots[t - 1]

            if mask[t, self.MID_HIP_IDX] == 1:
                torso_dist = np.linalg.norm(roots[t] - data[t, self.MID_HIP_IDX, :2])
                if torso_dist > cfg.PREPROCESS.normalize_eps:
                    torso_scales.append(float(torso_dist))

        shoulder_scale = np.median(shoulder_scales) if shoulder_scales else 1.0
        if cfg.PREPROCESS.scale_mode == "shoulder_torso_fusion" and torso_scales:
            video_scale = 0.7 * shoulder_scale + 0.3 * np.median(torso_scales)
        else:
            video_scale = shoulder_scale

        if video_scale <= cfg.PREPROCESS.normalize_eps or np.isnan(video_scale):
            video_scale = 1.0

        normalized = (data - roots[:, np.newaxis, :]) / float(video_scale)
        normalized[mask == 0] = 0.0
        return normalized

    def _resample_if_long(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """仅对超过 max_frames 的序列重采样，mask 使用最近邻。"""
        T = data.shape[0]
        if T > self.max_frames:
            V, C = data.shape[1], data.shape[2]

            flattened = data.reshape(T, -1)
            data_resampled = cv2.resize(
                flattened,
                (V * C, self.max_frames),
                interpolation=cv2.INTER_LINEAR,
            ).reshape(self.max_frames, V, C)

            mask_resampled = cv2.resize(
                mask.astype(np.float32),
                (V, self.max_frames),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_resampled = (mask_resampled > 0.5).astype(np.uint8)

            return data_resampled, mask_resampled, self.max_frames

        return data, mask, T

    def _pad_sequence(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """将数据和掩码一起补齐/裁剪到 max_frames。"""
        T = data.shape[0]

        if T > self.max_frames:
            return data[: self.max_frames], mask[: self.max_frames]

        if T < self.max_frames:
            pad_len = self.max_frames - T
            data_padding = ((0, pad_len), (0, 0), (0, 0))
            mask_padding = ((0, pad_len), (0, 0))
            data = np.pad(data, data_padding, mode="constant", constant_values=0)
            mask = np.pad(mask, mask_padding, mode="constant", constant_values=0)

        return data, mask

    def _add_velocity_features(self, data: np.ndarray, valid_len: int) -> np.ndarray:
        """增加一阶差分速度特征 (dx, dy)。"""
        if data.shape[0] == 0:
            return np.zeros((0, data.shape[1], 4), dtype=np.float32)

        velocity = np.zeros_like(data, dtype=np.float32)
        if valid_len > 1:
            velocity[1:valid_len] = data[1:valid_len] - data[: valid_len - 1]

        return np.concatenate([data, velocity], axis=-1)

    def _add_acceleration_features(self, data: np.ndarray, valid_len: int) -> np.ndarray:
        """在 x,y,dx,dy 基础上增加二阶差分加速度 (ddx, ddy)。"""
        if data.shape[0] == 0 or data.shape[2] < 4:
            return data

        velocity = data[:, :, 2:4]
        acceleration = np.zeros_like(velocity, dtype=np.float32)
        if valid_len > 1:
            acceleration[1:valid_len] = velocity[1:valid_len] - velocity[: valid_len - 1]

        return np.concatenate([data, acceleration], axis=-1)


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
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_info = {}

    # 判断 JSON 格式
    if isinstance(data, list):
        # WLASL_v0.3.json 格式（列表形式）
        for entry in data:
            gloss = entry["gloss"]
            for instance in entry["instances"]:
                video_id = instance["video_id"]
                split = instance.get("split", "train")

                video_info[video_id] = {
                    "gloss": gloss,
                    "split": split,
                    "bbox": instance.get("bbox"),
                    "fps": instance.get("fps"),
                    "frame_start": instance.get("frame_start", 1),
                    "frame_end": instance.get("frame_end", -1),
                    "variation_id": instance.get("variation_id"),
                    "signer_id": instance.get("signer_id"),
                }

    elif isinstance(data, dict):
        # nslt_*.json 格式（字典形式）
        # 需要从 WLASL_v0.3.json 获取 gloss 标签映射
        gloss_map = _build_gloss_map()

        for video_id, info in data.items():
            split = info.get("subset", "train")
            # action 字段包含 [label_id, ...]，取第一个作为主标签
            action = info.get("action", [])
            label_id = action[0] if action else -1
            gloss = gloss_map.get(label_id, f"unknown_{label_id}")

            video_info[video_id] = {
                "gloss": gloss,
                "split": split,
                "label_id": label_id,
                "bbox": None,
                "fps": None,
                "frame_start": 1,
                "frame_end": -1,
            }

    return video_info


def _build_gloss_map() -> dict:
    """
    从 WLASL_v0.3.json 构建 label_id -> gloss 的映射

    Returns:
        字典：{label_id: gloss}
    """
    gloss_map = {}

    # 优先使用 config 中配置的路径
    wlasl_json = cfg.PATHS.wlasl_gloss_json
    if wlasl_json is None or not os.path.exists(wlasl_json):
        # 回退到默认路径
        wlasl_json = os.path.join(os.path.dirname(DEFAULT_JSON_PATH), "WLASL_v0.3.json")

    if not os.path.exists(wlasl_json):
        print(f"  警告: 找不到 {wlasl_json}，无法获取 gloss 标签")
        return gloss_map

    with open(wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 在 WLASL_v0.3.json 中，gloss 按索引顺序排列
    for idx, entry in enumerate(data):
        gloss_map[idx] = entry["gloss"]

    return gloss_map


def process_video(
    video_path: str,
    extractor: KeypointExtractor,
    frame_start: int = 1,
    frame_end: int = -1,
    preprocess_helper: Optional["PreprocessHelper"] = None,
) -> tuple[Optional[np.ndarray], int, int, int, Optional[np.ndarray], float]:
    """
    处理单个视频，提取所有帧的关键点并进行后处理

    Args:
        video_path: 视频文件路径
        extractor: 关键点提取器
        frame_start: 起始帧（1-indexed）
        frame_end: 结束帧（-1 表示到最后）
        preprocess_helper: 预处理辅助器，执行归一化、时序对齐和速度特征提取

    Returns:
        (keypoints_array, valid_len, width, height, mask_array, quality_score)
        - keypoints_array: 处理后的数据
        - valid_len: 有效帧长度
        - width: 视频宽度
        - height: 视频高度
        - mask_array: 关键点有效掩码 (T, 135)
        - quality_score: 有效关键点比例
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, 0, 0, 0, None, 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 处理帧范围
    start_idx = max(0, frame_start - 1)  # 转换为 0-indexed
    end_idx = total_frames if frame_end == -1 else min(frame_end, total_frames)

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    all_keypoints = []
    all_masks = []
    frame_idx = start_idx

    while frame_idx < end_idx:
        ret, frame = cap.read()
        if not ret:
            break

        # 提取关键点
        keypoints, valid_mask = extractor.extract_frame(frame)
        if keypoints is not None and valid_mask is not None:
            all_keypoints.append(keypoints)
            all_masks.append(valid_mask)
        else:
            # 如果某帧检测失败，使用零填充
            all_keypoints.append(np.zeros((2, TARGET_KEYPOINTS), dtype=np.float64))
            all_masks.append(np.zeros((TARGET_KEYPOINTS,), dtype=np.uint8))

        frame_idx += 1

    cap.release()

    if len(all_keypoints) == 0:
        return None, 0, width, height, None, 0.0

    # 堆叠为 (T, 2, 135) 形状
    keypoints_array = np.stack(all_keypoints, axis=0)
    mask_array = np.stack(all_masks, axis=0)
    valid_len = keypoints_array.shape[0]
    quality_score = float(mask_array[:valid_len].mean()) if valid_len > 0 else 0.0

    # 应用后处理：归一化、时序对齐、速度特征
    if preprocess_helper is not None:
        keypoints_array, valid_len, mask_array, quality_score = (
            preprocess_helper.process_video_sequence(
                keypoints_array,
                mask_array,
            )
        )

    return keypoints_array, valid_len, width, height, mask_array, quality_score


def _worker_init():
    """
    多进程工作进程初始化函数
    在每个工作进程中初始化 MediaPipe 模型和预处理器
    """
    global _worker_extractor, _worker_preprocess_helper
    # 抑制子进程中的警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    _worker_extractor = KeypointExtractor()
    _worker_preprocess_helper = PreprocessHelper(max_frames=cfg.SEQUENCE.max_frames)


def _worker_process_video(task: tuple) -> dict:
    """
    多进程工作函数：处理单个视频

    Args:
        task: (video_id, video_path, frame_start, frame_end, gloss, split)

    Returns:
        处理结果字典，包含 success 状态和数据
    """
    global _worker_extractor, _worker_preprocess_helper
    video_id, video_path, frame_start, frame_end, gloss, split = task

    try:
        keypoints, valid_len, width, height, mask, quality = process_video(
            video_path,
            _worker_extractor,
            frame_start=frame_start,
            frame_end=frame_end,
            preprocess_helper=_worker_preprocess_helper,
        )

        if keypoints is None or keypoints.shape[0] == 0:
            return {"success": False, "video_id": video_id}

        if quality < cfg.PREPROCESS.min_valid_ratio_per_sample:
            return {
                "success": False,
                "video_id": video_id,
                "filtered": True,
                "reason": "low_quality",
            }

        return {
            "success": True,
            "video_id": video_id,
            "split": split,
            "data": keypoints.astype(np.float32),
            "mask": mask.astype(np.uint8) if mask is not None else None,
            "quality": float(quality),
            "length": valid_len,
            "label": gloss,
            "video_name": f"{video_id}.mp4",
            "width": width,
            "height": height,
        }
    except Exception as e:
        return {"success": False, "video_id": video_id, "error": str(e)}


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
    with h5py.File(output_path, "w") as f:
        # 使用数字索引作为 Group 名称（匹配官方格式）
        for idx, (video_id, data) in enumerate(
            tqdm(video_data.items(), desc=f"Writing {os.path.basename(output_path)}")
        ):
            # 创建 Group，使用数字索引
            grp = f.create_group(str(idx))

            # 存储数据 (float32)
            grp.create_dataset("data", data=data["data"], dtype="float32")

            if data.get("mask") is not None:
                grp.create_dataset("mask", data=data["mask"], dtype="uint8")

            if data.get("quality") is not None:
                grp.create_dataset("quality", data=data["quality"], dtype="float32")

            # 存储有效长度
            grp.create_dataset("length", data=data.get("length", 0), dtype="int64")

            # 存储标签 (字符串需要特殊处理)
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset("label", data=data["label"], dtype=dt)

            # 生成官方格式的 video_name 路径
            # 官方格式: rgb/WLASL{subset}/{split}/{video_id}.mp4
            if subset:
                video_name = f"rgb/WLASL{subset}/{split}/{video_id}.mp4"
            else:
                video_name = data["video_name"]
            grp.create_dataset("video_name", data=video_name, dtype=dt)

            # 存储视频尺寸
            grp.create_dataset("width", data=data["width"], dtype="int64")
            grp.create_dataset("height", data=data["height"], dtype="int64")

            grp.attrs["pipeline_version"] = cfg.PREPROCESS.pipeline_version
            grp.attrs["feature_channels"] = ",".join(cfg.SEQUENCE.base_feature_channels)


def create_maplabels_json(
    output_path: str,
    all_data: dict,
    subset: Optional[int] = None,
) -> None:
    """
    创建 label 映射文件（匹配官方格式）

    标签顺序来源于 WLASL_v0.3.json：
    - WLASL_v0.3.json 是一个列表，包含 2000 个 gloss
    - gloss 在列表中的索引就是它的 label_id
    - 对于 WLASL100，使用索引 0-99 对应的 gloss
    - 对于 WLASL300，使用索引 0-299 对应的 gloss
    - 以此类推

    标签映射契约：
    {
        "id_to_label": {
            "0": "book",
            "1": "drink",
            ...
        },
        "label_to_id": {
            "book": 0,
            "drink": 1,
            ...
        }
    }

    Args:
        output_path: 输出 JSON 文件路径
        all_data: 所有 split 的数据 {split: {video_id: {...}}}
        subset: 数据集规模 (100/300/1000/2000)
    """
    # 收集所有唯一的 label
    labels = set()
    for split_data in all_data.values():
        for video_data in split_data.values():
            labels.add(video_data["label"])

    # 从 WLASL_v0.3.json 读取 gloss 顺序
    # gloss 在列表中的索引就是它的 label_id
    wlasl_json_path = os.path.join(cfg.PATHS.raw_data_dir, "WLASL_v0.3.json")

    if os.path.exists(wlasl_json_path):
        try:
            with open(wlasl_json_path, "r", encoding="utf-8") as f:
                wlasl_data = json.load(f)

            # 确定要使用的 gloss 范围
            num_classes = subset if subset else len(wlasl_data)

            # 构建标签映射，保持 WLASL_v0.3.json 中的顺序
            id_to_label = {}
            label_to_id = {}

            for idx in range(min(num_classes, len(wlasl_data))):
                gloss = wlasl_data[idx]["gloss"]
                id_to_label[str(idx)] = gloss
                label_to_id[gloss] = idx

            print(f"  从 WLASL_v0.3.json 读取标签顺序: {len(id_to_label)} 个类别")

            # 验证所有收集到的标签都在映射中
            missing_labels = labels - set(label_to_id.keys())
            if missing_labels:
                print(
                    f"  警告: 以下标签不在 WLASL_v0.3.json 前 {num_classes} 个中: {missing_labels}"
                )
                # 为缺失的标签分配 ID（从 num_classes 开始）
                for label in sorted(missing_labels):
                    print(f"    添加缺失标签: {label} -> {num_classes}")
                    id_to_label[str(num_classes)] = label
                    label_to_id[label] = num_classes
                    num_classes += 1
        except Exception as e:
            print(f"  警告: 无法读取 {wlasl_json_path}: {e}")
            print("  回退到字母顺序排序")
            sorted_labels = sorted(labels)
            id_to_label = {str(idx): label for idx, label in enumerate(sorted_labels)}
            label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    else:
        # 如果没有 WLASL_v0.3.json，按字母顺序排序并分配 ID（回退方案）
        print(f"  警告: 未找到 {wlasl_json_path}，使用字母顺序排序")
        sorted_labels = sorted(labels)
        id_to_label = {str(idx): label for idx, label in enumerate(sorted_labels)}
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}

    # 写入 JSON 文件（包含双向映射，匹配官方格式）
    maplabels = {"id_to_label": id_to_label, "label_to_id": label_to_id}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(maplabels, f, indent=4, ensure_ascii=False)

    print(f"  已生成标签映射文件: {output_path} ({len(label_to_id)} 个类别)")


def compute_train_feature_stats(train_data: dict) -> Optional[dict]:
    """基于训练集计算每个通道的均值和标准差（仅统计有效关键点）。"""
    if not train_data:
        return None

    sample = next(iter(train_data.values()))
    channels = int(sample["data"].shape[1])

    sum_c = np.zeros((channels,), dtype=np.float64)
    sq_c = np.zeros((channels,), dtype=np.float64)
    cnt_c = np.zeros((channels,), dtype=np.float64)

    for item in train_data.values():
        data = item["data"]
        valid_len = min(int(item.get("length", data.shape[0])), data.shape[0])
        if valid_len <= 0:
            continue

        data_valid = data[:valid_len]
        mask = item.get("mask")

        if mask is not None:
            mask_valid = mask[:valid_len].astype(bool)
        else:
            mask_valid = np.ones((valid_len, data.shape[2]), dtype=bool)

        for c in range(channels):
            values = data_valid[:, c, :][mask_valid]
            if values.size == 0:
                continue
            sum_c[c] += float(values.sum())
            sq_c[c] += float(np.square(values).sum())
            cnt_c[c] += float(values.size)

    safe_cnt = np.maximum(cnt_c, 1.0)
    mean_c = sum_c / safe_cnt
    var_c = np.maximum((sq_c / safe_cnt) - np.square(mean_c), 0.0)
    std_c = np.sqrt(var_c)

    return {
        "channels": cfg.SEQUENCE.base_feature_channels,
        "mean": mean_c.tolist(),
        "std": std_c.tolist(),
        "count": cnt_c.tolist(),
        "version": cfg.PREPROCESS.pipeline_version,
    }


def save_train_feature_stats(stats: dict) -> None:
    """保存训练集统计量到配置路径。"""
    stats_path = cfg.PREPROCESS.feature_stats_path
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  已保存训练集统计量: {stats_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="WLASL 数据集预处理脚本 - 提取骨骼关键点并保存为 HDF5 格式"
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f"JSON 元数据文件路径 (默认: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--video-dir",
        "-v",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"视频文件目录 (默认: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"输出文件名前缀 (默认: {DEFAULT_OUTPUT_PREFIX})",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="限制处理的视频数量 (用于测试)"
    )
    args = parser.parse_args()

    # 数据集规模完全由 config.py 的 DATASET_SCALE 决定
    subset = DEFAULT_SUBSET
    args.subset = subset

    if subset is not None:
        if subset in SUBSET_JSON_MAP:
            args.json = SUBSET_JSON_MAP[subset]
            args.prefix = f"WLASL{subset}"
            args.video_dir = os.path.join(cfg.PATHS.raw_data_dir, f"WLASL{subset}")
            # 输出目录应该包含数据规模子目录
            args.output_dir = os.path.join(args.output_dir, f"WLASL{subset}")
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
    available_videos = {}  # 改为字典: {video_id: 完整路径}
    # 递归查找 mp4，兼容 train/val/test 子目录结构
    for vf in video_dir.rglob("*.mp4"):
        video_id = vf.stem
        available_videos[video_id] = str(vf)  # 存储完整路径
    print(f"  找到 {len(available_videos)} 个视频文件")

    # 统计各个 split 的数量
    split_counts = {"train": 0, "val": 0, "test": 0}
    for video_id in available_videos:
        if video_id in video_info:
            split = video_info[video_id]["split"]
            if split in split_counts:
                split_counts[split] += 1
    print(
        f"  分布: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}"
    )

    # 3. 获取工作进程数
    num_workers = cfg.PREPROCESS.num_workers
    if num_workers is None:
        num_workers = max(1, mp_process.cpu_count() - 1)  # 保留一个核心给系统
    num_workers = max(1, num_workers)
    use_multiprocessing = num_workers > 1

    # 4. 处理视频并生成 HDF5
    print("\n[3/5] 准备处理视频...")

    # 按 split 分组
    split_data = {"train": {}, "val": {}, "test": {}}

    # 处理每个可用的视频
    processed_count = 0
    failed_count = 0
    filtered_count = 0

    # 获取要处理的视频列表
    # available_videos 现在是 {video_id: 完整路径} 的字典
    videos_to_process = [
        (vid, video_info[vid], available_videos[vid])
        for vid in available_videos
        if vid in video_info
    ]

    # 应用限制
    if args.limit and args.limit < len(videos_to_process):
        videos_to_process = videos_to_process[: args.limit]
        print(f"  已限制为前 {args.limit} 个视频")

    # 构建任务列表
    tasks = []
    for video_id, info, video_path in videos_to_process:
        tasks.append(
            (
                video_id,
                video_path,
                info.get("frame_start", 1),
                info.get("frame_end", -1),
                info["gloss"],
                info.get("split", "train"),
            )
        )

    if tasks:
        num_workers = min(num_workers, len(tasks))
    use_multiprocessing = num_workers > 1

    print("\n[4/5] 处理视频并提取关键点...")
    if use_multiprocessing:
        print(f"  使用 {num_workers} 个工作进程并行处理")

        # 使用多进程处理
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
        ) as executor:
            # 提交所有任务
            futures = {executor.submit(_worker_process_video, task): task[0] for task in tasks}

            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                result = future.result()
                if result["success"]:
                    video_id = result["video_id"]
                    split = result["split"]
                    if split not in split_data:
                        split = "train"

                    split_data[split][video_id] = {
                        "data": result["data"],
                        "mask": result.get("mask"),
                        "quality": result.get("quality", 0.0),
                        "length": result.get("length", 0),
                        "label": result["label"],
                        "video_name": result["video_name"],
                        "width": result["width"],
                        "height": result["height"],
                    }
                    processed_count += 1
                else:
                    if result.get("filtered"):
                        filtered_count += 1
                    else:
                        failed_count += 1
    else:
        # 单进程处理
        print("  使用单进程处理")
        print("\n[3.5/5] 初始化 MediaPipe 模型和预处理器...")
        extractor = KeypointExtractor()
        preprocess_helper = PreprocessHelper(max_frames=cfg.SEQUENCE.max_frames)

        for video_id, info, video_path in tqdm(videos_to_process, desc="Processing videos"):
            try:
                keypoints, valid_len, width, height, mask, quality = process_video(
                    video_path,
                    extractor,
                    frame_start=info.get("frame_start", 1),
                    frame_end=info.get("frame_end", -1),
                    preprocess_helper=preprocess_helper,
                )

                if keypoints is None or keypoints.shape[0] == 0:
                    failed_count += 1
                    continue

                if quality < cfg.PREPROCESS.min_valid_ratio_per_sample:
                    filtered_count += 1
                    continue

                split = info.get("split", "train")
                if split not in split_data:
                    split = "train"

                split_data[split][video_id] = {
                    "data": keypoints.astype(np.float32),
                    "mask": mask.astype(np.uint8) if mask is not None else None,
                    "quality": float(quality),
                    "length": valid_len,
                    "label": info["gloss"],
                    "video_name": f"{video_id}.mp4",
                    "width": width,
                    "height": height,
                }

                processed_count += 1

            except Exception as e:
                tqdm.write(f"  警告: 处理视频 {video_id} 时出错: {e}")
                failed_count += 1
                continue

        extractor.close()

    # 5. 写入 HDF5 文件
    print("\n[5/5] 写入 HDF5 文件...")

    # split 名称映射（匹配官方格式）
    split_name_map = {"train": "Train", "val": "Val", "test": "Test"}

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
    create_maplabels_json(maplabels_path, split_data, subset=args.subset)

    # 导出训练集统计量（仅 train split）供训练阶段标准化使用
    train_stats = compute_train_feature_stats(split_data.get("train", {}))
    if train_stats is not None:
        save_train_feature_stats(train_stats)

    # 6. 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"  成功处理: {processed_count} 个视频")
    print(f"  质量过滤: {filtered_count} 个视频")
    print(f"  处理失败: {failed_count} 个视频")
    print(f"  输出目录: {args.output_dir}/")


if __name__ == "__main__":
    # Windows 多进程支持需要此保护
    mp_process.freeze_support()
    main()
