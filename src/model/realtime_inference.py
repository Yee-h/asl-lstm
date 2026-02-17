import sys
import os
import io
import time
from collections import deque
from typing import Deque, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")


# 确保能够以模块方式导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.config as cfg
from src.core.labels import load_id_to_label_map_compat
from src.model.model_lstm import get_model
from src.model.dataloader import (
    preprocess_keypoints,
    load_feature_stats,
)
from src.data_process.preprocess_wlasl import KeypointExtractor, PreprocessHelper


def load_chinese_font(size: int):
    """按配置路径顺序加载可用的中文字体，失败则回退默认字体。"""
    for path in cfg.UI.chinese_font_paths:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # 回退：默认字体英文可用，中文可能无法正常显示
    return ImageFont.load_default()


def draw_modern_ui(
    frame: np.ndarray,
    last_result: Tuple[str, float] | None,
    fps: float,
    font_main,
    font_small,
    show_skeleton: bool,
    mouse_pos: Tuple[int, int] | None,
    status_text: str | None = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    绘制现代化、极简主义风格的 UI 界面。
    """
    h, w = frame.shape[:2]

    # 转换为 PIL RGBA 进行透明度绘制
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ==========================
    # 1. 顶部信息栏 (FPS)
    # ==========================
    fps_text = f"FPS: {fps:.1f}"
    fps_bbox = draw.textbbox((0, 0), fps_text, font=font_small)
    fps_w = fps_bbox[2] - fps_bbox[0] + 24
    fps_h = fps_bbox[3] - fps_bbox[1] + 12
    fps_x = 20
    fps_y = 20

    # 绘制 FPS 背景胶囊 (半透明黑 + 亮色文字)
    draw.rounded_rectangle(
        [fps_x, fps_y, fps_x + fps_w, fps_y + fps_h],
        radius=12,
        fill=(30, 30, 30, 160),
        outline=(255, 255, 255, 30),
        width=1,
    )
    draw.text((fps_x + 12, fps_y + 6), fps_text, font=font_small, fill=(220, 220, 220, 255))

    # ==========================
    # 2. 交互按钮区域 (右上角)
    # ==========================
    margin_r, margin_t = cfg.UI.exit_button_margin
    btn_height = 44  # 稍微加大高度
    btn_radius = 12  # 圆角半径

    # --- 退出按钮 (Exit) ---
    exit_text = cfg.UI.exit_button_text
    exit_bbox = draw.textbbox((0, 0), exit_text, font=font_small)
    exit_text_w = exit_bbox[2] - exit_bbox[0]
    exit_w = max(90, exit_text_w + 40)

    exit_x1 = w - margin_r - exit_w
    exit_y1 = margin_t
    exit_x2 = exit_x1 + exit_w
    exit_y2 = exit_y1 + btn_height

    # 检测 Hover
    is_hover_exit = False
    if mouse_pos:
        mx, my = mouse_pos
        if exit_x1 <= mx <= exit_x2 and exit_y1 <= my <= exit_y2:
            is_hover_exit = True

    # 红色系渐变效果 (模拟)
    if is_hover_exit:
        exit_fill = (255, 60, 60, 230)
        exit_outline = (255, 200, 200, 180)
    else:
        exit_fill = (40, 40, 40, 160)
        exit_outline = (255, 255, 255, 40)

    draw.rounded_rectangle(
        [exit_x1, exit_y1, exit_x2, exit_y2],
        radius=btn_radius,
        fill=exit_fill,
        outline=exit_outline,
        width=1,
    )

    # 居中文字
    draw.text(
        (
            exit_x1 + (exit_w - exit_text_w) // 2,
            exit_y1 + (btn_height - (exit_bbox[3] - exit_bbox[1])) // 2 - 2,
        ),
        exit_text,
        font=font_small,
        fill=(255, 255, 255, 255),
    )

    # --- 骨骼切换按钮 (Skeleton) ---
    skel_text = "骨骼: 开" if show_skeleton else "骨骼: 关"
    skel_bbox = draw.textbbox((0, 0), skel_text, font=font_small)
    skel_text_w = skel_bbox[2] - skel_bbox[0]
    skel_w = max(110, skel_text_w + 40)

    skel_x2 = exit_x1 - 15  # 间距
    skel_x1 = skel_x2 - skel_w
    skel_y1 = margin_t
    skel_y2 = skel_y1 + btn_height

    # 检测 Hover
    is_hover_skel = False
    if mouse_pos:
        mx, my = mouse_pos
        if skel_x1 <= mx <= skel_x2 and skel_y1 <= my <= skel_y2:
            is_hover_skel = True

    # 绿色系 (开) / 灰色系 (关)
    if show_skeleton:
        if is_hover_skel:
            skel_fill = (0, 200, 120, 230)
            skel_outline = (200, 255, 200, 180)
        else:
            skel_fill = (0, 160, 90, 200)
            skel_outline = (255, 255, 255, 50)
    else:
        if is_hover_skel:
            skel_fill = (70, 70, 70, 230)
            skel_outline = (255, 255, 255, 100)
        else:
            skel_fill = (40, 40, 40, 160)
            skel_outline = (255, 255, 255, 40)

    draw.rounded_rectangle(
        [skel_x1, skel_y1, skel_x2, skel_y2],
        radius=btn_radius,
        fill=skel_fill,
        outline=skel_outline,
        width=1,
    )

    draw.text(
        (
            skel_x1 + (skel_w - skel_text_w) // 2,
            skel_y1 + (btn_height - (skel_bbox[3] - skel_bbox[1])) // 2 - 2,
        ),
        skel_text,
        font=font_small,
        fill=(255, 255, 255, 255),
    )

    # ==========================
    # 3. 底部预测结果展示区 (卡片式)
    # ==========================
    card_h = 100
    bottom_margin = 40

    if last_result:
        label, prob = last_result
        prob_percent = int(prob * 100)

        # 标签文字
        label_bbox = draw.textbbox((0, 0), label, font=font_main)
        label_w = label_bbox[2] - label_bbox[0]

        # 概率文字
        prob_text = f"{prob_percent}%"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=font_main)
        prob_w = prob_bbox[2] - prob_bbox[0]

        # 布局计算
        content_gap = 20
        min_card_w = 360
        total_content_w = max(min_card_w, label_w + content_gap + prob_w + 60)

        card_w = total_content_w
        card_x1 = (w - card_w) // 2
        card_y1 = h - card_h - bottom_margin
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        # 磨砂玻璃背景
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=(20, 20, 20, 220),
            outline=(255, 255, 255, 25),
            width=1,
        )

        # 顶部：标签 和 概率数值
        # 左侧放 Label, 右侧放 概率
        text_y_base = card_y1 + 25

        draw.text(
            (card_x1 + 30, text_y_base),
            label,
            font=font_main,
            fill=(255, 255, 255, 255),
        )

        # 概率颜色
        if prob > 0.8:
            prob_color = (100, 255, 100, 255)
        elif prob > 0.5:
            prob_color = (255, 200, 50, 255)
        else:
            prob_color = (255, 80, 80, 255)

        draw.text(
            (card_x2 - 30 - prob_w, text_y_base),
            prob_text,
            font=font_main,
            fill=prob_color,
        )

        # 底部：进度条
        bar_x1 = card_x1 + 30
        bar_x2 = card_x2 - 30
        bar_y1 = card_y2 - 30
        bar_y2 = bar_y1 + 8
        bar_full_w = bar_x2 - bar_x1

        # 进度条背景
        draw.rounded_rectangle(
            [bar_x1, bar_y1, bar_x2, bar_y2],
            radius=4,
            fill=(60, 60, 60, 255),
        )

        # 进度条前景
        fill_w = int(bar_full_w * prob)
        if fill_w > 0:
            draw.rounded_rectangle(
                [bar_x1, bar_y1, bar_x1 + fill_w, bar_y2],
                radius=4,
                fill=prob_color,
            )

    else:
        # 待机状态
        hint_text = status_text if status_text else "等待手语动作..."
        hint_bbox = draw.textbbox((0, 0), hint_text, font=font_main)
        hint_w = hint_bbox[2] - hint_bbox[0]

        card_w = max(320, hint_w + 80)
        card_h = 80
        card_x1 = (w - card_w) // 2
        card_y1 = h - card_h - bottom_margin
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=(30, 30, 30, 200),
            outline=(255, 255, 255, 20),
            width=1,
        )

        # 居中显示提示
        draw.text(
            (
                card_x1 + (card_w - hint_w) // 2,
                card_y1 + (card_h - (hint_bbox[3] - hint_bbox[1])) // 2 - 4,
            ),
            hint_text,
            font=font_main,
            fill=(180, 180, 180, 255),
        )

    # 混合图层
    out = Image.alpha_composite(image, overlay)
    rendered = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)

    return (
        rendered,
        (exit_x1, exit_y1, exit_x2, exit_y2),
        (skel_x1, skel_y1, skel_x2, skel_y2),
    )


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    在画面上绘制 MediaPipe 提取的骨骼关键点。

    Args:
        frame: BGR 图像
        keypoints: 形状为 (2, 135) 的归一化坐标，[0] 是 x，[1] 是 y

    Returns:
        绘制骨骼后的图像
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # 135 关键点布局：Body 25 + Left Hand 21 + Right Hand 21 + Face 68
    # Body: 0-24, Left Hand: 25-45, Right Hand: 46-66, Face: 67-134

    # Body 25 连接关系 (OpenPose Body 25 格式)
    body_connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # 右臂
        (1, 5),
        (5, 6),
        (6, 7),  # 左臂
        (1, 8),
        (8, 9),
        (9, 10),
        (10, 11),  # 右腿
        (8, 12),
        (12, 13),
        (13, 14),  # 左腿
        (0, 15),
        (0, 16),
        (15, 17),
        (16, 18),  # 面部
    ]

    # Hand 21 连接关系
    hand_connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # 拇指
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # 食指
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # 中指
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # 无名指
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # 小指
        (5, 9),
        (9, 13),
        (13, 17),  # 掌心连接
    ]

    point_color = cfg.UI.skeleton_point_color
    line_color = cfg.UI.skeleton_line_color
    radius = cfg.UI.skeleton_point_radius
    thickness = cfg.UI.skeleton_line_thickness

    def get_point(idx: int) -> Tuple[int, int] | None:
        """获取关键点像素坐标，无效点返回 None"""
        x_norm, y_norm = keypoints[0, idx], keypoints[1, idx]
        if x_norm == 0 and y_norm == 0:
            return None
        return int(x_norm * w), int(y_norm * h)

    # 绘制 Body 连接线
    for i, j in body_connections:
        p1, p2 = get_point(i), get_point(j)
        if p1 and p2:
            cv2.line(result, p1, p2, line_color, thickness)

    # 绘制左手连接线 (索引偏移 25)
    for i, j in hand_connections:
        p1, p2 = get_point(i + 25), get_point(j + 25)
        if p1 and p2:
            cv2.line(result, p1, p2, line_color, thickness)

    # 绘制右手连接线 (索引偏移 46)
    for i, j in hand_connections:
        p1, p2 = get_point(i + 46), get_point(j + 46)
        if p1 and p2:
            cv2.line(result, p1, p2, line_color, thickness)

    # 绘制所有关键点
    for idx in range(135):
        pt = get_point(idx)
        if pt:
            cv2.circle(result, pt, radius, point_color, -1)

    return result


def on_mouse(event: int, x: int, y: int, flags: int, params: Any):
    """鼠标回调：检测是否点击退出按钮或骨骼切换按钮。"""
    if params is None:
        return

    # 记录鼠标位置用于 Hover 效果
    if event == cv2.EVENT_MOUSEMOVE:
        params["mouse_pos"] = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # 检测退出按钮
    exit_rect = params.get("exit_rect")
    if exit_rect:
        x1, y1, x2, y2 = exit_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            params["quit"] = True
            return

    # 检测骨骼切换按钮
    skel_rect = params.get("skel_rect")
    if skel_rect:
        x1, y1, x2, y2 = skel_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            params["show_skeleton"] = not params.get("show_skeleton", False)


def prepare_sequence(
    frame_buffer: Deque[np.ndarray],
    helper: PreprocessHelper,
    stats: Dict[str, Any] | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将关键点序列转换为模型可接受的张量。
    复用 dataloader.py 和 preprocess_wlasl.py 中的方法。

    Args:
        frame_buffer: 存放 (2, 135) 关键点的队列，长度不超过 cfg.SEQUENCE.max_frames。
        helper: 预处理助手，负责几何归一化。
        stats: 统计特征，负责 Z-Score 标准化。

    Returns:
        inputs: 形状为 (1, max_frames, input_size) 的张量。
        lengths: 形状为 (1,) 的张量，表示有效帧长度。
    """
    if not frame_buffer:
        raise ValueError("frame_buffer 为空，无法进行推理")

    # 将队列堆叠为 numpy 数组: (seq_len, 2, 135)
    raw_data = np.stack(frame_buffer)
    T, C, V = raw_data.shape

    # 创建掩码 (T, V)
    raw_mask = ((raw_data[:, 0, :] != 0) | (raw_data[:, 1, :] != 0)).astype(np.uint8)

    # 1. 应用几何归一化 (PreprocessHelper)
    # 返回: (T, C, V), valid_len, mask, quality
    # 这包括插值、平滑、肩轴对齐、尺度归一化、速度/加速度特征提取、填充/截断
    processed_data, valid_len, processed_mask, _ = helper.process_video_sequence(raw_data, raw_mask)

    # 2. 应用标准化和张量转换 (preprocess_keypoints)
    # 这包括 Z-Score 标准化 (如果 stats 存在) 和转 Tensor
    data_tensor, valid_len = preprocess_keypoints(
        processed_data,
        max_frames=cfg.SEQUENCE.max_frames,
        valid_len=valid_len,
        stats=stats,
        standardize=cfg.PREPROCESS.enable_standardize and stats is not None,
        mask=processed_mask,
    )

    # 添加 batch 维度
    inputs = data_tensor.unsqueeze(0)  # (1, max_frames, input_size)
    lengths = torch.tensor([valid_len], dtype=torch.long)
    return inputs, lengths


def run_realtime_inference(camera_index: int | str | None = None) -> None:
    """
    使用摄像头或视频文件与预训练模型进行实时手语分类。

    Args:
        camera_index: 摄像头索引(int)或视频文件路径(str)，默认使用 cfg.INFERENCE.camera_index。

    按下键盘 "q" 或点击右上角按钮退出。
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    # 加载标签映射
    id_to_label = load_id_to_label_map_compat(cfg.PATHS.label_map_path)
    if not id_to_label:
        print("警告: 标签映射为空，将直接输出类别 ID。")

    # 加载模型
    model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
    model_path = cfg.PATHS.test_model_path
    if not os.path.exists(model_path):
        print(f"错误: 未找到预训练模型文件: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_name = "BiLSTM+Attention" if cfg.MODEL.use_attention else "BiLSTM"
    print(f"已加载模型: {model_name} -> {os.path.basename(model_path)}")

    # 初始化关键点提取器 (启用 VIDEO 模式以提升性能)
    extractor = KeypointExtractor(use_video_mode=True)

    # 初始化预处理助手 (负责几何归一化)
    preprocess_helper = PreprocessHelper(max_frames=cfg.SEQUENCE.max_frames)

    # 加载训练集统计量 (负责 Z-Score 标准化)
    stats = None
    if cfg.PREPROCESS.enable_standardize:
        if os.path.exists(cfg.PREPROCESS.feature_stats_path):
            stats = load_feature_stats(cfg.PREPROCESS.feature_stats_path)
            print(f"已加载标准化统计量: {cfg.PREPROCESS.feature_stats_path}")
        else:
            print("警告: 启用了标准化但未找到统计量文件，推理时将跳过标准化。")

    # 加载中文字体
    font_main = load_chinese_font(cfg.UI.font_size)
    font_small = load_chinese_font(cfg.UI.font_small_size)

    # 窗口与鼠标回调（用于退出按钮和骨骼切换按钮）
    window_name = "Real-time Sign Prediction"
    mouse_state: Dict[str, Any] = {
        "exit_rect": None,
        "skel_rect": None,
        "quit": False,
        "show_skeleton": False,
        "mouse_pos": None,
    }
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    # 摄像头输入 (支持视频文件)
    cam_source = cfg.INFERENCE.camera_index if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_source)

    # 仅当输入源为整数（摄像头索引）时设置摄像头参数
    if isinstance(cam_source, int):
        # 设置摄像头参数以提高帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.INFERENCE.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.INFERENCE.camera_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.INFERENCE.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减小缓冲区降低延迟
    else:
        print(f"输入源为视频文件: {cam_source}，跳过摄像头参数设置。")

    if not cap.isOpened():
        print(f"错误: 无法打开输入源 {cam_source}")
        extractor.close()
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头设置: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    frame_buffer: Deque[np.ndarray] = deque(maxlen=cfg.SEQUENCE.max_frames)
    last_result: Tuple[str, float] | None = None

    print('\n实时推理已启动，按 "q" 退出。\n')
    start_time = time.time()
    frame_count = 0
    status_text = "初始化..."

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("读取摄像头失败，即将退出。")
                    break

                # 水平翻转画面，使显示更自然（镜像效果）
                # 仅对摄像头输入进行翻转，视频文件保持原样
                if isinstance(cam_source, int):
                    frame = cv2.flip(frame, 1)

                frame_count += 1
                timestamp_ms = int(time.time() * 1000)

                # 使用优化后的提取方法：优先检测手部
                keypoints, valid_mask, has_hands = extractor.extract_frame_optimized(
                    frame, timestamp_ms
                )

                if not has_hands:
                    # 未检测到手部，清空缓冲区（或保持不变？建议清空以避免拼接错误动作）
                    # frame_buffer.clear() # 可选：是否清空取决于交互设计，这里选择不清空但暂停推理
                    last_result = None
                    # 更新状态文本
                    status_text = "未检测到手部骨骼点信息"
                    keypoints = None  # 不显示旧骨骼
                else:
                    # 检查关键点是否有效（至少有一定数量的非零点才认为检测成功）
                    # extract_frame_optimized 已经保证了 hand_landmarks 存在，这里再做一次数量检查
                    valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
                    keypoints_valid = valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame

                    # 只有检测到有效骨骼点时才更新 buffer
                    if keypoints_valid and keypoints is not None:
                        frame_buffer.append(keypoints)

                        # 按间隔进行模型推理以提高帧率
                        if frame_buffer and (frame_count % cfg.INFERENCE.inference_interval == 0):
                            try:
                                inputs, lengths = prepare_sequence(
                                    frame_buffer, preprocess_helper, stats
                                )
                                inputs = inputs.to(device)
                                lengths = lengths.to(device)

                                logits = model(inputs, lengths)
                                probs = F.softmax(logits, dim=1).squeeze(0)

                                top_prob, top_idx = torch.max(probs, dim=0)
                                pred_label = id_to_label.get(
                                    int(top_idx.item()), str(int(top_idx.item()))
                                )
                                last_result = (pred_label, float(top_prob.item()))
                            except Exception:
                                # 仅在调试时打印详细错误，避免刷屏
                                # print(f"推理错误: {e}")
                                last_result = None
                    else:
                        # 有手但关键点数量不足（极少情况）
                        status_text = "关键点数量不足"

                # 叠加显示信息（中文）
                # status_text 在上面已经处理了 "无手" 的情况，这里处理有结果的情况
                if has_hands and last_result:
                    pass  # UI draw_modern_ui 会处理 last_result
                elif has_hands and not last_result:
                    status_text = "正在分析..."  # 或者保持上一帧状态

                # FPS 估计
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

                # 如果开启骨骼显示，先绘制骨骼点
                display_frame = frame.copy()
                if mouse_state.get("show_skeleton") and keypoints is not None:
                    display_frame = draw_skeleton(display_frame, keypoints)

                overlay, exit_rect, skel_rect = draw_modern_ui(
                    display_frame,
                    last_result,
                    fps,
                    font_main,
                    font_small,
                    mouse_state.get("show_skeleton", False),
                    mouse_state.get("mouse_pos"),
                    status_text=status_text,
                )
                mouse_state["exit_rect"] = exit_rect
                mouse_state["skel_rect"] = skel_rect

                # 检查窗口是否被用户关闭 (X 按钮)
                # 必须在 imshow 之前检查，否则 imshow 会自动重建窗口导致无法检测关闭事件
                # 同时也解决了重建窗口后鼠标回调失效导致按钮不可用的问题
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("窗口被用户关闭。")
                    break

                cv2.imshow(window_name, overlay)

                # 退出条件：按键 q 或点击右上角按钮
                if (cv2.waitKey(1) & 0xFF == ord("q")) or mouse_state.get("quit"):
                    break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("已退出实时推理。")


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="实时手语推理")
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="摄像头索引 (默认使用 config.py 中的配置)",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="视频文件路径 (如果不使用摄像头)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _configure_windows_console()
    args = parse_args()

    # 如果指定了视频文件，临时修改 config 中的 camera_index 为视频路径
    # 注意：cv2.VideoCapture 支持整数索引或文件路径字符串
    source = args.camera if args.camera is not None else cfg.INFERENCE.camera_index
    if args.video:
        if os.path.exists(args.video):
            source = args.video
            print(f"将使用视频文件进行推理: {source}")
        else:
            print(f"错误: 视频文件不存在: {args.video}")
            sys.exit(1)

    run_realtime_inference(camera_index=source)
