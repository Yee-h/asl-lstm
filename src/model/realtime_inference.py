import sys
import os
import io
import json
import time
from collections import deque
from typing import Deque, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# 解决 Windows 终端编码与颜色支持问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")

# 确保能够以模块方式导入 src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import preprocess_keypoints
from src.data_process.preprocess_wlasl import KeypointExtractor


def load_label_map_inverse() -> dict:
    """读取并反转标签映射，返回 {id: label} 字典。"""
    with open(cfg.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "id_to_label" in data:
        mapping = data["id_to_label"]
    elif "label_to_id" in data:
        mapping = data["label_to_id"]
    else:
        mapping = data

    if not mapping:
        return {}

    k, v = next(iter(mapping.items()))
    if str(k).isdigit():
        return {int(key): val for key, val in mapping.items()}

    if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
        return {int(val): key for key, val in mapping.items()}

    return {}


def load_chinese_font(size: int):
    """按配置路径顺序加载可用的中文字体，失败则回退默认字体。"""
    for path in cfg.CHINESE_FONT_PATHS:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # 回退：默认字体英文可用，中文可能无法正常显示
    return ImageFont.load_default()


def draw_overlay_with_buttons(
    frame: np.ndarray,
    status_text: str,
    fps: float,
    font_main,
    font_small,
    show_skeleton: bool,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    使用 PIL 绘制中文叠加文本、退出按钮和骨骼显示切换按钮。
    返回绘制后的 BGR 图像、退出按钮矩形、骨骼切换按钮矩形。
    """
    w = frame.shape[1]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # 状态与 FPS 文本
    text_y1 = 18
    text_x = 20
    draw.text((text_x, text_y1), status_text, font=font_main, fill=(0, 255, 0))

    text_y2 = text_y1 + font_main.size + 8
    draw.text((text_x, text_y2), f"FPS：{fps:.1f}", font=font_small, fill=(255, 255, 0))

    margin_r, margin_t = cfg.EXIT_BUTTON_MARGIN

    # 退出按钮（右上角最右侧）
    exit_w, exit_h = cfg.EXIT_BUTTON_SIZE
    exit_x1 = w - exit_w - margin_r
    exit_y1 = margin_t
    exit_x2 = exit_x1 + exit_w
    exit_y2 = exit_y1 + exit_h

    draw.rectangle(
        [exit_x1, exit_y1, exit_x2, exit_y2],
        fill=(245, 245, 245),
        outline=(0, 0, 0),
        width=2,
    )
    bbox = draw.textbbox((0, 0), cfg.EXIT_BUTTON_TEXT, font=font_small)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x_btn = exit_x1 + (exit_w - text_w) // 2
    text_y_btn = exit_y1 + (exit_h - text_h) // 2
    draw.text(
        (text_x_btn, text_y_btn), cfg.EXIT_BUTTON_TEXT, font=font_small, fill=(0, 0, 0)
    )

    # 骨骼切换按钮（退出按钮左侧）
    skel_w, skel_h = cfg.SKELETON_BUTTON_SIZE
    skel_x2 = exit_x1 - cfg.SKELETON_BUTTON_GAP
    skel_x1 = skel_x2 - skel_w
    skel_y1 = margin_t
    skel_y2 = skel_y1 + skel_h

    skel_btn_text = (
        cfg.SKELETON_BUTTON_TEXT_ON if show_skeleton else cfg.SKELETON_BUTTON_TEXT_OFF
    )
    btn_fill = (200, 255, 200) if show_skeleton else (245, 245, 245)
    draw.rectangle(
        [skel_x1, skel_y1, skel_x2, skel_y2], fill=btn_fill, outline=(0, 0, 0), width=2
    )
    bbox_skel = draw.textbbox((0, 0), skel_btn_text, font=font_small)
    skel_text_w = bbox_skel[2] - bbox_skel[0]
    skel_text_h = bbox_skel[3] - bbox_skel[1]
    skel_text_x = skel_x1 + (skel_w - skel_text_w) // 2
    skel_text_y = skel_y1 + (skel_h - skel_text_h) // 2
    draw.text(
        (skel_text_x, skel_text_y), skel_btn_text, font=font_small, fill=(0, 0, 0)
    )

    rendered = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

    point_color = cfg.SKELETON_POINT_COLOR
    line_color = cfg.SKELETON_LINE_COLOR
    radius = cfg.SKELETON_POINT_RADIUS
    thickness = cfg.SKELETON_LINE_THICKNESS

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
    if params is None or event != cv2.EVENT_LBUTTONDOWN:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将关键点序列转换为模型可接受的张量。
    复用 dataloader.py 中的 preprocess_keypoints 函数。

    Args:
        frame_buffer: 存放 (2, 135) 关键点的队列，长度不超过 cfg.MAX_FRAMES。

    Returns:
        inputs: 形状为 (1, cfg.MAX_FRAMES, cfg.INPUT_SIZE) 的张量。
        lengths: 形状为 (1,) 的张量，表示有效帧长度。
    """
    if not frame_buffer:
        raise ValueError("frame_buffer 为空，无法进行推理")

    # 将队列堆叠为 numpy 数组: (seq_len, 2, 135)
    data = np.stack(frame_buffer)

    # 复用 dataloader 中的预处理函数
    data_tensor, valid_len = preprocess_keypoints(data, cfg.MAX_FRAMES)

    # 添加 batch 维度
    inputs = data_tensor.unsqueeze(0)  # (1, max_frames, input_size)
    lengths = torch.tensor([valid_len], dtype=torch.long)
    return inputs, lengths


def run_realtime_inference(camera_index: int | None = None) -> None:
    """
    使用摄像头与预训练模型进行实时手语分类。

    Args:
        camera_index: 摄像头索引，默认使用 cfg.CAMERA_INDEX。

    按下键盘 "q" 或点击右上角按钮退出。
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.DEVICE == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    # 加载标签映射
    id_to_label = load_label_map_inverse()
    if not id_to_label:
        print("警告: 标签映射为空，将直接输出类别 ID。")

    # 加载模型
    model = get_model(use_attention=cfg.USE_ATTENTION).to(device)
    model_path = cfg.TEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"错误: 未找到预训练模型文件: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_name = "BiLSTM+Attention" if cfg.USE_ATTENTION else "BiLSTM"
    print(f"已加载模型: {model_name} -> {os.path.basename(model_path)}")

    # 初始化关键点提取器
    extractor = KeypointExtractor()

    # 加载中文字体
    font_main = load_chinese_font(cfg.UI_FONT_SIZE)
    font_small = load_chinese_font(cfg.UI_FONT_SMALL_SIZE)

    # 窗口与鼠标回调（用于退出按钮和骨骼切换按钮）
    window_name = "Real-time Sign Prediction"
    mouse_state: Dict[str, Any] = {
        "exit_rect": None,
        "skel_rect": None,
        "quit": False,
        "show_skeleton": False,
    }
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    # 摄像头输入
    cam_idx = cfg.CAMERA_INDEX if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_idx)

    # 设置摄像头参数以提高帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减小缓冲区降低延迟

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {cam_idx}")
        extractor.close()
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头设置: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    frame_buffer: Deque[np.ndarray] = deque(maxlen=cfg.MAX_FRAMES)
    last_result: Tuple[str, float] | None = None

    print('\n实时推理已启动，按 "q" 退出。\n')
    start_time = time.time()
    frame_count = 0

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("读取摄像头失败，即将退出。")
                    break

                # 水平翻转画面，使显示更自然（镜像效果）
                frame = cv2.flip(frame, 1)

                frame_count += 1

                # 每帧都提取关键点用于显示，但推理可以间隔进行
                keypoints, valid_mask = extractor.extract_frame(frame)

                # 检查关键点是否有效（至少有一定数量的非零点才认为检测成功）
                keypoints_valid = False
                if keypoints is not None and valid_mask is not None:
                    valid_count = int(np.sum(valid_mask))
                    keypoints_valid = valid_count >= cfg.MIN_VALID_KEYPOINTS_PER_FRAME

                # 只有检测到有效骨骼点时才更新 buffer
                if keypoints_valid and keypoints is not None:
                    frame_buffer.append(keypoints)

                    # 按间隔进行模型推理以提高帧率
                    if frame_buffer and (frame_count % cfg.INFERENCE_INTERVAL == 0):
                        inputs, lengths = prepare_sequence(frame_buffer)
                        inputs = inputs.to(device)
                        lengths = lengths.to(device)

                        logits = model(inputs, lengths)
                        probs = F.softmax(logits, dim=1).squeeze(0)

                        top_prob, top_idx = torch.max(probs, dim=0)
                        pred_label = id_to_label.get(
                            int(top_idx.item()), str(int(top_idx.item()))
                        )
                        last_result = (pred_label, float(top_prob.item()))
                else:
                    # 未检测到有效骨骼点时清空缓冲区和预测结果
                    frame_buffer.clear()
                    last_result = None
                    keypoints = None  # 确保骨骼显示也不渲染

                # 叠加显示信息（中文）
                status_text = (
                    "未检测到人体..."
                    if last_result is None
                    else f"预测：{last_result[0]} ({last_result[1] * 100:.1f}%)"
                )

                # FPS 估计
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

                # 如果开启骨骼显示，先绘制骨骼点
                display_frame = frame.copy()
                if mouse_state.get("show_skeleton") and keypoints is not None:
                    display_frame = draw_skeleton(display_frame, keypoints)

                overlay, exit_rect, skel_rect = draw_overlay_with_buttons(
                    display_frame,
                    status_text,
                    fps,
                    font_main,
                    font_small,
                    mouse_state.get("show_skeleton", False),
                )
                mouse_state["exit_rect"] = exit_rect
                mouse_state["skel_rect"] = skel_rect

                cv2.imshow(window_name, overlay)

                # 退出条件：按键 q 或点击右上角按钮
                if (cv2.waitKey(1) & 0xFF == ord("q")) or mouse_state.get("quit"):
                    break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("已退出实时推理。")


if __name__ == "__main__":
    run_realtime_inference()
