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
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.system('')

# 确保能够以模块方式导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.config as cfg
from src.model.model_lstm import get_model
from src.data_process.preprocess_wlasl import KeypointExtractor


def load_label_map_inverse() -> dict:
    """读取并反转标签映射，返回 {id: label} 字典。"""
    with open(cfg.LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'id_to_label' in data:
        mapping = data['id_to_label']
    elif 'label_to_id' in data:
        mapping = data['label_to_id']
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


def load_chinese_font(size: int) -> ImageFont.FreeTypeFont:
    """按配置路径顺序加载可用的中文字体，失败则回退默认字体。"""
    for path in cfg.CHINESE_FONT_PATHS:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # 回退：默认字体英文可用，中文可能无法正常显示
    return ImageFont.load_default()


def draw_overlay_with_button(
    frame: np.ndarray,
    status_text: str,
    fps: float,
    font_main: ImageFont.FreeTypeFont,
    font_small: ImageFont.FreeTypeFont,
    button_text: str,
    button_size: Tuple[int, int],
    button_margin: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    使用 PIL 绘制中文叠加文本与右上角退出按钮。
    返回绘制后的 BGR 图像和按钮矩形 (x1, y1, x2, y2)。
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

    # 退出按钮（右上角）
    btn_w, btn_h = button_size
    margin_r, margin_t = button_margin
    x1 = w - btn_w - margin_r
    y1 = margin_t
    x2 = x1 + btn_w
    y2 = y1 + btn_h

    draw.rectangle([x1, y1, x2, y2], fill=(245, 245, 245), outline=(0, 0, 0), width=2)
    bbox = draw.textbbox((0, 0), button_text, font=font_small)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x_btn = x1 + (btn_w - text_w) // 2
    text_y_btn = y1 + (btn_h - text_h) // 2
    draw.text((text_x_btn, text_y_btn), button_text, font=font_small, fill=(0, 0, 0))

    rendered = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return rendered, (x1, y1, x2, y2)


def on_mouse(event: int, x: int, y: int, flags: int, params: Dict[str, Any]):
    """鼠标回调：检测是否点击退出按钮。"""
    if params is None:
        return
    rect = params.get("btn_rect")
    if event == cv2.EVENT_LBUTTONDOWN and rect:
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            params["quit"] = True


def prepare_sequence(frame_buffer: Deque[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将关键点序列转换为模型可接受的张量。

    Args:
        frame_buffer: 存放 (2, 135) 关键点的队列，长度不超过 cfg.MAX_FRAMES。

    Returns:
        inputs: 形状为 (1, cfg.MAX_FRAMES, cfg.INPUT_SIZE) 的张量。
        lengths: 形状为 (1,) 的张量，表示有效帧长度。
    """
    if not frame_buffer:
        raise ValueError("frame_buffer 为空，无法进行推理")

    data = np.stack(frame_buffer)  # (seq_len, 2, 135)
    seq_len = data.shape[0]

    # 展平特征 (Frames, 2, 135) -> (Frames, 270)
    data = data.reshape(seq_len, -1)

    # 截断或补零到 MAX_FRAMES
    if seq_len > cfg.MAX_FRAMES:
        data = data[-cfg.MAX_FRAMES:]
        valid_len = cfg.MAX_FRAMES
    elif seq_len < cfg.MAX_FRAMES:
        padding = np.zeros((cfg.MAX_FRAMES - seq_len, data.shape[1]), dtype=data.dtype)
        data = np.concatenate((data, padding), axis=0)
        valid_len = seq_len
    else:
        valid_len = seq_len

    inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, max_frames, input_size)
    lengths = torch.tensor([valid_len], dtype=torch.long)
    return inputs, lengths


def run_realtime_inference(camera_index: int | None = None) -> None:
    """
    使用摄像头与预训练模型进行实时手语分类。

    Args:
        camera_index: 摄像头索引，默认使用 cfg.CAMERA_INDEX。

    按下键盘 "q" 或点击右上角按钮退出。
    """
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.DEVICE == 'cuda' else 'cpu')
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

    # 窗口与鼠标回调（用于退出按钮）
    window_name = "Real-time Sign Prediction"
    mouse_state: Dict[str, Any] = {"btn_rect": None, "quit": False}
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    # 摄像头输入
    cam_idx = cfg.CAMERA_INDEX if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {cam_idx}")
        extractor.close()
        return

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

                frame_count += 1
                keypoints = extractor.extract_frame(frame)
                if keypoints is not None:
                    frame_buffer.append(keypoints)

                # 仅在存在有效关键点序列时推理
                if frame_buffer:
                    inputs, lengths = prepare_sequence(frame_buffer)
                    inputs = inputs.to(device)
                    lengths = lengths.to(device)

                    logits = model(inputs, lengths)
                    probs = F.softmax(logits, dim=1).squeeze(0)

                    top_prob, top_idx = torch.max(probs, dim=0)
                    pred_label = id_to_label.get(int(top_idx.item()), str(int(top_idx.item())))
                    last_result = (pred_label, float(top_prob.item()))

                # 叠加显示信息（中文）
                status_text = "检测中..." if last_result is None else f"预测：{last_result[0]} ({last_result[1]*100:.1f}%)"

                # FPS 估计
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

                overlay, btn_rect = draw_overlay_with_button(
                    frame,
                    status_text,
                    fps,
                    font_main,
                    font_small,
                    cfg.EXIT_BUTTON_TEXT,
                    cfg.EXIT_BUTTON_SIZE,
                    cfg.EXIT_BUTTON_MARGIN,
                )
                mouse_state["btn_rect"] = btn_rect

                cv2.imshow(window_name, overlay)

                # 退出条件：按键 q 或点击右上角按钮
                if (cv2.waitKey(1) & 0xFF == ord('q')) or mouse_state.get("quit"):
                    break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("已退出实时推理。")


if __name__ == "__main__":
    run_realtime_inference()
