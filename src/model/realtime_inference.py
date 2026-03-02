import sys
import os
import io
import time
import threading
from collections import deque
from typing import Deque, Tuple, Dict, Any, List

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
from src.core.labels import load_id_to_label_map
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
) -> Tuple[
    np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[int, int, int, int]
]:
    """
    绘制现代化、极简主义风格的 UI 界面。

    Returns:
        (rendered_frame, exit_rect, skel_rect, import_rect)
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

    # --- 导入视频按钮 (Import) ---
    import_text = cfg.UI.import_button_text
    import_bbox = draw.textbbox((0, 0), import_text, font=font_small)
    import_text_w = import_bbox[2] - import_bbox[0]
    import_w = max(110, import_text_w + 40)

    import_x2 = skel_x1 - 15  # 在骨骼按钮左侧
    import_x1 = import_x2 - import_w
    import_y1 = margin_t
    import_y2 = import_y1 + btn_height

    # 检测 Hover
    is_hover_import = False
    if mouse_pos:
        mx, my = mouse_pos
        if import_x1 <= mx <= import_x2 and import_y1 <= my <= import_y2:
            is_hover_import = True

    # 蓝色系
    if is_hover_import:
        import_fill = (60, 130, 255, 230)
        import_outline = (180, 210, 255, 180)
    else:
        import_fill = (40, 90, 180, 200)
        import_outline = (255, 255, 255, 50)

    draw.rounded_rectangle(
        [import_x1, import_y1, import_x2, import_y2],
        radius=btn_radius,
        fill=import_fill,
        outline=import_outline,
        width=1,
    )

    draw.text(
        (
            import_x1 + (import_w - import_text_w) // 2,
            import_y1 + (btn_height - (import_bbox[3] - import_bbox[1])) // 2 - 2,
        ),
        import_text,
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
        (import_x1, import_y1, import_x2, import_y2),
    )


def draw_offline_ui(
    video_frame: "np.ndarray | None",
    result: "Tuple[str, float] | None",
    status_text: str,
    font_main,
    font_small,
    mouse_pos: "Tuple[int, int] | None",
    show_skeleton: bool,
    target_size: "Tuple[int, int]",
) -> "Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int,int,int], Tuple[int,int,int,int] | None]":
    """
    绘制离线推理界面。

    布局：黑色背景 + 中央视频帧 + 右上角按钮（退出离线推理、骨骼切换）
          + 底部结果卡（带 × 关闭按钮）或状态文字卡。

    Returns:
        (rendered_frame, exit_offline_rect, skel_rect, close_result_rect)
        close_result_rect 在无结果时为 None。
    """
    tw, th = target_size

    # 黑色背景画布
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)

    # --- 中央视频帧 ---
    if video_frame is not None:
        vf_h, vf_w = video_frame.shape[:2]
        max_vw = int(tw * 0.85)
        max_vh = int(th * 0.72)
        scale = min(max_vw / max(vf_w, 1), max_vh / max(vf_h, 1))
        disp_w = max(1, int(vf_w * scale))
        disp_h = max(1, int(vf_h * scale))
        resized = cv2.resize(video_frame, (disp_w, disp_h))
        vx = (tw - disp_w) // 2
        vy = max(0, (th - disp_h) // 2 - 20)
        vy_end = min(th, vy + disp_h)
        vx_end = min(tw, vx + disp_w)
        canvas[vy:vy_end, vx:vx_end] = resized[: vy_end - vy, : vx_end - vx]
        cv2.rectangle(canvas, (vx - 1, vy - 1), (vx_end, vy_end), (255, 255, 255), 1)

    # --- PIL 叠加层 ---
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    _margin = cfg.UI.exit_button_margin
    margin_r: int = int(_margin[0])
    margin_t: int = int(_margin[1])
    btn_height = 44
    btn_radius = 12

    # --- 退出离线推理按钮（红色，最右）---
    exit_text = "退出离线推理"
    exit_bbox = draw.textbbox((0, 0), exit_text, font=font_small)
    exit_text_w: int = int(exit_bbox[2] - exit_bbox[0])
    exit_w: int = max(130, exit_text_w + 40)
    exit_x1: int = tw - margin_r - exit_w
    exit_y1: int = margin_t
    exit_x2: int = exit_x1 + exit_w
    exit_y2: int = exit_y1 + btn_height

    is_hover_exit = False
    if mouse_pos:
        mx, my = mouse_pos
        if exit_x1 <= mx <= exit_x2 and exit_y1 <= my <= exit_y2:
            is_hover_exit = True

    exit_fill = (255, 60, 60, 230) if is_hover_exit else (40, 40, 40, 160)
    exit_outline = (255, 200, 200, 180) if is_hover_exit else (255, 255, 255, 40)
    draw.rounded_rectangle(
        [exit_x1, exit_y1, exit_x2, exit_y2],
        radius=btn_radius,
        fill=exit_fill,
        outline=exit_outline,
        width=1,
    )
    draw.text(
        (
            exit_x1 + (exit_w - exit_text_w) // 2,
            exit_y1 + (btn_height - (exit_bbox[3] - exit_bbox[1])) // 2 - 2,
        ),
        exit_text,
        font=font_small,
        fill=(255, 255, 255, 255),
    )

    # --- 骨骼切换按钮（退出按钮左侧）---
    skel_text = "骨骼: 开" if show_skeleton else "骨骼: 关"
    skel_bbox = draw.textbbox((0, 0), skel_text, font=font_small)
    skel_text_w: int = int(skel_bbox[2] - skel_bbox[0])
    skel_w: int = max(110, skel_text_w + 40)
    skel_x2: int = exit_x1 - 15
    skel_x1: int = skel_x2 - skel_w
    skel_y1: int = margin_t
    skel_y2: int = skel_y1 + btn_height

    is_hover_skel = False
    if mouse_pos:
        mx, my = mouse_pos
        if skel_x1 <= mx <= skel_x2 and skel_y1 <= my <= skel_y2:
            is_hover_skel = True

    if show_skeleton:
        skel_fill = (0, 200, 120, 230) if is_hover_skel else (0, 160, 90, 200)
        skel_outline = (200, 255, 200, 180) if is_hover_skel else (255, 255, 255, 50)
    else:
        skel_fill = (70, 70, 70, 230) if is_hover_skel else (40, 40, 40, 160)
        skel_outline = (255, 255, 255, 100) if is_hover_skel else (255, 255, 255, 40)

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

    # --- 底部结果卡 或 状态文字卡 ---
    card_h = 100
    bottom_margin = 40
    close_result_rect = None

    if result:
        label, prob = result
        prob_percent = int(prob * 100)

        label_bbox = draw.textbbox((0, 0), label, font=font_main)
        label_w: int = int(label_bbox[2] - label_bbox[0])
        prob_text = f"{prob_percent}%"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=font_main)
        prob_w: int = int(prob_bbox[2] - prob_bbox[0])

        content_gap = 20
        min_card_w = 360
        card_w: int = max(min_card_w, label_w + content_gap + prob_w + 60)
        card_x1: int = (tw - card_w) // 2
        card_y1: int = th - card_h - bottom_margin
        card_x2: int = card_x1 + card_w
        card_y2: int = card_y1 + card_h

        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=(20, 20, 20, 220),
            outline=(255, 255, 255, 25),
            width=1,
        )

        text_y_base = card_y1 + 25
        draw.text((card_x1 + 30, text_y_base), label, font=font_main, fill=(255, 255, 255, 255))

        if prob > 0.8:
            prob_color = (100, 255, 100, 255)
        elif prob > 0.5:
            prob_color = (255, 200, 50, 255)
        else:
            prob_color = (255, 80, 80, 255)

        draw.text((card_x2 - 30 - prob_w, text_y_base), prob_text, font=font_main, fill=prob_color)

        bar_x1 = card_x1 + 30
        bar_x2 = card_x2 - 30
        bar_y1 = card_y2 - 30
        bar_y2 = bar_y1 + 8
        draw.rounded_rectangle([bar_x1, bar_y1, bar_x2, bar_y2], radius=4, fill=(60, 60, 60, 255))
        fill_w = int((bar_x2 - bar_x1) * prob)
        if fill_w > 0:
            draw.rounded_rectangle(
                [bar_x1, bar_y1, bar_x1 + fill_w, bar_y2], radius=4, fill=prob_color
            )

        # × 关闭按钮（结果卡右上角悬浮圆圈）
        close_size = 32
        close_x1: int = card_x2 - close_size // 2
        close_y1: int = card_y1 - close_size // 2
        close_x2: int = close_x1 + close_size
        close_y2: int = close_y1 + close_size

        is_hover_close = False
        if mouse_pos:
            mx, my = mouse_pos
            if close_x1 <= mx <= close_x2 and close_y1 <= my <= close_y2:
                is_hover_close = True

        close_fill = (255, 80, 80, 230) if is_hover_close else (160, 50, 50, 200)
        draw.ellipse([close_x1, close_y1, close_x2, close_y2], fill=close_fill)
        x_text = "×"
        x_bbox = draw.textbbox((0, 0), x_text, font=font_small)
        draw.text(
            (
                close_x1 + (close_size - (x_bbox[2] - x_bbox[0])) // 2,
                close_y1 + (close_size - (x_bbox[3] - x_bbox[1])) // 2 - 2,
            ),
            x_text,
            font=font_small,
            fill=(255, 255, 255, 255),
        )
        close_result_rect = (close_x1, close_y1, close_x2, close_y2)

    else:
        # 状态文字卡（待机 / 推理中 / 错误）
        hint_text = status_text if status_text else "请点击右上角导入视频..."
        hint_bbox = draw.textbbox((0, 0), hint_text, font=font_main)
        hint_w: int = int(hint_bbox[2] - hint_bbox[0])
        card_w = max(320, hint_w + 80)
        card_h_s = 80
        cx1 = (tw - card_w) // 2
        cy1 = th - card_h_s - bottom_margin
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h_s
        draw.rounded_rectangle(
            [cx1, cy1, cx2, cy2],
            radius=20,
            fill=(30, 30, 30, 200),
            outline=(255, 255, 255, 20),
            width=1,
        )
        draw.text(
            (
                cx1 + (card_w - hint_w) // 2,
                cy1 + (card_h_s - (hint_bbox[3] - hint_bbox[1])) // 2 - 4,
            ),
            hint_text,
            font=font_main,
            fill=(180, 180, 180, 255),
        )

    out = Image.alpha_composite(image, overlay)
    rendered = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)

    return (
        rendered,
        (exit_x1, exit_y1, exit_x2, exit_y2),
        (skel_x1, skel_y1, skel_x2, skel_y2),
        close_result_rect,
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
    """鼠标回调：检测是否点击退出按钮、骨骼切换按钮或导入视频按钮。"""
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
            return

    # 检测导入视频按钮（仅当未处于视频处理中时响应）
    import_rect = params.get("import_rect")
    if import_rect and not params.get("importing", False):
        x1, y1, x2, y2 = import_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            params["import_video"] = True


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


def _load_ensemble_models(device: torch.device) -> List[torch.nn.Module]:
    """
    加载集成推理所需的多个模型。

    优先使用 cfg.EVALUATION.ensemble_model_paths（4模型集成），
    若为空则回退到 cfg.PATHS.test_model_path（单模型）。

    Returns:
        已加载到 device 并设置为 eval 模式的模型列表。
    """
    ensemble_paths = list(cfg.EVALUATION.ensemble_model_paths)

    if not ensemble_paths:
        # 回退：单模型模式
        single_path = cfg.PATHS.test_model_path
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"未找到预训练模型文件: {single_path}")
        model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
        state_dict = torch.load(single_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"已加载单模型: {os.path.basename(single_path)}")
        return [model]

    models = []
    for path in ensemble_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"集成模型文件未找到: {path}")
        m = get_model(use_attention=cfg.MODEL.use_attention).to(device)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
        print(f"  已加载: {os.path.relpath(path)}")

    print(f"集成推理：共加载 {len(models)} 个模型（softmax 概率平均）")
    return models


def _ensemble_predict(
    models: List[torch.nn.Module],
    inputs: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[int, float]:
    """
    使用多模型集成推理（softmax 概率平均），返回预测类别索引和概率。

    Args:
        models: 已加载的模型列表。
        inputs: (1, max_frames, input_size) 张量。
        lengths: (1,) 张量。

    Returns:
        (top_idx, top_prob) —— 整数类别索引和 float 概率。
    """
    avg_probs = None
    for m in models:
        logits = m(inputs, lengths)
        probs = F.softmax(logits, dim=1)  # (1, num_classes)
        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs = avg_probs + probs

    avg_probs = avg_probs / len(models)  # type: ignore[operator]
    avg_probs = avg_probs.squeeze(0)  # (num_classes,)

    top_prob, top_idx = torch.max(avg_probs, dim=0)
    return int(top_idx.item()), float(top_prob.item())


def _open_file_dialog() -> str | None:
    """
    在独立线程中弹出系统文件选择对话框，选择视频文件。
    Windows 下使用 tkinter.filedialog；失败时返回 None。

    Returns:
        选中的文件路径字符串，或 None（用户取消/失败）。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # 隐藏 tk 主窗口
        root.attributes("-topmost", True)  # 置顶对话框
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print(f"文件对话框打开失败: {e}")
        return None


def _run_offline_inference(
    video_path: str,
    models: List[torch.nn.Module],
    preprocess_helper,
    stats,
    device: torch.device,
    id_to_label: Dict[int, str],
    result_holder: Dict[str, Any],
) -> None:
    """
    离线视频推理：逐帧提取关键点，使用4模型集成推理，结果写入 result_holder。

    Args:
        video_path: 视频文件路径。
        models: 集成模型列表。
        preprocess_helper: 几何归一化处理器。
        stats: Z-Score 统计量（可为 None）。
        device: 推理设备。
        id_to_label: 类别 ID → 标签映射。
        result_holder: 用于跨线程传递结果的共享字典：
            - "status": str，当前处理状态描述
            - "result": Tuple[str, float] | None，最终推理结果
            - "done": bool，处理是否完成
    """
    result_holder["status"] = "正在打开视频..."
    result_holder["result"] = None
    result_holder["done"] = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result_holder["status"] = f"无法打开视频: {os.path.basename(video_path)}"
        result_holder["done"] = True
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extractor = KeypointExtractor(use_video_mode=True)
    frame_buffer: Deque[np.ndarray] = deque(maxlen=cfg.SEQUENCE.max_frames)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # 每5帧更新一次进度状态，减少字符串分配
            if frame_idx % 5 == 0 or frame_idx == 1:
                pct = int(frame_idx / max(total_frames, 1) * 100)
                result_holder["status"] = f"正在处理视频... {pct}%"

            timestamp_ms = frame_idx * 33  # 假设约 30fps

            keypoints, valid_mask, has_hands = extractor.extract_frame_optimized(
                frame, timestamp_ms
            )

            if has_hands and keypoints is not None:
                valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
                if valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame:
                    frame_buffer.append(keypoints)

        result_holder["status"] = "正在推理中..."

        if len(frame_buffer) < 3:
            result_holder["status"] = "视频中未检测到足够手部关键点"
            result_holder["done"] = True
            return

        with torch.no_grad():
            inputs, lengths = prepare_sequence(frame_buffer, preprocess_helper, stats)
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            top_idx, top_prob = _ensemble_predict(models, inputs, lengths)

        pred_label = id_to_label.get(top_idx, str(top_idx))
        result_holder["result"] = (pred_label, top_prob)
        result_holder["status"] = f"识别结果: {pred_label} ({int(top_prob * 100)}%)"

    except Exception as e:
        result_holder["status"] = f"推理出错: {e}"
    finally:
        cap.release()
        extractor.close()
        result_holder["done"] = True


def run_realtime_inference(camera_index: int | str | None = None) -> None:
    """
    使用摄像头或视频文件与预训练模型进行实时手语分类（4模型集成推理）。

    新增功能：
    - 4模型集成推理（softmax 概率平均），对应 cfg.EVALUATION.ensemble_model_paths。
    - UI 右上角新增"导入视频"按钮，点击后弹出文件对话框进行离线推理。

    Args:
        camera_index: 摄像头索引(int)或视频文件路径(str)，默认使用 cfg.INFERENCE.camera_index。

    按下键盘 "q" 或点击右上角退出按钮退出。
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    # 加载标签映射
    id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
    if not id_to_label:
        print("警告: 标签映射为空，将直接输出类别 ID。")

    # 加载集成模型（优先4模型，回退单模型）
    try:
        models = _load_ensemble_models(device)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

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

    # 窗口与鼠标回调
    window_name = "Real-time Sign Prediction"
    mouse_state: Dict[str, Any] = {
        "exit_rect": None,
        "skel_rect": None,
        "import_rect": None,
        "quit": False,
        "show_skeleton": False,
        "mouse_pos": None,
        "import_video": False,  # 点击"导入视频"后置 True
        "importing": False,  # 离线推理进行中时置 True（防止重复触发）
    }
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    # 摄像头输入 (支持视频文件)
    cam_source = cfg.INFERENCE.camera_index if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_source)

    # 仅当输入源为整数（摄像头索引）时设置摄像头参数
    if isinstance(cam_source, int):
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

    # 离线推理状态（跨线程共享，由子线程写入，主线程读取）
    offline_state: Dict[str, Any] = {
        "status": "",
        "result": None,
        "done": True,
    }

    # 离线推理完成后的结果（持久显示，直到下一次实时推理检测到手部才清除）
    offline_result: Tuple[str, float] | None = None
    # 是否处于"离线结果展示"模式（True时实时推理不覆盖结果）
    offline_result_mode: bool = False

    print('\n实时推理已启动（4模型集成），按 "q" 退出。\n')
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

                # 水平翻转画面（仅摄像头，视频文件保持原样）
                if isinstance(cam_source, int):
                    frame = cv2.flip(frame, 1)

                frame_count += 1
                timestamp_ms = int(time.time() * 1000)

                # ---- 处理"导入视频"按钮点击 ----
                if mouse_state.get("import_video") and not mouse_state.get("importing"):
                    mouse_state["import_video"] = False
                    mouse_state["importing"] = True
                    # 清除上次离线结果，恢复实时推理（此次新导入完成前暂不显示旧结果）
                    offline_result = None
                    offline_result_mode = False
                    offline_state["done"] = False
                    offline_state["status"] = "请在弹出对话框中选择视频文件..."
                    offline_state["result"] = None

                    # 在主线程弹出文件对话框（tkinter 要求主线程）
                    video_path = _open_file_dialog()

                    if video_path:
                        print(f"开始离线推理: {video_path}")
                        # 启动后台线程处理视频，避免阻塞 UI
                        t = threading.Thread(
                            target=_run_offline_inference,
                            args=(
                                video_path,
                                models,
                                preprocess_helper,
                                stats,
                                device,
                                id_to_label,
                                offline_state,
                            ),
                            daemon=True,
                        )
                        t.start()
                    else:
                        # 用户取消文件选择
                        offline_state["status"] = ""
                        offline_state["done"] = True
                        mouse_state["importing"] = False

                # ---- 离线推理完成后同步状态 ----
                if mouse_state.get("importing") and offline_state.get("done"):
                    mouse_state["importing"] = False
                    if offline_state.get("result"):
                        offline_result = offline_state["result"]
                        offline_result_mode = True  # 进入离线结果展示模式

                # ---- 离线结果展示模式：锁定结果，完全跳过实时推理 ----
                if offline_result_mode:
                    keypoints = None
                    last_result = offline_result
                    status_text = "离线推理完成"

                # ---- 实时推理逻辑（离线推理进行中或离线结果展示时跳过） ----
                elif not mouse_state.get("importing"):
                    keypoints, valid_mask, has_hands = extractor.extract_frame_optimized(
                        frame, timestamp_ms
                    )

                    if not has_hands:
                        last_result = None
                        status_text = "未检测到手部骨骼点信息"
                        keypoints = None
                    else:
                        valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
                        keypoints_valid = (
                            valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame
                        )

                        if keypoints_valid and keypoints is not None:
                            frame_buffer.append(keypoints)

                            if frame_buffer and (
                                frame_count % cfg.INFERENCE.inference_interval == 0
                            ):
                                try:
                                    inputs, lengths = prepare_sequence(
                                        frame_buffer, preprocess_helper, stats
                                    )
                                    inputs = inputs.to(device)
                                    lengths = lengths.to(device)

                                    top_idx, top_prob = _ensemble_predict(models, inputs, lengths)
                                    pred_label = id_to_label.get(top_idx, str(top_idx))
                                    last_result = (pred_label, top_prob)
                                except Exception:
                                    last_result = None
                        else:
                            status_text = "关键点数量不足"

                    if has_hands and last_result:
                        pass
                    elif has_hands and not last_result:
                        status_text = "正在分析..."
                else:
                    # 离线推理进行中：显示进度，冻结实时结果
                    keypoints = None
                    status_text = offline_state.get("status", "正在处理视频...")
                    last_result = None

                # FPS 估计
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

                # 绘制骨骼（仅实时模式且已开启）
                display_frame = frame.copy()
                if mouse_state.get("show_skeleton") and keypoints is not None:
                    display_frame = draw_skeleton(display_frame, keypoints)

                overlay, exit_rect, skel_rect, import_rect = draw_modern_ui(
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
                mouse_state["import_rect"] = import_rect

                # 检查窗口是否被用户关闭
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("窗口被用户关闭。")
                    break

                cv2.imshow(window_name, overlay)

                # 退出条件：按键 q 或点击退出按钮
                if (cv2.waitKey(1) & 0xFF == ord("q")) or mouse_state.get("quit"):
                    break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("已退出实时推理。")


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="实时手语推理（4模型集成）")
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

    source = args.camera if args.camera is not None else cfg.INFERENCE.camera_index
    if args.video:
        if os.path.exists(args.video):
            source = args.video
            print(f"将使用视频文件进行推理: {source}")
        else:
            print(f"错误: 视频文件不存在: {args.video}")
            sys.exit(1)

    run_realtime_inference(camera_index=source)
