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
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    绘制现代化、温馨极简主义风格的实时 UI 界面。

    Returns:
        (rendered_frame, exit_rect, skel_rect)
    """
    h, w = frame.shape[:2]

    # 转换为 PIL RGBA 进行透明度绘制
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 颜色定义 (保持与启动器类似的暖调)
    color_surface = (245, 250, 255, 230)
    color_surface_dim = (220, 230, 240, 200)
    color_accent = (140, 160, 250, 240)
    color_accent_dim = (110, 130, 220, 200)
    color_danger = (255, 120, 110, 240)
    color_danger_dim = (255, 150, 140, 200)
    color_teal = (130, 220, 160, 240)
    color_teal_dim = (150, 230, 180, 220)
    color_text = (55, 65, 80, 255)
    color_text_dim = (110, 120, 140, 255)

    # ==========================
    # 1. 顶部信息栏 (FPS)
    # ==========================
    fps_text = f"FPS: {fps:.1f}"
    fps_bbox = draw.textbbox((0, 0), fps_text, font=font_small)
    fps_w = fps_bbox[2] - fps_bbox[0] + 24
    fps_h = fps_bbox[3] - fps_bbox[1] + 12
    fps_x = 20
    fps_y = 20

    # 绘制 FPS 背景胶囊 (暖白半透明)
    draw.rounded_rectangle(
        [fps_x, fps_y, fps_x + fps_w, fps_y + fps_h],
        radius=12,
        fill=color_surface_dim,
        outline=(200, 210, 220, 150),
        width=1,
    )
    draw.text((fps_x + 12, fps_y + 6), fps_text, font=font_small, fill=color_text)

    # ==========================
    # 2. 交互按钮区域 (右上角)
    # ==========================
    margin_r, margin_t = cfg.UI.exit_button_margin
    btn_height = 44  # 稍微加大高度
    btn_radius = 22  # 更圆滑的角

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

    # 退出按钮：暖红渐变
    if is_hover_exit:
        exit_fill = color_danger
        exit_outline = (255, 100, 90, 200)
    else:
        exit_fill = color_surface
        exit_outline = (200, 210, 220, 150)

    draw.rounded_rectangle(
        [exit_x1, exit_y1 + 2, exit_x2, exit_y2 + 2], radius=btn_radius, fill=(200, 200, 190, 80)
    )

    draw.rounded_rectangle(
        [exit_x1, exit_y1, exit_x2, exit_y2],
        radius=btn_radius,
        fill=exit_fill,
        outline=exit_outline,
        width=1,
    )

    # 居中文字
    exit_text_color = (255, 255, 255, 255) if is_hover_exit else color_danger_dim
    draw.text(
        (
            exit_x1 + (exit_w - exit_text_w) // 2,
            exit_y1 + (btn_height - (exit_bbox[3] - exit_bbox[1])) // 2 - 2,
        ),
        exit_text,
        font=font_small,
        fill=exit_text_color,
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

    # 绿色系 (开) / 暖白系 (关)
    if show_skeleton:
        skel_fill = color_teal if is_hover_skel else color_teal_dim
        skel_outline = (100, 200, 130, 200) if is_hover_skel else (120, 210, 150, 150)
        skel_text_color = (255, 255, 255, 255)
    else:
        skel_fill = color_surface if is_hover_skel else color_surface_dim
        skel_outline = (200, 210, 220, 200) if is_hover_skel else (210, 220, 230, 150)
        skel_text_color = color_text_dim

    draw.rounded_rectangle(
        [skel_x1, skel_y1 + 2, skel_x2, skel_y2 + 2], radius=btn_radius, fill=(200, 200, 190, 80)
    )

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
        fill=skel_text_color,
    )

    # ==========================
    # 3. 底部结果卡 / 状态卡
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

        # 暖白磨砂玻璃背景
        draw.rounded_rectangle(
            [card_x1, card_y1 + 4, card_x2, card_y2 + 4], radius=20, fill=(200, 200, 190, 80)
        )
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=color_surface,
            outline=(200, 210, 220, 180),
            width=2,
        )

        # 顶部：标签 和 概率数值
        # 左侧放 Label, 右侧放 概率
        text_y_base = card_y1 + 25

        draw.text(
            (card_x1 + 30, text_y_base),
            label,
            font=font_main,
            fill=color_text,
        )

        # 概率颜色 (更暖更柔和)
        if prob > 0.8:
            prob_color = (100, 200, 120, 255)
        elif prob > 0.5:
            prob_color = (240, 180, 80, 255)
        else:
            prob_color = (230, 120, 110, 255)

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
            fill=(220, 230, 240, 255),
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
            [card_x1, card_y1 + 4, card_x2, card_y2 + 4], radius=20, fill=(200, 200, 190, 80)
        )
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=color_surface_dim,
            outline=(200, 210, 220, 180),
            width=2,
        )

        # 居中显示提示
        draw.text(
            (
                card_x1 + (card_w - hint_w) // 2,
                card_y1 + (card_h - (hint_bbox[3] - hint_bbox[1])) // 2 - 4,
            ),
            hint_text,
            font=font_main,
            fill=color_text_dim,
        )

    # 混合图层
    out = Image.alpha_composite(image, overlay)
    rendered = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)

    return (
        rendered,
        (exit_x1, exit_y1, exit_x2, exit_y2),
        (skel_x1, skel_y1, skel_x2, skel_y2),
    )
    draw.text((fps_x + 12, fps_y + 6), fps_text, font=font_small, fill=(100, 80, 70, 255))

    # ==========================
    # 2. 交互按钮区域 (右上角)
    # ==========================
    margin_r, margin_t = cfg.UI.exit_button_margin
    btn_height = 44  # 稍微加大高度
    btn_radius = 22  # 更圆滑的角

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

    # 退出按钮：暖红渐变
    if is_hover_exit:
        exit_fill = (255, 120, 110, 240)
        exit_outline = (255, 150, 140, 200)
    else:
        exit_fill = (255, 250, 245, 200)
        exit_outline = (255, 200, 190, 150)

    draw.rounded_rectangle(
        [exit_x1, exit_y1 + 2, exit_x2, exit_y2 + 2], radius=btn_radius, fill=(180, 120, 110, 50)
    )

    draw.rounded_rectangle(
        [exit_x1, exit_y1, exit_x2, exit_y2],
        radius=btn_radius,
        fill=exit_fill,
        outline=exit_outline,
        width=1,
    )

    # 居中文字
    exit_text_color = (255, 255, 255, 255) if is_hover_exit else (220, 80, 70, 255)
    draw.text(
        (
            exit_x1 + (exit_w - exit_text_w) // 2,
            exit_y1 + (btn_height - (exit_bbox[3] - exit_bbox[1])) // 2 - 2,
        ),
        exit_text,
        font=font_small,
        fill=exit_text_color,
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

    # 绿色系 (开) / 暖白系 (关)
    if show_skeleton:
        skel_fill = (130, 220, 160, 240) if is_hover_skel else (150, 230, 180, 220)
        skel_outline = (180, 240, 200, 200) if is_hover_skel else (200, 250, 220, 150)
        skel_text_color = (255, 255, 255, 255)
    else:
        skel_fill = (240, 235, 230, 240) if is_hover_skel else (255, 250, 245, 200)
        skel_outline = (220, 210, 200, 200) if is_hover_skel else (230, 220, 210, 150)
        skel_text_color = (120, 110, 100, 255)

    draw.rounded_rectangle(
        [skel_x1, skel_y1 + 2, skel_x2, skel_y2 + 2], radius=btn_radius, fill=(150, 150, 140, 40)
    )

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
        fill=skel_text_color,
    )

    # ==========================
    # 3. 底部结果卡 / 状态卡
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

        # 暖白磨砂玻璃背景
        draw.rounded_rectangle(
            [card_x1, card_y1 + 4, card_x2, card_y2 + 4], radius=20, fill=(150, 140, 130, 50)
        )
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=(255, 252, 248, 230),
            outline=(240, 230, 220, 150),
            width=2,
        )

        # 顶部：标签 和 概率数值
        # 左侧放 Label, 右侧放 概率
        text_y_base = card_y1 + 25

        draw.text(
            (card_x1 + 30, text_y_base),
            label,
            font=font_main,
            fill=(70, 60, 50, 255),
        )

        # 概率颜色 (更暖更柔和)
        if prob > 0.8:
            prob_color = (100, 200, 120, 255)
        elif prob > 0.5:
            prob_color = (240, 180, 80, 255)
        else:
            prob_color = (230, 120, 110, 255)

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
            fill=(235, 230, 225, 255),
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
            [card_x1, card_y1 + 4, card_x2, card_y2 + 4], radius=20, fill=(150, 140, 130, 40)
        )
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=20,
            fill=(255, 250, 245, 220),
            outline=(240, 230, 220, 150),
            width=2,
        )

        # 居中显示提示
        draw.text(
            (
                card_x1 + (card_w - hint_w) // 2,
                card_y1 + (card_h - (hint_bbox[3] - hint_bbox[1])) // 2 - 4,
            ),
            hint_text,
            font=font_main,
            fill=(140, 130, 120, 255),
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
            return


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


def run_realtime_inference(camera_index: int | str | None = None) -> None:
    """
    使用摄像头或视频文件与预训练模型进行实时手语分类（4模型集成推理）。

    # 新增功能：
    #   - 4模型集成推理（softmax 概率平均），对应 cfg.EVALUATION.ensemble_model_paths。

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

                # ---- 实时推理逻辑 ----
                keypoints, valid_mask, has_hands = extractor.extract_frame_optimized(
                    frame, timestamp_ms
                )

                if not has_hands:
                    last_result = None
                    status_text = "未检测到手部骨骼点信息"
                    keypoints = None
                else:
                    valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
                    keypoints_valid = valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame

                    if keypoints_valid and keypoints is not None:
                        frame_buffer.append(keypoints)

                        if frame_buffer and (frame_count % cfg.INFERENCE.inference_interval == 0):
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

                if has_hands and not last_result:
                    status_text = "正在分析..."

                # FPS 估计
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

                # 绘制骨骼（仅实时模式且已开启）
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


def _load_shared_resources():
    """
    加载推理所需的共享资源（模型、字体、预处理等）。
    供 run_launcher / run_realtime_inference / run_offline_inference_standalone 复用。

    Returns:
        (device, id_to_label, models, extractor, preprocess_helper, stats, font_main, font_small)
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
    if not id_to_label:
        print("警告: 标签映射为空，将直接输出类别 ID。")

    try:
        models = _load_ensemble_models(device)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        raise

    extractor = KeypointExtractor(use_video_mode=True)
    preprocess_helper = PreprocessHelper(max_frames=cfg.SEQUENCE.max_frames)

    stats = None
    if cfg.PREPROCESS.enable_standardize:
        if os.path.exists(cfg.PREPROCESS.feature_stats_path):
            stats = load_feature_stats(cfg.PREPROCESS.feature_stats_path)
            print(f"已加载标准化统计量: {cfg.PREPROCESS.feature_stats_path}")
        else:
            print("警告: 启用了标准化但未找到统计量文件，推理时将跳过标准化。")

    font_main = load_chinese_font(cfg.UI.font_size)
    font_small = load_chinese_font(cfg.UI.font_small_size)

    return device, id_to_label, models, extractor, preprocess_helper, stats, font_main, font_small


def _draw_launcher_ui(
    width: int,
    height: int,
    font_main,
    font_small,
    mouse_pos: "Tuple[int, int] | None",
) -> "Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int,int,int]]":
    """
    绘制启动选择界面 (简约、温馨、美观、充满人文关怀的暖色调)。

    Returns:
        (rendered, realtime_btn_rect, offline_btn_rect)
    """
    # 暖色背景: 浅米色 (B, G, R)
    canvas = np.full((height, width, 3), (237, 246, 253), dtype=np.uint8)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cx = width // 2
    cy = height // 2

    # 标题 (深棕色/暖黑色)
    title = "手语识别系统"
    title_bbox = draw.textbbox((0, 0), title, font=font_main)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        (cx - title_w // 2, cy - 130),
        title,
        font=font_main,
        fill=(80, 65, 55, 255),
    )

    # 副标题 (柔和的中性棕色)
    subtitle = "请选择推理模式"
    sub_bbox = draw.textbbox((0, 0), subtitle, font=font_small)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(
        (cx - sub_w // 2, cy - 80),
        subtitle,
        font=font_small,
        fill=(140, 120, 110, 220),
    )

    btn_w, btn_h, btn_gap = 200, 56, 40
    btn_radius = 28  # 更圆润的按钮

    # --- 实时推理按钮 (暖珊瑚色) ---
    rt_x1 = cx - btn_w - btn_gap // 2
    rt_x2 = rt_x1 + btn_w
    rt_y1 = cy - btn_h // 2 + 10
    rt_y2 = rt_y1 + btn_h

    is_hover_rt = False
    if mouse_pos:
        mx, my = mouse_pos
        if rt_x1 <= mx <= rt_x2 and rt_y1 <= my <= rt_y2:
            is_hover_rt = True

    # 暖珊瑚色 (浅橙红)
    rt_fill = (255, 150, 130, 240) if is_hover_rt else (255, 170, 150, 220)
    rt_outline = (255, 130, 110, 200) if is_hover_rt else (255, 190, 170, 150)

    # 按钮阴影 (简单的向下偏移)
    draw.rounded_rectangle(
        [rt_x1, rt_y1 + 4, rt_x2, rt_y2 + 4], radius=btn_radius, fill=(200, 130, 110, 60)
    )
    draw.rounded_rectangle(
        [rt_x1, rt_y1, rt_x2, rt_y2], radius=btn_radius, fill=rt_fill, outline=rt_outline, width=2
    )

    rt_text = "实时推理"
    rt_bbox = draw.textbbox((0, 0), rt_text, font=font_small)
    rt_tw = rt_bbox[2] - rt_bbox[0]
    draw.text(
        (rt_x1 + (btn_w - rt_tw) // 2, rt_y1 + (btn_h - (rt_bbox[3] - rt_bbox[1])) // 2 - 2),
        rt_text,
        font=font_small,
        fill=(255, 255, 255, 255),
    )

    # --- 离线推理按钮 (暖阳黄/沙金色) ---
    of_x1 = cx + btn_gap // 2
    of_x2 = of_x1 + btn_w
    of_y1 = rt_y1
    of_y2 = rt_y2

    is_hover_of = False
    if mouse_pos:
        mx, my = mouse_pos
        if of_x1 <= mx <= of_x2 and of_y1 <= my <= of_y2:
            is_hover_of = True

    of_fill = (245, 195, 120, 240) if is_hover_of else (245, 210, 140, 220)
    of_outline = (230, 170, 90, 200) if is_hover_of else (245, 220, 160, 150)

    # 按钮阴影
    draw.rounded_rectangle(
        [of_x1, of_y1 + 4, of_x2, of_y2 + 4], radius=btn_radius, fill=(200, 160, 100, 60)
    )
    draw.rounded_rectangle(
        [of_x1, of_y1, of_x2, of_y2], radius=btn_radius, fill=of_fill, outline=of_outline, width=2
    )

    of_text = "离线推理"
    of_bbox = draw.textbbox((0, 0), of_text, font=font_small)
    of_tw = of_bbox[2] - of_bbox[0]
    draw.text(
        (of_x1 + (btn_w - of_tw) // 2, of_y1 + (btn_h - (of_bbox[3] - of_bbox[1])) // 2 - 2),
        of_text,
        font=font_small,
        fill=(255, 255, 255, 255),
    )

    # 底部提示
    hint = "按 Q 退出"
    hint_bbox = draw.textbbox((0, 0), hint, font=font_small)
    hint_w = hint_bbox[2] - hint_bbox[0]
    draw.text(
        (cx - hint_w // 2, rt_y2 + 40),
        hint,
        font=font_small,
        fill=(160, 145, 135, 180),
    )

    out = Image.alpha_composite(image, overlay)
    rendered = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)
    return rendered, (rt_x1, rt_y1, rt_x2, rt_y2), (of_x1, of_y1, of_x2, of_y2)


def run_launcher() -> None:
    """
    主入口：显示模式选择界面，用户选择实时推理或离线推理后跳转。
    """
    _configure_windows_console()

    try:
        device, id_to_label, models, extractor, preprocess_helper, stats, font_main, font_small = (
            _load_shared_resources()
        )
    except FileNotFoundError:
        return

    window_name = "Sign Language Recognition System"
    win_w, win_h = 800, 480
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)

    launcher_mouse: Dict[str, Any] = {
        "mouse_pos": None,
        "choice": None,  # "realtime" | "offline"
    }

    def _on_launcher_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEMOVE:
            params["mouse_pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            rt_rect = params.get("rt_rect")
            of_rect = params.get("of_rect")
            if rt_rect:
                x1, y1, x2, y2 = rt_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    params["choice"] = "realtime"
                    return
            if of_rect:
                x1, y1, x2, y2 = of_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    params["choice"] = "offline"

    cv2.setMouseCallback(window_name, _on_launcher_mouse, launcher_mouse)

    try:
        while True:
            rendered, rt_rect, of_rect = _draw_launcher_ui(
                win_w, win_h, font_main, font_small, launcher_mouse.get("mouse_pos")
            )
            launcher_mouse["rt_rect"] = rt_rect
            launcher_mouse["of_rect"] = of_rect

            cv2.imshow(window_name, rendered)

            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            choice = launcher_mouse.get("choice")
            if choice == "realtime":
                # 直接在同一窗口运行实时推理
                _run_realtime_in_window(
                    window_name=window_name,
                    models=models,
                    extractor=extractor,
                    preprocess_helper=preprocess_helper,
                    stats=stats,
                    device=device,
                    id_to_label=id_to_label,
                    font_main=font_main,
                    font_small=font_small,
                )
                launcher_mouse["choice"] = None
                # 返回启动界面
                cv2.resizeWindow(window_name, win_w, win_h)
                cv2.setMouseCallback(window_name, _on_launcher_mouse, launcher_mouse)
            elif choice == "offline":
                _run_offline_in_window(
                    window_name=window_name,
                    models=models,
                    preprocess_helper=preprocess_helper,
                    stats=stats,
                    device=device,
                    id_to_label=id_to_label,
                    font_main=font_main,
                    font_small=font_small,
                    win_w=win_w,
                    win_h=win_h,
                )
                launcher_mouse["choice"] = None
                cv2.resizeWindow(window_name, win_w, win_h)
                cv2.setMouseCallback(window_name, _on_launcher_mouse, launcher_mouse)
    finally:
        extractor.close()
        cv2.destroyAllWindows()
        print("已退出。")


def _run_realtime_in_window(
    window_name: str,
    models: List[torch.nn.Module],
    extractor,
    preprocess_helper,
    stats,
    device: torch.device,
    id_to_label: Dict[int, str],
    font_main,
    font_small,
) -> None:
    """在给定窗口中运行实时推理，退出后返回。"""
    cam_source = cfg.INFERENCE.camera_index
    cap = cv2.VideoCapture(cam_source)

    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.INFERENCE.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.INFERENCE.camera_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.INFERENCE.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {cam_source}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.resizeWindow(window_name, actual_w, actual_h)

    mouse_state: Dict[str, Any] = {
        "exit_rect": None,
        "skel_rect": None,
        "quit": False,
        "show_skeleton": False,
        "mouse_pos": None,
    }
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    frame_buffer: Deque[np.ndarray] = deque(maxlen=cfg.SEQUENCE.max_frames)
    last_result: Tuple[str, float] | None = None
    start_time = time.time()
    frame_count = 0
    status_text = "初始化..."

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if isinstance(cam_source, int):
                    frame = cv2.flip(frame, 1)

                frame_count += 1
                timestamp_ms = int(time.time() * 1000)

                keypoints, valid_mask, has_hands = extractor.extract_frame_optimized(
                    frame, timestamp_ms
                )

                if not has_hands:
                    last_result = None
                    status_text = "未检测到手部骨骼点信息"
                    keypoints = None
                else:
                    valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
                    keypoints_valid = valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame

                    if keypoints_valid and keypoints is not None:
                        frame_buffer.append(keypoints)
                        if frame_buffer and (frame_count % cfg.INFERENCE.inference_interval == 0):
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

                if has_hands and not last_result:
                    status_text = "正在分析..."

                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-5)

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

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                cv2.imshow(window_name, overlay)

                if (cv2.waitKey(1) & 0xFF == ord("q")) or mouse_state.get("quit"):
                    break
    finally:
        cap.release()
        print("已退出实时推理。")


def _run_offline_in_window(
    window_name: str,
    models: List[torch.nn.Module],
    preprocess_helper,
    stats,
    device: torch.device,
    id_to_label: Dict[int, str],
    font_main,
    font_small,
    win_w: int,
    win_h: int,
) -> None:
    """在给定窗口中运行离线推理模式，退出后返回。"""
    from src.model.offline_inference import run_offline_mode  # noqa: PLC0415

    run_offline_mode(
        models=models,
        preprocess_helper=preprocess_helper,
        stats=stats,
        device=device,
        id_to_label=id_to_label,
        font_main=font_main,
        font_small=font_small,
        window_name=window_name,
        target_size=(win_w, win_h),
    )


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
