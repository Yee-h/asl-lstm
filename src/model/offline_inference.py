"""
offline_inference.py — 离线视频推理模块

职责：
  - draw_offline_ui()       : 渲染离线推理界面（全新深色科技风设计）
  - on_offline_mouse()      : 离线界面鼠标回调
  - run_offline_mode()      : 离线推理主循环（可重复导入视频）
  - _open_file_dialog()     : tkinter 文件选择对话框
  - _run_offline_inference(): 后台推理线程
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ── 确保 src 可作为模块导入 ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.config as cfg
from src.core.labels import load_id_to_label_map
from src.model.dataloader import load_feature_stats, preprocess_keypoints
from src.data_process.preprocess_wlasl import (
    KeypointExtractor,
    ParallelKeypointExtractor,
    PreprocessHelper,
)


# ─────────────────────────────────────────────────────────────────────────────
# 调色板（简约、温馨、美观、人文关怀的暖色调）
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    # 背景 / 表面
    "bg": (237, 246, 253),  # 暖米色/浅黄 (BGR: 浅黄色 253, 246, 237)
    "surface": (245, 250, 255),  # 卡片背景 (偏暖白)
    "surface2": (220, 230, 240),  # 次级面板
    # 品牌主色：温馨暖色
    "accent": (140, 160, 250),  # 主强调 (暖粉/紫)
    "accent_dim": (110, 130, 220),  # 暗态
    "teal": (120, 200, 100),  # 柔和绿（骨骼按钮）
    "teal_dim": (100, 180, 80),
    "danger": (110, 120, 255),  # 暖红（退出）
    "danger_dim": (80, 90, 220),
    "success": (120, 200, 100),  # 成功绿
    "warning": (80, 180, 240),  # 警告黄
    "error": (110, 120, 230),  # 错误红
    # 文字
    "text": (55, 65, 80),  # 主文字 (深棕)
    "text_dim": (110, 120, 140),  # 次文字 (中棕)
    "text_muted": (135, 145, 160),  # 弱文字
    # 边框
    "border": (200, 210, 220),  # 普通边框
    "border_hi": (180, 190, 200),  # 高亮边框
}


def _rgba(bgr: Tuple[int, int, int], a: int = 255) -> Tuple[int, int, int, int]:
    """BGR → RGBA（PIL 使用 RGB）"""
    return (bgr[2], bgr[1], bgr[0], a)


def _open_file_dialog() -> str | None:
    """弹出文件选择对话框，返回视频路径或 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="选择手语视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print(f"文件对话框错误: {e}")
        return None


def _ensemble_predict(
    models: List[torch.nn.Module],
    inputs: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[int, float]:
    """多模型集成推理，返回 (类别索引, 平均最大概率)。"""
    probs_list = []
    for model in models:
        model.eval()
        out = model(inputs, lengths)
        probs_list.append(F.softmax(out, dim=-1))
    avg_probs = torch.stack(probs_list).mean(dim=0)
    top_prob, top_idx = avg_probs.max(dim=-1)
    return int(top_idx.item()), float(top_prob.item())


def _run_offline_inference(
    video_path: str,
    models: List[torch.nn.Module],
    preprocess_helper: PreprocessHelper,
    stats: Dict[str, Any] | None,
    device: torch.device,
    id_to_label: Dict[int, str],
    state: Dict[str, Any],
) -> None:
    """
    后台推理线程：并行提取关键点 → 集成推理 → 写入 state。

    使用 ThreadPoolExecutor + IMAGE 模式的 KeypointExtractor 并行处理帧，
    提取完成后按帧序合并结果。当 parallel_workers=1 时回退到单线程顺序提取。

    state 字段：
      status  : 进度提示文字
      result  : (label, prob) 或 None
      done    : 推理是否完成
      keypoints_by_frame : List[np.ndarray | None]，每帧关键点（用于骨骼叠加）
    """
    from collections import deque
    from src.model.dataloader import preprocess_keypoints

    state["keypoints_by_frame"] = []

    # ── 1. 读取全部视频帧到内存 ────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        state["status"] = "错误: 无法打开视频文件"
        state["done"] = True
        return

    state["status"] = "正在读取视频帧..."
    all_frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        state["status"] = "错误: 视频无有效帧"
        state["done"] = True
        return

    # ── 2. 关键点提取（并行或顺序）──────────────────────────────────────────
    parallel_extractor = ParallelKeypointExtractor(cfg.INFERENCE.parallel_workers)
    num_workers = parallel_extractor.num_workers

    if num_workers <= 1:
        # ── 单线程顺序提取（使用 VIDEO 模式以获取跟踪优势）──
        parallel_extractor.close()  # 不需要线程池，释放
        extractor = KeypointExtractor(use_video_mode=True)
        fps_val = cv2.VideoCapture(video_path)
        fps = fps_val.get(cv2.CAP_PROP_FPS)
        fps_val.release()
        if fps <= 0:
            fps = 30.0

        extraction_results: List[Tuple[np.ndarray | None, np.ndarray | None, bool]] = []
        try:
            for idx, frame in enumerate(all_frames):
                timestamp_ms = int(((idx + 1) / fps) * 1000)
                kp, valid_mask, has_hands = extractor.extract_frame_optimized(frame, timestamp_ms)
                extraction_results.append((kp, valid_mask, has_hands))
                pct = int((idx + 1) / total_frames * 100)
                state["status"] = f"正在提取特征... {pct}%"
        finally:
            extractor.close()
    else:
        # ── 多线程并行提取（IMAGE 模式，无状态，可安全并行）──
        def _progress_cb(completed: int, total: int) -> None:
            pct = int(completed / total * 100)
            state["status"] = f"正在提取特征... {pct}%（{num_workers}线程并行）"

        try:
            extraction_results = parallel_extractor.extract_batch(
                all_frames, progress_callback=_progress_cb
            )
        finally:
            parallel_extractor.close()

    # ── 3. 按帧序合并结果 ─────────────────────────────────────────────────
    keypoints_seq: List[np.ndarray] = []
    keypoints_by_frame: List[np.ndarray | None] = []

    for kp, valid_mask, has_hands in extraction_results:
        if has_hands and kp is not None:
            valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
            if valid_count >= cfg.PREPROCESS.min_valid_keypoints_per_frame:
                keypoints_seq.append(kp)
                keypoints_by_frame.append(kp)
            else:
                keypoints_by_frame.append(None)
        else:
            keypoints_by_frame.append(None)

    state["keypoints_by_frame"] = keypoints_by_frame

    if len(keypoints_seq) < 5:
        state["status"] = "有效帧不足，无法推理"
        state["done"] = True
        return

    # ── 4. 模型推理 ───────────────────────────────────────────────────────
    state["status"] = "正在推理..."

    try:
        from collections import deque as _deque

        buf = _deque(keypoints_seq, maxlen=cfg.SEQUENCE.max_frames)
        raw_data = np.stack(buf)
        raw_mask = ((raw_data[:, 0, :] != 0) | (raw_data[:, 1, :] != 0)).astype(np.uint8)

        processed_data, valid_len, processed_mask, _ = preprocess_helper.process_video_sequence(
            raw_data, raw_mask
        )
        data_tensor, valid_len = preprocess_keypoints(
            processed_data,
            max_frames=cfg.SEQUENCE.max_frames,
            valid_len=valid_len,
            stats=stats,
            standardize=cfg.PREPROCESS.enable_standardize and stats is not None,
            mask=processed_mask,
        )
        inputs = data_tensor.unsqueeze(0).to(device)
        lengths = torch.tensor([valid_len], dtype=torch.long).to(device)

        with torch.no_grad():
            top_idx, top_prob = _ensemble_predict(models, inputs, lengths)

        pred_label = id_to_label.get(top_idx, str(top_idx))
        state["result"] = (pred_label, top_prob)
        state["status"] = "推理完成"
    except Exception as e:
        state["status"] = f"推理错误: {e}"

    state["done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 骨骼绘制（复用 realtime_inference 中的连接关系定义）
# ─────────────────────────────────────────────────────────────────────────────

_BODY_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (0, 16),
    (15, 17),
    (16, 18),
]
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_skeleton_on_frame(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """在帧上绘制骨骼关键点，返回叠加后的帧（不修改原帧）。"""
    h, w = frame.shape[:2]
    result = frame.copy()
    radius = cfg.UI.skeleton_point_radius
    pt_color = cfg.UI.skeleton_point_color
    ln_color = cfg.UI.skeleton_line_color
    thickness = cfg.UI.skeleton_line_thickness

    def get_pt(idx: int):
        x, y = float(keypoints[0, idx]), float(keypoints[1, idx])
        if x == 0.0 and y == 0.0:
            return None
        return (int(x * w), int(y * h))

    for i, j in _BODY_CONNECTIONS:
        p1, p2 = get_pt(i), get_pt(j)
        if p1 and p2:
            cv2.line(result, p1, p2, ln_color, thickness)
    for i, j in _HAND_CONNECTIONS:
        p1, p2 = get_pt(i + 25), get_pt(j + 25)
        if p1 and p2:
            cv2.line(result, p1, p2, ln_color, thickness)
    for i, j in _HAND_CONNECTIONS:
        p1, p2 = get_pt(i + 46), get_pt(j + 46)
        if p1 and p2:
            cv2.line(result, p1, p2, ln_color, thickness)
    for idx in range(135):
        pt = get_pt(idx)
        if pt:
            cv2.circle(result, pt, radius, pt_color, -1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI 绘制
# ─────────────────────────────────────────────────────────────────────────────


def _draw_btn(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    font,
    is_hover: bool,
    fill_normal: Tuple[int, int, int, int],
    fill_hover: Tuple[int, int, int, int],
    outline_normal: Tuple[int, int, int, int],
    outline_hover: Tuple[int, int, int, int],
    radius: int = 20,  # 增大圆角
) -> None:
    """通用圆角按钮绘制（带 hover 高亮）。"""
    fill = fill_hover if is_hover else fill_normal
    outline = outline_hover if is_hover else outline_normal
    # 下沉阴影效果
    draw.rounded_rectangle([x1, y1 + 2, x2, y2 + 2], radius=radius, fill=(200, 200, 190, 80))
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=1)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th_t = bbox[3] - bbox[1]
    bw, bh = x2 - x1, y2 - y1
    draw.text(
        (x1 + (bw - tw) // 2, y1 + (bh - th_t) // 2 - 1),
        text,
        font=font,
        fill=(255, 255, 255, 255),  # 白色文字清晰度高
    )


def draw_offline_ui(
    video_frame: np.ndarray | None,
    result: Tuple[str, float] | None,
    status_text: str,
    font_main,
    font_small,
    mouse_pos: Tuple[int, int] | None,
    show_skeleton: bool,
    target_size: Tuple[int, int],
    is_playing: bool = False,
    video_rect: Tuple[int, int, int, int] | None = None,
    current_keypoints: np.ndarray | None = None,
    has_video: bool = False,
) -> Tuple[
    np.ndarray,
    Tuple[int, int, int, int],  # exit_offline_rect
    Tuple[int, int, int, int],  # skel_rect
    Tuple[int, int, int, int] | None,  # close_result_rect
    Tuple[int, int, int, int] | None,  # import_new_rect
    Tuple[int, int, int, int] | None,  # video_area_rect
    Tuple[int, int, int, int] | None,  # rerun_rect
]:
    """
    渲染离线推理界面（深色科技风）。

    Returns:
        (rendered, exit_rect, skel_rect, close_result_rect, import_new_rect, video_area_rect, rerun_rect)
    """
    tw, th = target_size
    mx_raw, my_raw = mouse_pos if mouse_pos else (0, 0)

    def hover(x1, y1, x2, y2) -> bool:
        return mouse_pos is not None and x1 <= mx_raw <= x2 and y1 <= my_raw <= y2

    # ── 深色背景画布 ─────────────────────────────────────────────────────────
    bg = _C["bg"]
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)

    # ── 顶部装饰线（蓝色细线）────────────────────────────────────────────────
    cv2.line(canvas, (0, 2), (tw, 2), _C["accent"], 2)

    # ── 中央视频区域 ──────────────────────────────────────────────────────────
    computed_video_rect: Tuple[int, int, int, int] | None = None
    if video_frame is not None:
        # 骨骼叠加（叠加在缩放前的原始帧）
        display_vf = video_frame.copy()
        if show_skeleton and current_keypoints is not None:
            display_vf = draw_skeleton_on_frame(display_vf, current_keypoints)

        vf_h, vf_w = display_vf.shape[:2]
        max_vw = int(tw * 0.84)
        max_vh = int(th * 0.68)
        scale = min(max_vw / max(vf_w, 1), max_vh / max(vf_h, 1))
        disp_w = max(1, int(vf_w * scale))
        disp_h = max(1, int(vf_h * scale))
        resized = cv2.resize(display_vf, (disp_w, disp_h))

        vx = int((tw - disp_w) // 2)
        vy = max(10, int((th - disp_h) // 2 - 24))
        vy_end = min(th, vy + disp_h)
        vx_end = min(tw, vx + disp_w)
        canvas[vy:vy_end, vx:vx_end] = resized[: vy_end - vy, : vx_end - vx]

        # 视频边框：双层（外层暗 + 内层亮）
        cv2.rectangle(canvas, (vx - 2, vy - 2), (vx_end + 1, vy_end + 1), _C["border_hi"], 1)
        cv2.rectangle(canvas, (vx - 1, vy - 1), (vx_end, vy_end), _C["accent_dim"], 1)
        computed_video_rect = (vx, vy, vx_end, vy_end)

    # ── PIL 叠加层 ────────────────────────────────────────────────────────────
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── 顶部按钮栏 ────────────────────────────────────────────────────────────
    margin_r: int = int(cfg.UI.exit_button_margin[0]) + 4
    margin_t: int = 10
    btn_h = 42
    btn_r = 12
    gap = 12

    # --- 退出离线推理（最右）---
    exit_text = "退出"
    eb = draw.textbbox((0, 0), exit_text, font=font_small)
    exit_w = max(88, int(eb[2] - eb[0] + 36))
    exit_x1 = int(tw - margin_r - exit_w)
    exit_y1 = int(margin_t)
    exit_x2 = int(exit_x1 + exit_w)
    exit_y2 = int(exit_y1 + btn_h)
    _draw_btn(
        draw,
        exit_x1,
        exit_y1,
        exit_x2,
        exit_y2,
        exit_text,
        font_small,
        hover(exit_x1, exit_y1, exit_x2, exit_y2),
        fill_normal=_rgba(_C["surface2"], 200),
        fill_hover=_rgba(_C["danger"], 230),
        outline_normal=_rgba(_C["border"], 180),
        outline_hover=_rgba(_C["danger"], 200),
        radius=btn_r,
    )

    # --- 骨骼切换（退出左侧）---
    skel_text = "骨骼 开" if show_skeleton else "骨骼 关"
    sb = draw.textbbox((0, 0), skel_text, font=font_small)
    skel_w = max(100, int(sb[2] - sb[0] + 36))
    skel_x2 = int(exit_x1 - gap)
    skel_x1 = int(skel_x2 - skel_w)
    skel_y1 = int(margin_t)
    skel_y2 = int(skel_y1 + btn_h)
    _draw_btn(
        draw,
        skel_x1,
        skel_y1,
        skel_x2,
        skel_y2,
        skel_text,
        font_small,
        hover(skel_x1, skel_y1, skel_x2, skel_y2),
        fill_normal=_rgba(_C["teal_dim"], 200) if show_skeleton else _rgba(_C["surface2"], 200),
        fill_hover=_rgba(_C["teal"], 230),
        outline_normal=_rgba(_C["teal_dim"], 160) if show_skeleton else _rgba(_C["border"], 160),
        outline_hover=_rgba(_C["teal"], 200),
        radius=btn_r,
    )

    # --- 导入新视频（骨骼按钮左侧）---
    import_new_text = "导入视频"
    ib = draw.textbbox((0, 0), import_new_text, font=font_small)
    import_w = max(108, int(ib[2] - ib[0] + 36))
    import_x2 = int(skel_x1 - gap)
    import_x1 = int(import_x2 - import_w)
    import_y1 = int(margin_t)
    import_y2 = int(import_y1 + btn_h)
    _draw_btn(
        draw,
        import_x1,
        import_y1,
        import_x2,
        import_y2,
        import_new_text,
        font_small,
        hover(import_x1, import_y1, import_x2, import_y2),
        fill_normal=_rgba(_C["accent_dim"], 200),
        fill_hover=_rgba(_C["accent"], 230),
        outline_normal=_rgba(_C["accent_dim"], 160),
        outline_hover=_rgba(_C["accent"], 200),
        radius=btn_r,
    )
    import_new_rect: Tuple[int, int, int, int] = (import_x1, import_y1, import_x2, import_y2)

    # --- 再次推理（导入视频按钮左侧，仅当已有视频时显示）---
    rerun_rect: Tuple[int, int, int, int] | None = None
    if has_video:
        rerun_text = "再次推理"
        rb = draw.textbbox((0, 0), rerun_text, font=font_small)
        rerun_w = max(108, int(rb[2] - rb[0] + 36))
        rerun_x2 = int(import_x1 - gap)
        rerun_x1 = int(rerun_x2 - rerun_w)
        rerun_y1 = int(margin_t)
        rerun_y2 = int(rerun_y1 + btn_h)
        _draw_btn(
            draw,
            rerun_x1,
            rerun_y1,
            rerun_x2,
            rerun_y2,
            rerun_text,
            font_small,
            hover(rerun_x1, rerun_y1, rerun_x2, rerun_y2),
            fill_normal=_rgba(_C["teal_dim"], 200),
            fill_hover=_rgba(_C["teal"], 230),
            outline_normal=_rgba(_C["teal_dim"], 160),
            outline_hover=_rgba(_C["teal"], 200),
            radius=btn_r,
        )
        rerun_rect = (rerun_x1, rerun_y1, rerun_x2, rerun_y2)

    # ── 视频区域播放/暂停叠层图标 ────────────────────────────────────────────
    if computed_video_rect is not None:
        vx, vy, vx_end, vy_end = computed_video_rect
        vc_x = int((vx + vx_end) // 2)
        vc_y = int((vy + vy_end) // 2)
        icon_r = 30
        is_hover_video = hover(vx, vy, vx_end, vy_end)

        if is_hover_video or not is_playing:
            a_circle = 190 if is_hover_video else 110
            draw.ellipse(
                [vc_x - icon_r, vc_y - icon_r, vc_x + icon_r, vc_y + icon_r],
                fill=(8, 10, 16, a_circle),
                outline=(255, 255, 255, a_circle),
                width=2,
            )
            if not is_playing:
                # ▶ 三角
                tri = [(vc_x - 7, vc_y - 13), (vc_x - 7, vc_y + 13), (vc_x + 16, vc_y)]
                draw.polygon(tri, fill=(235, 238, 248, 220))
            else:
                # ⏸ 双竖线
                bw2, bh2, bgap = 6, 20, 6
                lx = int(vc_x - bgap // 2 - bw2)
                rx = int(vc_x + bgap // 2)
                by2 = int(vc_y - bh2 // 2)
                draw.rectangle([lx, by2, lx + bw2, by2 + bh2], fill=(235, 238, 248, 220))
                draw.rectangle([rx, by2, rx + bw2, by2 + bh2], fill=(235, 238, 248, 220))

    # ── 底部结果卡 / 状态卡 ───────────────────────────────────────────────────
    bottom_margin = 28
    close_result_rect = None

    if result:
        label, prob = result
        prob_pct = int(prob * 100)

        if prob > 0.8:
            prob_color = _rgba(_C["success"])
        elif prob > 0.5:
            prob_color = _rgba(_C["warning"])
        else:
            prob_color = _rgba(_C["error"])

        lb = draw.textbbox((0, 0), label, font=font_main)
        lw = lb[2] - lb[0]
        pt_str = f"{prob_pct}%"
        pb = draw.textbbox((0, 0), pt_str, font=font_main)
        pw = pb[2] - pb[0]

        card_w = max(380, lw + pw + 80)
        card_h = 96
        card_x1 = int((tw - card_w) // 2)
        card_y1 = int(th - card_h - bottom_margin)
        card_x2 = int(card_x1 + card_w)
        card_y2 = int(card_y1 + card_h)

        # 结果卡背景（带左侧亮边）
        draw.rounded_rectangle(
            [card_x1, card_y1 + 4, card_x2, card_y2 + 4], radius=18, fill=(200, 200, 190, 80)
        )
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=18,
            fill=_rgba(_C["surface"], 230),
            outline=_rgba(_C["border_hi"], 160),
            width=2,
        )
        # 左侧彩色竖条
        bar_col = prob_color[:3] + (200,)
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x1 + 5, card_y2],
            radius=3,
            fill=bar_col,
        )

        # 文字
        ty = card_y1 + 20
        draw.text((card_x1 + 24, ty), label, font=font_main, fill=_rgba(_C["text"]))
        draw.text((card_x2 - 28 - pw, ty), pt_str, font=font_main, fill=prob_color)

        # 进度条
        bx1 = int(card_x1 + 24)
        bx2 = int(card_x2 - 24)
        bary = int(card_y2 - 22)
        draw.rounded_rectangle(
            [bx1, bary, bx2, bary + 7], radius=4, fill=_rgba(_C["surface2"], 255)
        )
        fill_w = int((bx2 - bx1) * prob)
        if fill_w > 0:
            draw.rounded_rectangle([bx1, bary, bx1 + fill_w, bary + 7], radius=4, fill=prob_color)

        # × 关闭按钮
        close_r = 16
        cx1 = int(card_x2 - close_r)
        cy1 = int(card_y1 - close_r)
        cx2 = int(cx1 + close_r * 2)
        cy2 = int(cy1 + close_r * 2)
        is_hov_close = hover(cx1, cy1, cx2, cy2)
        draw.ellipse(
            [cx1, cy1, cx2, cy2],
            fill=_rgba(_C["danger"], 220) if is_hov_close else _rgba(_C["danger_dim"], 200),
        )
        xb = draw.textbbox((0, 0), "×", font=font_small)
        draw.text(
            (
                int(cx1 + (close_r * 2 - (xb[2] - xb[0])) // 2),
                int(cy1 + (close_r * 2 - (xb[3] - xb[1])) // 2 - 1),
            ),
            "×",
            font=font_small,
            fill=(235, 238, 248, 255),
        )
        close_result_rect = (int(cx1), int(cy1), int(cx2), int(cy2))

    else:
        # 状态提示卡
        hint = status_text if status_text else "请点击上方「导入视频」选择文件..."
        hb = draw.textbbox((0, 0), hint, font=font_small)
        hw = hb[2] - hb[0]
        card_w = max(340, int(hw + 72))
        card_h_s = 68
        cx1s = int((tw - card_w) // 2)
        cy1s = int(th - card_h_s - bottom_margin)
        cx2s = int(cx1s + card_w)
        cy2s = int(cy1s + card_h_s)
        draw.rounded_rectangle(
            [cx1s, cy1s + 4, cx2s, cy2s + 4], radius=16, fill=(200, 200, 190, 80)
        )
        draw.rounded_rectangle(
            [cx1s, cy1s, cx2s, cy2s],
            radius=16,
            fill=_rgba(_C["surface"], 210),
            outline=_rgba(_C["border"], 140),
            width=2,
        )
        draw.text(
            (int(cx1s + (card_w - hw) // 2), int(cy1s + (card_h_s - (hb[3] - hb[1])) // 2 - 2)),
            hint,
            font=font_small,
            fill=_rgba(_C["text_dim"]),
        )

    # ── 合并 ─────────────────────────────────────────────────────────────────
    out = Image.alpha_composite(image, overlay)
    rendered = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)

    return (
        rendered,
        (exit_x1, exit_y1, exit_x2, exit_y2),
        (skel_x1, skel_y1, skel_x2, skel_y2),
        close_result_rect,
        import_new_rect,
        computed_video_rect,
        rerun_rect,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 鼠标回调
# ─────────────────────────────────────────────────────────────────────────────


def on_offline_mouse(event: int, x: int, y: int, flags: int, params: Any) -> None:
    """离线界面鼠标回调。"""
    if params is None:
        return

    if event == cv2.EVENT_MOUSEMOVE:
        params["mouse_pos"] = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    def _in(rect):
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    if _in(params.get("exit_offline_rect")):
        params["exit_offline"] = True
        return
    if _in(params.get("skel_rect")):
        params["show_skeleton"] = not params.get("show_skeleton", False)
        return
    if _in(params.get("import_new_rect")):
        params["import_new_video"] = True
        return
    if _in(params.get("rerun_rect")):
        params["rerun_inference"] = True
        return
    if _in(params.get("close_result_rect")):
        params["close_result"] = True
        return
    if _in(params.get("video_area_rect")):
        params["toggle_play"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────


def run_offline_mode(
    models: List[torch.nn.Module],
    preprocess_helper: PreprocessHelper,
    stats: Dict[str, Any] | None,
    device: torch.device,
    id_to_label: Dict[int, str],
    font_main,
    font_small,
    window_name: str,
    target_size: Tuple[int, int],
) -> None:
    """
    离线推理主循环。

    流程：
      1. 弹出文件对话框，用户取消则直接返回。
      2. 后台线程执行视频推理（_run_offline_inference）。
      3. 主线程可选播放视频帧，显示推理进度和结果。
      4. 结果可用 × 关闭；关闭后可继续「导入视频」进行下一次推理。
      5. 点击「退出」返回，由调用方恢复鼠标回调。
    """
    tw, th = target_size

    # 离线界面鼠标状态（整个离线会话共享）
    offline_mouse: Dict[str, Any] = {
        "mouse_pos": None,
        "show_skeleton": False,
        "exit_offline": False,
        "close_result": False,
        "import_new_video": False,
        "rerun_inference": False,
        "toggle_play": False,
        "exit_offline_rect": None,
        "skel_rect": None,
        "close_result_rect": None,
        "import_new_rect": None,
        "video_area_rect": None,
        "rerun_rect": None,
    }
    cv2.setMouseCallback(window_name, on_offline_mouse, offline_mouse)

    # 第一次弹出文件选择
    video_path = _open_file_dialog()
    if not video_path:
        return

    def _start_inference(path: str):
        """启动一次新推理，返回 (state, cap, fps)。"""
        state: Dict[str, Any] = {
            "status": "正在初始化...",
            "result": None,
            "done": False,
            "keypoints_by_frame": [],
        }
        threading.Thread(
            target=_run_offline_inference,
            args=(path, models, preprocess_helper, stats, device, id_to_label, state),
            daemon=True,
        ).start()
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        return state, cap, fps

    offline_state, cap_vid, vid_fps = _start_inference(video_path)

    offline_result: Tuple[str, float] | None = None
    current_video_frame: np.ndarray | None = None
    current_frame_idx: int = 0  # 当前播放帧索引（用于同步关键点）
    is_playing: bool = False
    last_frame_time: float = time.time()

    # 读取第一帧静止展示
    ret, first_frame = cap_vid.read()
    if ret:
        current_video_frame = first_frame
    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame_idx = 0

    try:
        while True:
            # ── 处理交互事件 ─────────────────────────────────────────────────
            if offline_mouse.get("exit_offline"):
                break

            if offline_mouse.get("toggle_play"):
                offline_mouse["toggle_play"] = False
                is_playing = not is_playing
                if is_playing:
                    last_frame_time = time.time()

            if offline_mouse.get("import_new_video"):
                offline_mouse["import_new_video"] = False
                new_path = _open_file_dialog()
                if new_path:
                    cap_vid.release()
                    offline_result = None
                    is_playing = False
                    current_frame_idx = 0
                    video_path = new_path
                    offline_state, cap_vid, vid_fps = _start_inference(video_path)
                    ret, first_frame = cap_vid.read()
                    if ret:
                        current_video_frame = first_frame
                    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frame_idx = 0

            if offline_mouse.get("rerun_inference"):
                offline_mouse["rerun_inference"] = False
                cap_vid.release()
                offline_result = None
                is_playing = False
                current_frame_idx = 0
                offline_state, cap_vid, vid_fps = _start_inference(video_path)
                ret, first_frame = cap_vid.read()
                if ret:
                    current_video_frame = first_frame
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_idx = 0

            if offline_mouse.get("close_result"):
                offline_mouse["close_result"] = False
                offline_result = None

            # ── 视频帧推进 ────────────────────────────────────────────────────
            if is_playing:
                now = time.time()
                if now - last_frame_time >= 1.0 / vid_fps:
                    ret, vframe = cap_vid.read()
                    if not ret:
                        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        current_frame_idx = 0
                        ret, vframe = cap_vid.read()
                    if ret:
                        current_video_frame = vframe
                        current_frame_idx += 1
                    last_frame_time = now

            # ── 同步推理结果 ──────────────────────────────────────────────────
            if offline_state.get("done") and offline_result is None:
                if offline_state.get("result"):
                    offline_result = offline_state["result"]

            # ── 当前帧对应的关键点（用于骨骼叠加）──────────────────────────
            kp_list = offline_state.get("keypoints_by_frame", [])
            current_kp: np.ndarray | None = None
            if kp_list and 0 <= current_frame_idx < len(kp_list):
                current_kp = kp_list[current_frame_idx]

            # ── 状态文字 ──────────────────────────────────────────────────────
            if offline_result:
                status = ""
            elif offline_state.get("done"):
                status = offline_state.get("status", "")
            else:
                status = offline_state.get("status", "正在处理视频...")

            # ── 渲染 ──────────────────────────────────────────────────────────
            (
                rendered,
                exit_rect,
                skel_rect,
                close_rect,
                import_new_rect,
                video_area_rect,
                rerun_rect,
            ) = draw_offline_ui(
                video_frame=current_video_frame,
                result=offline_result,
                status_text=status,
                font_main=font_main,
                font_small=font_small,
                mouse_pos=offline_mouse.get("mouse_pos"),
                show_skeleton=offline_mouse.get("show_skeleton", False),
                target_size=(tw, th),
                is_playing=is_playing,
                video_rect=offline_mouse.get("video_area_rect"),
                current_keypoints=current_kp,
                has_video=True,
            )

            offline_mouse["exit_offline_rect"] = exit_rect
            offline_mouse["skel_rect"] = skel_rect
            offline_mouse["close_result_rect"] = close_rect
            offline_mouse["import_new_rect"] = import_new_rect
            offline_mouse["video_area_rect"] = video_area_rect
            offline_mouse["rerun_rect"] = rerun_rect

            cv2.imshow(window_name, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                is_playing = not is_playing
                if is_playing:
                    last_frame_time = time.time()

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap_vid.release()


# ─────────────────────────────────────────────────────────────────────────────
# 独立启动入口（不经过 main.py 启动器）
# ─────────────────────────────────────────────────────────────────────────────


def run_offline_inference_standalone() -> None:
    """
    独立启动离线推理（不经过模式选择界面）。
    """
    import io as _io

    if sys.platform == "win32":
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        os.system("")

    from src.model.realtime_inference import (
        _load_shared_resources,
        load_chinese_font,
    )

    try:
        device, id_to_label, models, extractor, preprocess_helper, stats, font_main, font_small = (
            _load_shared_resources()
        )
    except FileNotFoundError:
        return

    extractor.close()

    window_name = "Sign Language Recognition System - Offline Mode"
    win_w, win_h = 960, 600
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)

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

    cv2.destroyAllWindows()
    print("已退出离线推理。")


if __name__ == "__main__":
    run_offline_inference_standalone()
