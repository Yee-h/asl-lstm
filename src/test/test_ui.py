import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_chinese_font(size: int):
    # 回退：默认字体英文可用，中文可能无法正常显示
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except:
        return ImageFont.load_default()

font_main = load_chinese_font(32)
font_small = load_chinese_font(24)

def _draw_launcher_ui(
    width: int,
    height: int,
    font_main,
    font_small,
    mouse_pos,
):
    # 温馨暖色背景: 浅米色 (B, G, R)
    # RGB: 253, 246, 237 -> BGR: 237, 246, 253
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
    btn_radius = 28 # 圆润的按钮

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
    
    # 按钮阴影 (简单的向下的偏移阴影)
    draw.rounded_rectangle(
        [rt_x1, rt_y1+4, rt_x2, rt_y2+4], radius=btn_radius, fill=(200, 130, 110, 80)
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
        [of_x1, of_y1+4, of_x2, of_y2+4], radius=btn_radius, fill=(200, 160, 100, 80)
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
        fill=(255, 255, 255, 255), # 白色字体在暖黄色上有时不够显眼，也可以用深棕色
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
    cv2.imwrite("test_ui.png", rendered)

_draw_launcher_ui(800, 480, font_main, font_small, None)
