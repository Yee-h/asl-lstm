"""
main.py — 推理入口

双击或命令行运行此文件，进入推理模式选择界面。
支持：
  - 实时推理（摄像头）
  - 离线推理（导入视频文件）
"""

import sys
import os
import io

# Windows 控制台编码修正
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")

# 确保 src 可作为模块导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.realtime_inference import run_launcher


if __name__ == "__main__":
    run_launcher()
