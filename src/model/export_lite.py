"""
ASL-LSTM 模型导出脚本

将训练好的 BiLSTMAttention 模型导出为 PyTorch Lite (.ptl) 格式，
供 Android 端使用。同时导出 Z-Score 标准化统计量。

使用方法:
    uv run python src/model/export_lite.py
    uv run python src/model/export_lite.py --checkpoint src/checkpoints/seed456_temporal_mask/best_model.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import BiLSTMAttention


def export_model_to_ptl(checkpoint_path: str, output_path: str) -> None:
    model = BiLSTMAttention()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.randn(1, cfg.SEQUENCE.max_frames, cfg.SEQUENCE.input_size)
    example_lengths = torch.tensor([cfg.SEQUENCE.max_frames], dtype=torch.long)

    class BiLSTMAttentionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, lengths):
            return self.model(x, lengths)

    wrapper = BiLSTMAttentionWrapper(model)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    optimized = torch._subclasses.meta_utils._get_tracing_state(
        [example_input, example_lengths],
        torch.jit.trace(wrapper, (example_input, example_lengths)),
    )

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(scripted)
        optimized._save_for_lite_interpreter(output_path)
    except Exception:
        scripted._save_for_lite_interpreter(output_path)

    print(f"模型已导出到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Verify
    with torch.no_grad():
        output = wrapper(example_input, example_lengths)
        print(f"验证输出形状: {output.shape}")
        probs = torch.softmax(output, dim=1)
        print(f"预测类别: {torch.argmax(probs, dim=1).item()}")
        print(f"最高概率: {probs.max().item():.4f}")


def export_zscore_stats(output_path: str) -> None:
    stats = {
        "mean": [-0.02456032891334202, -0.11962947312034139, -0.0008191781815231185, 0.002500323402822938],
        "std": [0.24632292725976973, 0.6917867353827447, 0.03799941547934166, 0.061761387958482336],
        "eps": 1e-6,
        "channels": ["x", "y", "dx", "dy"],
        "num_keypoints": 135,
        "input_size": 540,
        "max_frames": 90,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Z-Score 统计量已导出到: {output_path}")
    print(f"内容:")
    for key, value in stats.items():
        if key not in ("eps",):
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="导出 ASL-LSTM 模型为 PyTorch Lite 格式")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(cfg.PATHS.test_model_path),
        help="模型检查点路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 .ptl 文件路径 (默认: 同目录下 best_model.ptl)",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Z-Score 统计量输出 JSON 路径",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="跳过模型导出，仅导出统计量",
    )
    args = parser.parse_args()

    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(checkpoint_dir, "best_model.ptl")

    if args.stats_output is None:
        args.stats_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Android",
            "app",
            "src",
            "main",
            "assets",
            "zscore_stats.json",
        )

    if not args.skip_export:
        if not os.path.exists(args.checkpoint):
            print(f"错误: 检查点文件不存在: {args.checkpoint}")
            sys.exit(1)
        export_model_to_ptl(args.checkpoint, args.output)
    else:
        print("跳过模型导出 (--skip-export)")

    export_zscore_stats(args.stats_output)

    print("\n导出完成!")


if __name__ == "__main__":
    main()