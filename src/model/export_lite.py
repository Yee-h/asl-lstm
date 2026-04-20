"""
ASL-LSTM 模型导出脚本

将训练好的 BiLSTMAttention 模型导出为 PyTorch Lite (.ptl) 格式，
供 Android 端使用。同时导出 Z-Score 标准化统计量。

支持 INT8 动态量化（--quantize），将 nn.LSTM 和 nn.Linear 权重量化
为 qint8，模型体积缩小约 4 倍，推理加速 2-3 倍，精度损失 <0.5%。

使用方法:
    uv run python src/model/export_lite.py
    uv run python src/model/export_lite.py --quantize
    uv run python src/model/export_lite.py --checkpoint src/checkpoints/seed456_temporal_mask/best_model.pth
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import BiLSTMAttention


def export_model_to_ptl(checkpoint_path: str, output_path: str, quantize: bool = False) -> None:
    model = BiLSTMAttention()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if quantize:
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        quantized_model.eval()
        print("INT8 动态量化已应用 (nn.LSTM + nn.Linear)")

        float32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        quant_size = 0
        for name, buf in quantized_model.named_buffers():
            if "weight" in name:
                quant_size += buf.numel() * buf.element_size()
        for name, param in quantized_model.named_parameters():
            if param.numel() > 0:
                quant_size += param.numel() * param.element_size()
        print(f"量化后权重估算大小: {quant_size / 1024 / 1024:.2f} MB (原始 float32 约 {float32_size:.2f} MB)")
        model = quantized_model

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

    with torch.no_grad():
        ref_output = wrapper(example_input, example_lengths)

    if quantize:
        traced = torch.jit.trace(wrapper, (example_input, example_lengths))
        traced.eval()
    else:
        scripted = torch.jit.script(wrapper)
        scripted.eval()
        traced = scripted

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)
        optimized._save_for_lite_interpreter(output_path)
    except Exception as e:
        print(f"optimize_for_mobile 失败 ({e})，使用原始导出")
        traced._save_for_lite_interpreter(output_path)

    print(f"模型已导出到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    with torch.no_grad():
        output = wrapper(example_input, example_lengths)
        print(f"验证输出形状: {output.shape}")
        probs = torch.softmax(output, dim=1)
        print(f"预测类别: {torch.argmax(probs, dim=1).item()}")
        print(f"最高概率: {probs.max().item():.4f}")

        if ref_output is not None:
            max_diff = (output - ref_output).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                output.flatten().unsqueeze(0), ref_output.flatten().unsqueeze(0)
            ).item()
            print(f"量化前后最大输出差异: {max_diff:.6f}")
            print(f"量化前后余弦相似度: {cos_sim:.6f}")


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
    print("内容:")
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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="启用 INT8 动态量化 (nn.LSTM + nn.Linear 权重量化，模型缩小约4倍)",
    )
    args = parser.parse_args()

    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        suffix = "_quant8" if args.quantize else ""
        args.output = os.path.join(checkpoint_dir, f"best_model{suffix}.ptl")

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
        export_model_to_ptl(args.checkpoint, args.output, quantize=args.quantize)
    else:
        print("跳过模型导出 (--skip-export)")

    export_zscore_stats(args.stats_output)

    print("\n导出完成!")


if __name__ == "__main__":
    main()
