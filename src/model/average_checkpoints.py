import argparse
import os
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.checkpoint_utils import average_state_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="平均多个 epoch checkpoint 并导出新模型")
    parser.add_argument("--start-epoch", type=int, required=True, help="起始 epoch（含）")
    parser.add_argument("--end-epoch", type=int, required=True, help="结束 epoch（含）")
    parser.add_argument("--step", type=int, default=5, help="checkpoint 间隔，默认 5")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 checkpoint 路径，默认保存到 src/checkpoints/averaged_*.pth",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start_epoch > args.end_epoch:
        print("错误: --start-epoch 不能大于 --end-epoch")
        return 1

    selected_paths = []
    for epoch in range(args.start_epoch, args.end_epoch + 1, max(1, args.step)):
        checkpoint_path = os.path.join(cfg.PATHS.model_save_dir, f"lstm_epoch_{epoch}.pth")
        if os.path.exists(checkpoint_path):
            selected_paths.append(checkpoint_path)

    if len(selected_paths) < 2:
        print("错误: 可用 checkpoint 少于 2 个，无法执行平均")
        return 1

    print("将参与平均的 checkpoint:")
    for path in selected_paths:
        print(f"  - {path}")

    states = [torch.load(path, map_location="cpu") for path in selected_paths]
    averaged_state = average_state_dicts(states)

    output_path = args.output
    if not output_path:
        output_name = f"averaged_{args.start_epoch}_{args.end_epoch}.pth"
        output_path = os.path.join(cfg.PATHS.model_save_dir, output_name)

    torch.save(averaged_state, output_path)
    print(f"已输出平均模型: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
