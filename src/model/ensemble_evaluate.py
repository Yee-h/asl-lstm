"""多种子集成评估脚本。

加载多个模型（可以是不同种子训练的 best_model 或检查点平均模型），
对每个样本的 softmax 概率取平均后投票，输出集成预测的准确率。

用法:
    python -m src.model.ensemble_evaluate \
        --model-paths path/to/model1.pth path/to/model2.pth path/to/model3.pth
"""

import argparse
import io
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多种子集成评估")
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=True,
        help="参与集成的模型路径列表",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="评估的数据集划分 (默认: all — 同时评估 train/val/test)",
    )
    return parser.parse_args()


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """加载单个模型。"""
    model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _evaluate_ensemble_on_loader(
    models: list[torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, int, int]:
    """在给定 loader 上评估集成模型。返回 (accuracy%, correct, total)。"""
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            inputs, labels, lengths = batch[0].to(device), batch[1].to(device), batch[2]

            # 收集每个模型的 softmax 概率
            probs_sum = None
            for model in models:
                logits = model(inputs, lengths)
                probs = F.softmax(logits, dim=1)
                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum = probs_sum + probs

            # 平均概率投票
            avg_probs = probs_sum / len(models)  # type: ignore[operator]
            _, predicted = avg_probs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main() -> int:
    _configure_windows_console()
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"设备: {device}")
    print(f"集成模型数量: {len(args.model_paths)}")

    # 验证路径
    for mp in args.model_paths:
        if not os.path.exists(mp):
            print(f"错误: 模型文件不存在: {mp}")
            return 1
        print(f"  - {mp}")

    # 加载所有模型
    print("正在加载模型...")
    models = [_load_model(mp, device) for mp in args.model_paths]
    print(f"已加载 {len(models)} 个模型")

    # 加载数据
    print("正在加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print("数据集加载完成。")

    # 评估
    start_time = time.time()

    splits_to_eval: dict[str, torch.utils.data.DataLoader] = {}
    if args.split in ("train", "all"):
        splits_to_eval["train"] = train_loader
    if args.split in ("val", "all"):
        splits_to_eval["val"] = val_loader
    if args.split in ("test", "all"):
        splits_to_eval["test"] = test_loader

    print("\n" + "=" * 50)
    print("集成评估结果")
    print("=" * 50)

    results = {}
    for split_name, loader in splits_to_eval.items():
        acc, correct, total = _evaluate_ensemble_on_loader(models, loader, device)
        results[split_name] = acc
        print(f"  {split_name:>5s}: {acc:.2f}% ({correct}/{total})")

    elapsed = time.time() - start_time
    print(f"\n耗时: {elapsed:.2f}s")
    print("=" * 50)

    # 同时评估单模型用于对比
    print("\n--- 单模型对比 ---")
    for i, (mp, model) in enumerate(zip(args.model_paths, models)):
        model_name = os.path.basename(mp)
        for split_name, loader in splits_to_eval.items():
            acc, correct, total = _evaluate_ensemble_on_loader([model], loader, device)
            print(f"  模型 {i + 1} ({model_name}) {split_name}: {acc:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
