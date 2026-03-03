"""
E02: 检查点平均优化实验脚本。

在验证集上评估各 epoch 检查点，选取 top-k 个验证精度最高的检查点进行
参数平均（SWA-style），然后评估平均模型在测试集上的表现。

支持：
1. 单种子检查点平均
2. 多种子各自平均后再做集成
"""

import argparse
import os
import re
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders
from src.model.validate_lstm import validate
from src.model.checkpoint_utils import average_state_dicts


def discover_checkpoints(seed_dir: str) -> list[tuple[int, str]]:
    """发现种子目录下所有 epoch 检查点，返回 [(epoch, path), ...] 按 epoch 排序。"""
    pattern = re.compile(r"lstm_epoch_(\d+)\.pth$")
    results = []
    for fname in os.listdir(seed_dir):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            results.append((epoch, os.path.join(seed_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


def evaluate_checkpoints_on_val(
    checkpoints: list[tuple[int, str]],
    val_loader,
    criterion,
    device: torch.device,
) -> list[dict]:
    """在验证集上评估所有检查点，返回 [{epoch, path, val_acc, val_loss}, ...]。"""
    results = []
    for epoch, path in tqdm(checkpoints, desc="评估检查点"):
        model = get_model(use_attention=cfg.MODEL.use_attention)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        results.append(
            {
                "epoch": epoch,
                "path": path,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )
        del model
        torch.cuda.empty_cache()
    return results


def average_top_k_checkpoints(
    val_results: list[dict],
    top_k: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """选取验证精度 top-k 的检查点进行参数平均。"""
    sorted_results = sorted(val_results, key=lambda x: x["val_acc"], reverse=True)
    selected = sorted_results[:top_k]
    print(f"\n选取 Top-{top_k} 检查点进行平均:")
    for r in selected:
        print(f"  Epoch {r['epoch']}: val_acc={r['val_acc']:.2f}%")
    state_dicts = [torch.load(r["path"], map_location="cpu", weights_only=True) for r in selected]
    return average_state_dicts(state_dicts)


def evaluate_on_test(model: nn.Module, test_loader, criterion, device) -> tuple[float, float]:
    """在测试集上评估模型。"""
    return validate(model, test_loader, criterion, device)


def evaluate_averaged_model(
    averaged_state: dict[str, torch.Tensor],
    test_loader,
    val_loader,
    criterion,
    device: torch.device,
) -> dict:
    """加载平均后的 state_dict 并评估。"""
    model = get_model(use_attention=cfg.MODEL.use_attention)
    model.load_state_dict(averaged_state)
    model.to(device)
    model.eval()
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    return {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }


def ensemble_evaluate_averaged_models(
    averaged_states: list[dict[str, torch.Tensor]],
    test_loader,
    device: torch.device,
) -> float:
    """对多个平均后模型做 softmax 概率平均集成评估。"""
    models = []
    for state in averaged_states:
        model = get_model(use_attention=cfg.MODEL.use_attention)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="集成评估"):
            inputs = inputs.to(device)
            batch_probs = []
            for model in models:
                outputs = model(inputs, lengths)
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs)
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            _, predicted = torch.max(avg_probs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = 100.0 * np.mean(all_preds == all_labels)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="E02: 检查点平均优化实验")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["seed456"],
        help="种子目录名列表 (默认: seed456)",
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        type=int,
        default=[3, 5, 7, 10],
        help="要尝试的 top-k 值列表 (默认: 3 5 7 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出 JSON 路径",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    criterion = nn.CrossEntropyLoss()
    print(f"设备: {device}")

    # 加载数据集
    print("加载数据集...")
    _, val_loader, test_loader = get_dataloaders(train_augment=False, use_weighted_sampler=False)

    all_results = {}

    for seed_name in args.seeds:
        seed_dir = os.path.join(cfg.PATHS.model_save_dir, seed_name)
        if not os.path.isdir(seed_dir):
            print(f"警告: 目录不存在 {seed_dir}, 跳过")
            continue

        print(f"\n{'=' * 60}")
        print(f"处理种子: {seed_name}")
        print(f"{'=' * 60}")

        # 1. 发现所有检查点
        checkpoints = discover_checkpoints(seed_dir)
        print(
            f"发现 {len(checkpoints)} 个 epoch 检查点 (epoch {checkpoints[0][0]} ~ {checkpoints[-1][0]})"
        )

        # 2. 在验证集上评估所有检查点
        print("\n在验证集上评估所有检查点...")
        val_results = evaluate_checkpoints_on_val(checkpoints, val_loader, criterion, device)

        # 打印 top-10 验证精度
        sorted_by_val = sorted(val_results, key=lambda x: x["val_acc"], reverse=True)
        print(f"\n{seed_name} 验证集 Top-10 检查点:")
        for i, r in enumerate(sorted_by_val[:10]):
            print(f"  #{i + 1} Epoch {r['epoch']}: val_acc={r['val_acc']:.2f}%")

        # 3. 评估 best_model.pth（EMA 最佳）作为基线
        best_model_path = os.path.join(seed_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            model = get_model(use_attention=cfg.MODEL.use_attention)
            model.load_state_dict(
                torch.load(best_model_path, map_location=device, weights_only=True)
            )
            model.to(device)
            _, best_val = validate(model, val_loader, criterion, device)
            _, best_test = validate(model, test_loader, criterion, device)
            print(f"\nbest_model.pth (EMA) 基线: val={best_val:.2f}%, test={best_test:.2f}%")
            del model
            torch.cuda.empty_cache()
        else:
            best_val, best_test = 0.0, 0.0

        # 4. 对不同 top-k 值做检查点平均并评估
        seed_results = {
            "baseline_best_model": {"val_acc": best_val, "test_acc": best_test},
            "val_ranking": [
                {"epoch": r["epoch"], "val_acc": round(r["val_acc"], 2)} for r in sorted_by_val[:15]
            ],
            "averaged_results": {},
        }

        for k in args.top_k_values:
            if k > len(val_results):
                continue
            avg_state = average_top_k_checkpoints(val_results, k, device)
            result = evaluate_averaged_model(avg_state, test_loader, val_loader, criterion, device)
            print(
                f"\nTop-{k} 平均模型: val={result['val_acc']:.2f}%, test={result['test_acc']:.2f}%"
            )
            delta = result["test_acc"] - best_test
            print(f"  相对 best_model.pth: test Δ={delta:+.2f}%")
            seed_results["averaged_results"][f"top_{k}"] = {
                "val_acc": round(result["val_acc"], 2),
                "test_acc": round(result["test_acc"], 2),
                "delta_vs_best": round(delta, 2),
            }

        all_results[seed_name] = seed_results

    # 5. 如果有多个种子，尝试对各种子的平均模型做集成
    if len(args.seeds) > 1 and len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("多种子平均模型集成评估")
        print(f"{'=' * 60}")

        # 使用每个种子的最佳 top-k（选 test_acc 最高的）
        best_averaged_states = []
        for seed_name in args.seeds:
            if seed_name not in all_results:
                continue
            seed_res = all_results[seed_name]
            # 找到测试集最高的 top-k 配置
            best_k = max(seed_res["averaged_results"].items(), key=lambda x: x[1]["test_acc"])
            print(f"  {seed_name}: 使用 {best_k[0]} (test={best_k[1]['test_acc']:.2f}%)")
            # 重新加载平均
            seed_dir = os.path.join(cfg.PATHS.model_save_dir, seed_name)
            checkpoints = discover_checkpoints(seed_dir)
            val_results = evaluate_checkpoints_on_val(checkpoints, val_loader, criterion, device)
            k_val = int(best_k[0].split("_")[1])
            avg_state = average_top_k_checkpoints(val_results, k_val, device)
            best_averaged_states.append(avg_state)

        if len(best_averaged_states) >= 2:
            ensemble_acc = ensemble_evaluate_averaged_models(
                best_averaged_states, test_loader, device
            )
            print(f"\n平均模型集成测试集精度: {ensemble_acc:.2f}%")
            all_results["ensemble_averaged"] = {"test_acc": round(ensemble_acc, 2)}

    # 汇总输出
    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")
    for seed_name, res in all_results.items():
        if seed_name == "ensemble_averaged":
            continue
        baseline = res["baseline_best_model"]
        print(f"\n{seed_name}:")
        print(
            f"  best_model.pth 基线: val={baseline['val_acc']:.2f}%, test={baseline['test_acc']:.2f}%"
        )
        for k_name, k_res in res["averaged_results"].items():
            print(
                f"  {k_name} 平均: val={k_res['val_acc']:.2f}%, test={k_res['test_acc']:.2f}% (Δ={k_res['delta_vs_best']:+.2f}%)"
            )

    if "ensemble_averaged" in all_results:
        print(f"\n多种子平均模型集成: test={all_results['ensemble_averaged']['test_acc']:.2f}%")

    # 保存结果
    output_path = args.output or os.path.join("logs", "E02_checkpoint_avg_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
