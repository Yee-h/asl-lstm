"""
批量评估脚本：遍历 src/checkpoints 下所有子目录中的 best_model.pth，
对测试集进行推理，保存每个模型的评估结果。

排除根级 best_model.pth 和 best_model copy.pth，仅使用各子目录下的 best_model.pth。
"""

import argparse
import io
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.config as cfg
from src.core.labels import load_id_to_label_map
from src.model.dataloader import CSLDataset
from src.model.model_lstm import BiLSTMAttention, Attention, MultiHeadAttention


# ──────────────────────────── 实验特殊配置 ────────────────────────────
# key=子目录名, value=需要覆盖的模型/数据参数
# accel6ch: 6通道特征 (x,y,dx,dy,ddx,ddy) → input_size=810, 特殊统计量
# bilstm192: hidden_size=192
# multihead: num_heads=4 (E06实验)
EXPERIMENT_OVERRIDES: dict[str, dict] = {
    "seed456_accel6ch": {
        "input_size": 810,
        "enable_accel_feature": True,
        "feature_stats_path": os.path.join(
            cfg.PATHS.processed_data_dir, "WLASL100_train_stats_6ch.json"
        ),
    },
    "seed456_bilstm192": {
        "hidden_size": 192,
    },
    "seed456_multihead": {
        "num_heads": 4,
    },
}


def _detect_model_params(state_dict: dict) -> dict:
    """从 checkpoint 的 state_dict 推断模型参数。"""
    params: dict = {}

    # input_size
    params["input_size"] = state_dict["lstm.weight_ih_l0"].shape[1]

    # hidden_size
    params["hidden_size"] = state_dict["lstm.weight_ih_l0"].shape[0] // 4

    # num_layers: count LSTM layer indices
    layer_indices: set[int] = set()
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            try:
                layer_indices.add(int(key.split("_l")[-1].split("_")[0]))
            except (ValueError, IndexError):
                layer_indices.add(int(key.split("_l")[-1]))
        elif key.startswith("lstm.weight_hh_l"):
            try:
                layer_indices.add(int(key.split("_l")[-1].split("_")[0]))
            except (ValueError, IndexError):
                layer_indices.add(int(key.split("_l")[-1]))
    params["num_layers"] = len(layer_indices) if layer_indices else 2

    # bidirectional
    params["bidirectional"] = any("_reverse" in key for key in state_dict)

    # num_classes
    params["num_classes"] = state_dict["fc.weight"].shape[0]

    # dropout (from LSTM internal dropout - approximate)
    params["dropout"] = cfg.MODEL.dropout

    # use_layer_norm
    params["use_layer_norm"] = "layer_norm.weight" in state_dict

    # attention type detection
    if "attention.q_proj.weight" in state_dict:
        params["use_multihead"] = True
        # multihead: num_heads can't be detected from state_dict
        # default to 4 unless overridden
        params["num_heads"] = 4
        params["attention_dim"] = 32  # not used in multihead, placeholder
    else:
        params["use_multihead"] = False
        params["num_heads"] = 1
        if "attention.attention_fc.weight" in state_dict:
            params["attention_dim"] = state_dict["attention.attention_fc.weight"].shape[0]
        else:
            params["attention_dim"] = cfg.MODEL.attention_dim

    return params


def create_model_from_params(params: dict) -> BiLSTMAttention:
    """根据推断出的参数创建模型实例。"""
    model = BiLSTMAttention(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_classes=params["num_classes"],
        dropout=params["dropout"],
        attention_dim=params["attention_dim"],
        use_layer_norm=params["use_layer_norm"],
        num_heads=params["num_heads"],
    )
    # Override bidirectional if needed (BiLSTMAttention reads from cfg)
    model.bidirectional = params["bidirectional"]
    return model


def _get_model_config(model_dir_name: str, detected_params: dict) -> dict:
    """根据目录名和检测结果，合并最终模型配置。

    检测到的参数优先，因为它们是 checkpoint 的实际结构。
    目录名覆盖用于检测不到的参数（如 num_heads, enable_accel_feature）。
    """
    merged = dict(detected_params)

    overrides = EXPERIMENT_OVERRIDES.get(model_dir_name, {})
    for key, value in overrides.items():
        if key in ("enable_accel_feature", "feature_stats_path"):
            continue  # 这些不传给模型构造函数
        merged[key] = value

    return merged


def evaluate_single_model(
    model_path: str,
    model_dir_name: str,
    device: torch.device,
    output_dir: str,
    save_predictions: bool = True,
) -> dict | None:
    """评估单个模型，返回指标字典，None 表示失败。"""
    print(f"\n{'=' * 60}")
    print(f"评估模型: {model_dir_name}")
    print(f"路径: {model_path}")
    print(f"{'=' * 60}")

    # ── 1. 加载 checkpoint 并推断参数 ──
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  [错误] 无法加载 checkpoint: {e}")
        return None

    detected_params = _detect_model_params(state_dict)
    model_params = _get_model_config(model_dir_name, detected_params)

    # ── 2. 创建模型并加载权重 ──
    try:
        model = create_model_from_params(model_params)
        param_count = sum(p.numel() for p in model.parameters())
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  [错误] 加载模型权重失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"  参数量: {param_count:,}  "
          f"input={model_params['input_size']}, "
          f"hidden={model_params['hidden_size']}, "
          f"layers={model_params['num_layers']}, "
          f"heads={model_params['num_heads']}, "
          f"ln={model_params['use_layer_norm']}")

    # ── 3. 处理特殊配置的数据加载与推理 ──
    overrides = EXPERIMENT_OVERRIDES.get(model_dir_name, {})
    need_special_data = "enable_accel_feature" in overrides

    orig_sequence = cfg.SEQUENCE
    orig_preprocess = cfg.PREPROCESS

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confidences: list[float] = []
    avg_loss = 0.0
    accuracy = 0.0
    top5_accuracy = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    duration = 0.0

    try:
        if need_special_data:
            cfg.SEQUENCE = replace(cfg.SEQUENCE, enable_accel_feature=True)
            cfg.PREPROCESS = replace(
                cfg.PREPROCESS, feature_stats_path=overrides["feature_stats_path"],
            )

        test_dataset = CSLDataset(
            cfg.PATHS.test_data_path, cfg.PATHS.label_map_path, augment=False,
        )
        print(f"  测试集样本数: {len(test_dataset)}")

        test_loader = DataLoader(
            test_dataset, batch_size=cfg.TRAINING.batch_size,
            shuffle=False, num_workers=0, pin_memory=False,
        )

        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(
                total=len(test_loader), desc=f"  {model_dir_name}",
                ncols=100, dynamic_ncols=True, leave=False, ascii=False, file=sys.stderr,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

            for inputs, labels, lengths in test_loader:
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, dim=1)

                # Top-5
                k = min(5, probs.size(1))
                _, top5_preds = torch.topk(probs, k, dim=1)
                top5_correct += (labels.view(-1, 1) == top5_preds).any(dim=1).sum().item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_confidences.extend(max_probs.cpu().tolist())

                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix(
                    ordered_dict={"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"},
                    refresh=False,
                )
                pbar.update(1)

            pbar.close()

        duration = time.time() - start_time
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        top5_accuracy = 100.0 * top5_correct / total

    finally:
        cfg.SEQUENCE = orig_sequence
        cfg.PREPROCESS = orig_preprocess

    print(f"  Top-1: {accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%  Loss: {avg_loss:.4f}  "
          f"正确/总数: {correct}/{total}  耗时: {duration:.2f}s")

    # ── 5. 构建结果 ──
    result: dict = {
        "model_name": model_dir_name,
        "model_path": model_path,
        "test_loss": round(float(avg_loss), 6),
        "test_accuracy": round(float(accuracy), 4),
        "test_top5_accuracy": round(float(top5_accuracy), 4),
        "correct": correct,
        "top5_correct": top5_correct,
        "total": total,
        "param_count": param_count,
        "duration_seconds": round(float(duration), 2),
        "params": {
            "input_size": model_params["input_size"],
            "hidden_size": model_params["hidden_size"],
            "num_layers": model_params["num_layers"],
            "bidirectional": model_params["bidirectional"],
            "num_classes": model_params["num_classes"],
            "use_layer_norm": model_params["use_layer_norm"],
            "num_heads": model_params["num_heads"],
        },
    }

    # 分类报告
    try:
        from sklearn.metrics import classification_report

        id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
        target_names = [
            id_to_label.get(i, str(i)) for i in range(model_params["num_classes"])
        ]
        report = classification_report(
            all_labels, all_preds, target_names=target_names,
            digits=4, zero_division=0.0, output_dict=True,
        )
        result["classification_report"] = report
    except Exception as e:
        print(f"  [警告] 无法生成分类报告: {e}")

    # ── 6. 保存逐样本预测 ──
    if save_predictions:
        predictions_file = os.path.join(output_dir, "predictions.json")
        id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
        predictions_list: list[dict] = []
        for i in range(len(all_labels)):
            predictions_list.append({
                "sample_idx": i,
                "true_label_id": all_labels[i],
                "true_label": id_to_label.get(all_labels[i], str(all_labels[i])),
                "predicted_label_id": all_preds[i],
                "predicted_label": id_to_label.get(all_preds[i], str(all_preds[i])),
                "confidence": round(float(all_confidences[i]), 6),
                "correct": all_preds[i] == all_labels[i],
            })
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions_list, f, ensure_ascii=False, indent=2)
        print(f"  逐样本预测已保存至: {predictions_file}")

    # ── 7. 保存汇总结果 ──
    result_file = os.path.join(output_dir, "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  评估结果已保存至: {result_file}")

    return result


def evaluate_ensemble(
    model_paths: list[str],
    ensemble_name: str,
    device: torch.device,
    output_dir: str,
    save_predictions: bool = True,
) -> dict | None:
    """多模型集成评估（softmax 概率平均）。

    Args:
        model_paths: 模型检查点路径列表
        ensemble_name: 集成名称
        device: 计算设备
        output_dir: 结果输出目录
        save_predictions: 是否保存逐样本预测
    """
    print(f"\n{'=' * 60}")
    print(f"集成评估: {ensemble_name}")
    print(f"模型数: {len(model_paths)}")
    for p in model_paths:
        print(f"  - {p}")
    print(f"{'=' * 60}")

    models: list[BiLSTMAttention] = []
    total_param_count = 0
    for i, model_path in enumerate(model_paths):
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            detected = _detect_model_params(state_dict)
            model_params = _get_model_config(
                os.path.basename(os.path.dirname(model_path)), detected
            )
            model = create_model_from_params(model_params)
            total_param_count += sum(p.numel() for p in model.parameters())
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            models.append(model)
            print(f"  [模型 {i+1}] {os.path.basename(os.path.dirname(model_path))} 加载成功 "
                  f"(input={model_params['input_size']}, "
                  f"hidden={model_params['hidden_size']}, "
                  f"heads={model_params['num_heads']})")
        except Exception as e:
            print(f"  [错误] 加载模型失败 {model_path}: {e}")
            return None

    # 所有集成模型必须使用相同的架构参数（由第一个模型决定数据加载方式）
    first_params = _detect_model_params(
        torch.load(model_paths[0], map_location="cpu", weights_only=True)
    )
    first_name = os.path.basename(os.path.dirname(model_paths[0]))
    first_overrides = EXPERIMENT_OVERRIDES.get(first_name, {})
    need_special_data = "enable_accel_feature" in first_overrides

    orig_sequence = cfg.SEQUENCE
    orig_preprocess = cfg.PREPROCESS

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confidences: list[float] = []
    avg_loss = 0.0
    accuracy = 0.0
    top5_accuracy = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    duration = 0.0

    try:
        if need_special_data:
            cfg.SEQUENCE = replace(cfg.SEQUENCE, enable_accel_feature=True)
            cfg.PREPROCESS = replace(
                cfg.PREPROCESS, feature_stats_path=first_overrides["feature_stats_path"]
            )

        test_dataset = CSLDataset(
            cfg.PATHS.test_data_path, cfg.PATHS.label_map_path, augment=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.TRAINING.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(
                total=len(test_loader),
                desc=f"  {ensemble_name}",
                ncols=100,
                dynamic_ncols=True,
                leave=False,
                ascii=False,
                file=sys.stderr,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

            for inputs, labels, lengths in test_loader:
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                batch_probs: list[torch.Tensor] = []
                for model in models:
                    outputs = model(inputs, lengths)
                    batch_probs.append(torch.softmax(outputs, dim=1))

                # softmax 概率平均
                avg_probs = torch.stack(batch_probs).mean(dim=0)
                max_probs, predicted = torch.max(avg_probs, dim=1)

                # Top-5
                k = min(5, avg_probs.size(1))
                _, top5_preds = torch.topk(avg_probs, k, dim=1)
                top5_correct += (labels.view(-1, 1) == top5_preds).any(dim=1).sum().item()

                # 用平均概率计算 Loss（与真实标签）
                log_avg_probs = torch.log(avg_probs.clamp(min=1e-9))
                loss = criterion(log_avg_probs, labels)

                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_confidences.extend(max_probs.cpu().tolist())

                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix(
                    ordered_dict={"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"},
                    refresh=False,
                )
                pbar.update(1)

            pbar.close()

        duration = time.time() - start_time
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        top5_accuracy = 100.0 * top5_correct / total

    finally:
        cfg.SEQUENCE = orig_sequence
        cfg.PREPROCESS = orig_preprocess

    print(f"  Top-1: {accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%  Loss: {avg_loss:.4f}  "
          f"正确/总数: {correct}/{total}  耗时: {duration:.2f}s")

    result: dict = {
        "model_name": ensemble_name,
        "ensemble_models": model_paths,
        "num_models": len(models),
        "test_loss": round(float(avg_loss), 6),
        "test_accuracy": round(float(accuracy), 4),
        "test_top5_accuracy": round(float(top5_accuracy), 4),
        "correct": correct,
        "top5_correct": top5_correct,
        "total": total,
        "param_count": total_param_count,
        "duration_seconds": round(float(duration), 2),
        "params": {
            "input_size": int(first_params.get("input_size", 0)),
            "hidden_size": int(first_params.get("hidden_size", 0)),
            "num_classes": int(first_params.get("num_classes", 0)),
        },
    }

    try:
        from sklearn.metrics import classification_report
        id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
        num_classes = int(first_params.get("num_classes", 100))
        target_names = [id_to_label.get(i, str(i)) for i in range(num_classes)]
        report = classification_report(
            all_labels, all_preds, target_names=target_names,
            digits=4, zero_division=0.0, output_dict=True,
        )
        result["classification_report"] = report
    except Exception as e:
        print(f"  [警告] 无法生成分类报告: {e}")

    if save_predictions:
        predictions_file = os.path.join(output_dir, "predictions.json")
        id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
        predictions_list: list[dict] = []
        for i in range(len(all_labels)):
            predictions_list.append({
                "sample_idx": i,
                "true_label_id": all_labels[i],
                "true_label": id_to_label.get(all_labels[i], str(all_labels[i])),
                "predicted_label_id": all_preds[i],
                "predicted_label": id_to_label.get(all_preds[i], str(all_preds[i])),
                "confidence": round(float(all_confidences[i]), 6),
                "correct": all_preds[i] == all_labels[i],
            })
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions_list, f, ensure_ascii=False, indent=2)
        print(f"  逐样本预测已保存至: {predictions_file}")

    result_file = os.path.join(output_dir, "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  评估结果已保存至: {result_file}")

    return result


def main() -> None:
    _configure_windows_console()

    parser = argparse.ArgumentParser(description="批量评估所有子目录下的 best_model.pth")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出根目录（默认: project_root/results）",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="不保存逐样本预测（减少文件大小）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cuda/cpu，默认自动选择)",
    )
    args = parser.parse_args()

    # 设置设备
    if args.device:
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            print(f"[警告] 无效设备 '{args.device}'，使用自动选择")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    # 结果输出目录
    project_root = cfg.PATHS.project_root
    output_root = args.output_dir or os.path.join(project_root, "results")
    os.makedirs(output_root, exist_ok=True)
    print(f"结果输出目录: {output_root}")

    # 检查点目录
    checkpoints_dir = cfg.PATHS.model_save_dir
    if not os.path.isdir(checkpoints_dir):
        print(f"[错误] checkpoints 目录不存在: {checkpoints_dir}")
        sys.exit(1)

    # 收集所有子目录下的 best_model.pth（排除根级文件）
    model_dirs: list[tuple[str, str]] = []  # (dir_name, model_path)
    with os.scandir(checkpoints_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            best_path = os.path.join(entry.path, "best_model.pth")
            if os.path.isfile(best_path):
                model_dirs.append((entry.name, best_path))

    model_dirs.sort()
    print(f"\n找到 {len(model_dirs)} 个待评估模型:")
    for name, path in model_dirs:
        print(f"  - {name}")

    # 逐模型评估
    all_results: list[dict] = []
    failed_models: list[str] = []

    for name, model_path in model_dirs:
        model_output_dir = os.path.join(output_root, name)
        os.makedirs(model_output_dir, exist_ok=True)

        try:
            result = evaluate_single_model(
                model_path=model_path,
                model_dir_name=name,
                device=device,
                output_dir=model_output_dir,
                save_predictions=not args.no_save_predictions,
            )
        except Exception as e:
            print(f"\n  [严重错误] {name} 评估异常: {e}")
            import traceback
            traceback.print_exc()
            result = None

        if result is None:
            failed_models.append(name)
        else:
            all_results.append(result)

    # ── 集成评估：4 模型 softmax 概率平均 ──
    ensemble_name = "ensemble_temporal_mask"
    ensemble_paths = [
        os.path.join(checkpoints_dir, "seed42_temporal_mask", "best_model.pth"),
        os.path.join(checkpoints_dir, "seed123_temporal_mask", "best_model.pth"),
        os.path.join(checkpoints_dir, "seed456_temporal_mask", "best_model.pth"),
        os.path.join(checkpoints_dir, "seed789_temporal_mask", "best_model.pth"),
    ]
    all_exist = all(os.path.isfile(p) for p in ensemble_paths)
    if all_exist:
        ensemble_output_dir = os.path.join(output_root, ensemble_name)
        os.makedirs(ensemble_output_dir, exist_ok=True)
        try:
            ensemble_result = evaluate_ensemble(
                model_paths=ensemble_paths,
                ensemble_name=ensemble_name,
                device=device,
                output_dir=ensemble_output_dir,
                save_predictions=not args.no_save_predictions,
            )
            if ensemble_result is not None:
                all_results.append(ensemble_result)
        except Exception as e:
            print(f"\n  [严重错误] 集成评估异常: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[警告] 集成模型文件不完整，跳过集成评估")

    # ── 保存汇总结果 ──
    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")

    summary: dict[str, object] = {
        "total_models": len(model_dirs) + (1 if all_exist else 0),
        "successful": len(all_results),
        "failed": len(failed_models),
        "failed_models": failed_models,
        "results": sorted(all_results, key=lambda r: r["test_accuracy"], reverse=True),
    }

    summary_file = os.path.join(output_root, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存至: {summary_file}")

    # 控制台输出排名
    print(f"\n{'排名':<5} {'模型':<35} {'准确率':<12} {'Loss':<10} {'耗时':<10}")
    print("-" * 75)
    for rank, r in enumerate(summary["results"], start=1):  # type: ignore[arg-type]
        name = str(r["model_name"])  # type: ignore[index]
        acc = r["test_accuracy"]  # type: ignore[index]
        loss = r["test_loss"]  # type: ignore[index]
        dur = r["duration_seconds"]  # type: ignore[index]
        print(f"{rank:<5} {name:<35} {acc:>8.2f}%   {loss:>8.4f}   {dur:>6.1f}s")

    if failed_models:
        print(f"\n失败模型 ({len(failed_models)}): {', '.join(failed_models)}")

    print("\n批量评估完成。")


if __name__ == "__main__":
    main()
