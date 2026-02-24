"""
多种子模型集成评估脚本。

支持多个模型的 softmax 概率平均集成，用于提升测试集泛化稳定性。
"""

import argparse
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders
from src.core.labels import load_id_to_label_map


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """加载单个模型检查点。"""
    model = get_model(use_attention=cfg.MODEL.use_attention)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def ensemble_evaluate(
    model_paths: list[str],
    use_tta_hflip: bool = False,
) -> dict:
    """
    多模型集成评估（softmax 概率平均）。

    Args:
        model_paths: 模型检查点路径列表
        use_tta_hflip: 是否使用水平翻转 TTA

    Returns:
        包含准确率、预测详情等信息的字典
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"设备: {device}")
    print(f"加载 {len(model_paths)} 个模型...")

    models = [load_model(p, device) for p in model_paths]

    print("加载数据集...")
    _, _, test_loader = get_dataloaders(train_augment=False, use_weighted_sampler=False)

    all_labels = []
    all_predictions = []
    all_probs = []

    print("开始集成评估...")
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="评估中"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 收集所有模型的预测概率
            batch_probs = []
            for model in models:
                outputs = model(inputs, lengths)
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs)

                if use_tta_hflip:
                    # 水平翻转 TTA
                    flipped_inputs = flip_keypoints(inputs)
                    flipped_outputs = model(flipped_inputs, lengths)
                    flipped_probs = F.softmax(flipped_outputs, dim=1)
                    batch_probs.append(flipped_probs)

            # 概率平均
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            _, predicted = torch.max(avg_probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(avg_probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = 100.0 * np.mean(all_predictions == all_labels)

    id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)

    # 计算每个类别的准确率
    class_correct = Counter()
    class_total = Counter()
    for label, pred in zip(all_labels, all_predictions):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    class_accuracies = {}
    for cls in sorted(class_total.keys()):
        cls_name = id_to_label.get(str(cls), f"class_{cls}")
        class_accuracies[cls_name] = {
            "correct": class_correct[cls],
            "total": class_total[cls],
            "accuracy": 100.0 * class_correct[cls] / class_total[cls],
        }

    return {
        "accuracy": accuracy,
        "total_samples": len(all_labels),
        "correct": int(np.sum(all_predictions == all_labels)),
        "class_accuracies": class_accuracies,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def flip_keypoints(inputs: torch.Tensor) -> torch.Tensor:
    """
    水平翻转关键点（用于 TTA）。

    注意：需要根据实际关键点定义进行左右交换。
    这里仅作简单 x 坐标翻转，实际使用时需要补充左右交换逻辑。
    """
    flipped = inputs.clone()
    # 简单翻转 x 坐标（假设 x 在偶数索引位置）
    # 实际应根据关键点布局进行正确的左右交换
    flipped[:, :, 0::4] = -flipped[:, :, 0::4]  # x -> -x
    return flipped


def print_results(results: dict, model_names: list[str] | None = None):
    """打印评估结果。"""
    print("\n" + "=" * 60)
    print("集成评估结果")
    print("=" * 60)
    print(f"模型数量: {len(model_names) if model_names else 'N/A'}")
    if model_names:
        print("模型列表:")
        for name in model_names:
            print(f"  - {name}")
    print("-" * 60)
    print(f"总样本数: {results['total_samples']}")
    print(f"正确预测: {results['correct']}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print("-" * 60)

    # 打印前10个类别的准确率
    print("\n各类别准确率（前10个）:")
    for i, (cls_name, stats) in enumerate(list(results["class_accuracies"].items())[:10]):
        print(f"  {cls_name}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    if len(results["class_accuracies"]) > 10:
        print(f"  ... (共 {len(results['class_accuracies'])} 个类别)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模型集成评估")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="模型检查点路径列表",
    )
    parser.add_argument(
        "--use-tta-hflip",
        action="store_true",
        help="使用水平翻转 TTA",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件路径（JSON 格式）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 验证模型路径
    for path in args.models:
        if not os.path.exists(path):
            print(f"错误: 模型文件不存在 - {path}")
            sys.exit(1)

    # 提取模型名称用于显示
    model_names = [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in args.models]

    results = ensemble_evaluate(
        model_paths=args.models,
        use_tta_hflip=args.use_tta_hflip,
    )

    print_results(results, model_names)

    # 保存结果
    if args.output:
        import json

        output_results = {
            "accuracy": results["accuracy"],
            "total_samples": results["total_samples"],
            "correct": results["correct"],
            "models": args.models,
            "use_tta_hflip": args.use_tta_hflip,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
