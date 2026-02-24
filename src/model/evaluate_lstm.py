import torch
import torch.nn as nn
import sys
import os
import io
import time
import argparse
from tqdm import tqdm


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")


# 将 src 添加到 python 路径以允许导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.core.labels import load_id_to_label_map
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders
from src.model.tta import build_hflip_tta_batch


def evaluate_model(
    use_tta_hflip: bool | None = None,
    model_path_override: str | None = None,
    verbose_report: bool | None = None,
    save_report: bool | None = None,
):
    """
    在测试数据集上评估保存的模型。

    Args:
        use_tta_hflip: 是否使用 TTA（None 则使用配置值）
        model_path_override: 模型路径覆盖（None 则使用配置值）
        verbose_report: 是否输出详细报告（None 则使用配置值）
        save_report: 是否保存报告（None 则使用配置值）
    """
    # 1. 设置设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    # 2. 加载数据
    print("正在加载测试数据...")
    _, _, test_loader = get_dataloaders()
    print(f"测试数据加载完成。批次数量: {len(test_loader)}")

    # 3. 初始化模型
    model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
    model_type = "BiLSTM+Attention" if cfg.MODEL.use_attention else "BiLSTM"
    print(f"模型架构: {model_type}")

    # 4. 加载权重（优先使用参数覆盖，其次使用配置）
    model_path = (
        model_path_override if model_path_override is not None else cfg.EVALUATION.model_path
    )

    print(f"正在加载模型: {model_path}")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("模型权重加载成功。")
    else:
        print(f"错误: 未在 {model_path} 找到模型文件")
        return

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 6. 评估循环
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    print("开始评估...")
    start_time = time.time()

    # 使用参数或配置值
    actual_use_tta = use_tta_hflip if use_tta_hflip is not None else cfg.EVALUATION.use_tta_hflip
    print(f"评估 TTA: {'开启' if actual_use_tta else '关闭'}")

    with torch.no_grad():
        # 单条覆盖式进度条（评估阶段）
        pbar = tqdm(
            total=len(test_loader),
            desc="评估中",
            ncols=100,
            dynamic_ncols=True,
            leave=False,
            ascii=False,
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        for inputs, labels, lengths in test_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)  # 确保标签是 LongTensor

            # 前向传播 (传递 lengths 参数)
            outputs = model(inputs, lengths)
            if actual_use_tta:
                flipped_inputs = build_hflip_tta_batch(inputs)
                flipped_outputs = model(flipped_inputs, lengths)
                outputs = 0.5 * (outputs + flipped_outputs)
            loss = criterion(outputs, labels)

            # 统计指标
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 存储用于计算指标
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 进度条信息
            current_loss = running_loss / total
            current_acc = 100 * correct / total
            pbar.set_postfix(
                ordered_dict={
                    "Loss": f"{current_loss:.4f}",
                    "Acc": f"{current_acc:.2f}%",
                },
                refresh=False,
            )
            pbar.update(1)

        pbar.close()

    end_time = time.time()
    duration = end_time - start_time

    # 7. 计算最终指标
    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    print("\n" + "=" * 40)
    print("评估结果")
    print("=" * 40)
    print(f"模型: {os.path.basename(model_path)}")
    print(f"测试集 Loss:     {avg_loss:.4f}")
    print(f"测试集准确率: {accuracy:.2f}%")
    print(f"样本总数: {total}")
    print(f"耗时:    {duration:.2f}s")
    print("=" * 40)

    # 8. 详细报告 (如果可用 sklearn)
    try:
        from sklearn.metrics import classification_report

        # 加载标签名称
        id_to_label = load_id_to_label_map(cfg.PATHS.label_map_path)
        target_names = [id_to_label.get(i, str(i)) for i in range(cfg.SEQUENCE.num_classes)]

        # 分类报告
        actual_verbose = (
            verbose_report if verbose_report is not None else cfg.EVALUATION.verbose_report
        )
        report = classification_report(
            all_labels,
            all_preds,
            target_names=target_names,
            digits=4,
            zero_division=0.0,
            output_dict=False,
        )
        if actual_verbose:
            print("\n详细分类报告:")
            print(report)

        # --- 保存报告到 logs 目录 ---
        actual_save = save_report if save_report is not None else cfg.EVALUATION.save_report
        if actual_save:
            log_dir = cfg.PATHS.log_dir
            os.makedirs(log_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"evaluation_report_{timestamp}.txt"
            log_path = os.path.join(log_dir, log_filename)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 40 + "\n")
                f.write("评估报告\n")
                f.write("=" * 40 + "\n")
                f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型: {model_path}\n")
                f.write(f"测试集 Loss:     {avg_loss:.4f}\n")
                f.write(f"测试集准确率: {accuracy:.2f}%\n")
                f.write(f"样本总数: {total}\n")
                f.write(f"耗时:    {duration:.2f}s\n")
                f.write(f"TTA:     {'开启' if actual_use_tta else '关闭'}\n")
                f.write("-" * 40 + "\n")
                f.write("详细分类报告:\n")
                f.write(str(report))

            print(f"\n详细评估报告已保存至: {log_path}")

    except ImportError:
        print("\n注意: 未安装 'scikit-learn'。跳过详细分类报告。")
    except Exception as e:
        print(f"\n无法生成详细报告: {e}")


if __name__ == "__main__":
    _configure_windows_console()

    parser = argparse.ArgumentParser(
        description="评估已训练模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置中的默认模型评估
  python evaluate_lstm.py

  # 指定模型路径评估
  python evaluate_lstm.py --model-path checkpoints/seed42/best_model.pth

  # 关闭 TTA
  python evaluate_lstm.py --no-tta

  # 不输出详细报告
  python evaluate_lstm.py --quiet
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"指定待评估模型路径（默认使用配置: {cfg.EVALUATION.model_path}）",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="关闭水平翻转 TTA",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="开启水平翻转 TTA",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不输出详细分类报告",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存评估报告到日志",
    )
    args = parser.parse_args()

    # 确定 TTA 设置
    use_tta = None
    if args.tta:
        use_tta = True
    elif args.no_tta:
        use_tta = False

    # 确定报告设置
    verbose = False if args.quiet else None
    save = False if args.no_save else None

    evaluate_model(
        use_tta_hflip=use_tta,
        model_path_override=args.model_path,
        verbose_report=verbose,
        save_report=save,
    )
