import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import sys
import os
import io
import argparse
import importlib.util
import matplotlib.pyplot as plt
from tqdm import tqdm


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")


# 检查是否可以绘图，否则使用非交互式后端
if importlib.util.find_spec("tkinter") is None:
    plt.switch_backend("agg")

# 将 src 目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.model.model_lstm import get_model
from src.model.dataloader import get_dataloaders
from src.model.validate_lstm import validate
from src.model.training_utils import EarlyStopping, ModelEMA, set_global_seed


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sequence-level Mixup: 对输入序列和标签进行线性插值混合。

    Args:
        x: 输入特征 (batch, seq_len, input_size)
        y: 标签 (batch,)
        lengths: 有效长度 (batch,)
        alpha: Beta 分布参数（越大混合越强）

    Returns:
        mixed_x: 混合后的输入
        y_a: 原始标签
        y_b: 混合对标签
        mixed_lengths: 混合后的有效长度（取两者最大值）
        lam: 混合系数 lambda
    """
    if alpha <= 0:
        return x, y, y, lengths, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index.to(x.device)]
    y_a, y_b = y, y[index.to(y.device)]
    # lengths 始终在 CPU（pack_padded_sequence 要求），index 也用 CPU 索引
    mixed_lengths = torch.max(lengths, lengths[index])

    return mixed_x, y_a, y_b, mixed_lengths, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixup 损失: lambda * L(pred, y_a) + (1-lambda) * L(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    """Focal Loss — 对易分类样本降权，聚焦难分类样本。

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: 聚焦参数，gamma=0 等价于 CrossEntropyLoss。推荐 1.0~2.0。
        label_smoothing: 标签平滑系数，与 CrossEntropyLoss 用法一致。
    """

    def __init__(self, gamma: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 使用 CrossEntropyLoss 计算 per-sample log(p_t)（含 label_smoothing）
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t = exp(-CE)
        pt = torch.exp(-ce_loss)
        # Focal 权重
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()


def build_training_profile(overfit_debug: bool = False) -> dict[str, float | bool]:
    """构建训练策略配置，可按需切换到过拟合诊断模式。"""
    profile: dict[str, float | bool] = {
        "label_smoothing": float(cfg.MODEL.label_smoothing),
        "weight_decay": float(cfg.TRAINING.weight_decay),
        "train_augment": True,
        "use_weighted_sampler": bool(cfg.TRAINING.use_weighted_sampler),
        "disable_dropout": False,
        "use_val_scheduler": True,
        "use_early_stopping": bool(cfg.TRAINING.early_stopping_enabled),
        "use_ema": bool(cfg.TRAINING.use_ema),
        "ema_decay": float(cfg.TRAINING.ema_decay),
        "ema_start_epoch": int(cfg.TRAINING.ema_start_epoch),
        "use_eval_tta_hflip": bool(cfg.TRAINING.eval_use_tta_hflip),
        "mixup_alpha": float(cfg.TRAINING.mixup_alpha),
        "use_swa": bool(cfg.TRAINING.use_swa),
        "swa_start_epoch": int(cfg.TRAINING.swa_start_epoch),
        "swa_lr": float(cfg.TRAINING.swa_lr),
        "use_focal_loss": bool(cfg.MODEL.use_focal_loss),
        "focal_gamma": float(cfg.MODEL.focal_gamma),
    }

    if overfit_debug:
        profile.update(
            {
                "label_smoothing": 0.0,
                "weight_decay": 0.0,
                "train_augment": False,
                "use_weighted_sampler": False,
                "disable_dropout": True,
                "use_val_scheduler": False,
                "use_early_stopping": False,
                "use_ema": False,
                "use_eval_tta_hflip": False,
                "mixup_alpha": 0.0,
                "use_swa": False,
                "use_focal_loss": False,
            }
        )

    return profile


def _disable_model_dropout(model: nn.Module) -> None:
    """关闭模型中的 Dropout，用于过拟合可达性诊断。"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    lstm = getattr(model, "lstm", None)
    if lstm is not None and hasattr(lstm, "dropout"):
        lstm.dropout = 0.0


def _evaluate_no_tta_val_acc(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """统一使用无 TTA 指标作为跨轮次模型晋升基准。"""
    return validate(model, val_loader, criterion, device, use_tta_hflip=False)


def train(overfit_debug: bool = False):
    """
    模型训练主函数。
    """
    set_global_seed(
        cfg.TRAINING.seed,
        deterministic=cfg.TRAINING.deterministic,
        benchmark=cfg.TRAINING.cudnn_benchmark,
        use_deterministic_algorithms=cfg.TRAINING.use_deterministic_algorithms,
    )

    # --- 设置训练设备 ---
    # 优先使用 GPU，如果不可用或配置指定使用 CPU，则退回 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.TRAINING.device == "cuda" else "cpu"
    )
    print(f"当前使用的设备: {device}")

    training_profile = build_training_profile(overfit_debug=overfit_debug)
    if overfit_debug:
        print("[诊断模式] 已启用 overfit-debug：关闭增强、采样重权、Dropout、正则与标签平滑")
    print(
        "训练策略: "
        f"augment={training_profile['train_augment']}, "
        f"weighted_sampler={training_profile['use_weighted_sampler']}, "
        f"label_smoothing={training_profile['label_smoothing']}, "
        f"weight_decay={training_profile['weight_decay']}, "
        f"val_scheduler={training_profile['use_val_scheduler']}, "
        f"early_stopping={training_profile['use_early_stopping']}, "
        f"ema={training_profile['use_ema']}, "
        f"eval_tta={training_profile['use_eval_tta_hflip']}, "
        f"mixup_alpha={training_profile['mixup_alpha']}, "
        f"swa={training_profile['use_swa']}"
    )

    # --- 加载数据集 ---
    print("正在加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_augment=bool(training_profile["train_augment"]),
        use_weighted_sampler=bool(training_profile["use_weighted_sampler"]),
    )
    print("数据集加载完成。")

    # --- 初始化模型 ---
    # 根据配置选择 BiLSTM 或 BiLSTM+Attention 模型
    # 并将模型参数迁移到指定设备 (CPU/GPU)
    model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
    if bool(training_profile["disable_dropout"]):
        _disable_model_dropout(model)
    model_type = "BiLSTM+Attention" if cfg.MODEL.use_attention else "BiLSTM"
    print(f"模型架构: {model_type}")
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 定义损失函数和优化器 ---
    if bool(training_profile["use_focal_loss"]):
        criterion = FocalLoss(
            gamma=float(training_profile["focal_gamma"]),
            label_smoothing=float(training_profile["label_smoothing"]),
        )
        print(
            f"损失函数: FocalLoss(gamma={training_profile['focal_gamma']}, "
            f"label_smoothing={training_profile['label_smoothing']})"
        )
    else:
        # CrossEntropyLoss 适用于多分类任务
        criterion = nn.CrossEntropyLoss(label_smoothing=float(training_profile["label_smoothing"]))
        print(f"损失函数: CrossEntropyLoss(label_smoothing={training_profile['label_smoothing']})")
    # 使用 Adam 优化器，学习率从配置文件获取 (加入 L2 正则化以减轻过拟合)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAINING.learning_rate,
        weight_decay=float(training_profile["weight_decay"]),
    )

    ema_tracker = None
    if bool(training_profile["use_ema"]):
        ema_tracker = ModelEMA(model, decay=float(training_profile["ema_decay"]))
    ema_start_epoch = max(1, int(training_profile["ema_start_epoch"]))

    # 学习率调度器：根据配置选择 ReduceLROnPlateau 或 CosineAnnealingWarmRestarts
    scheduler = None
    scheduler_type = (
        cfg.TRAINING.scheduler_type if bool(training_profile["use_val_scheduler"]) else None
    )
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # 监控指标为 Loss，越小越好
            factor=cfg.TRAINING.scheduler_factor,  # 学习率衰减因子
            patience=cfg.TRAINING.scheduler_patience,  # 容忍多少轮 Loss 不下降
            min_lr=cfg.TRAINING.scheduler_min_lr,  # 最小学习率
        )
    elif scheduler_type == "cosine_warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.TRAINING.cosine_T0,  # 初始重启周期
            T_mult=cfg.TRAINING.cosine_T_mult,  # 周期倍增因子
            eta_min=cfg.TRAINING.scheduler_min_lr,  # 最小学习率
        )
        print(
            f"学习率调度器: CosineAnnealingWarmRestarts (T0={cfg.TRAINING.cosine_T0}, T_mult={cfg.TRAINING.cosine_T_mult})"
        )
    elif scheduler_type is not None:
        raise ValueError(
            f"未知的学习率调度器类型: {scheduler_type!r}（支持 'plateau' 或 'cosine_warm'）"
        )

    early_stopper = None
    if bool(training_profile["use_early_stopping"]):
        metric_mode = "max" if cfg.TRAINING.early_stopping_metric == "val_acc" else "min"
        early_stopper = EarlyStopping(
            mode=metric_mode,
            patience=cfg.TRAINING.early_stopping_patience,
            min_delta=cfg.TRAINING.early_stopping_min_delta,
        )

    # --- SWA 初始化 ---
    swa_model = None
    swa_scheduler = None
    swa_start = int(training_profile["swa_start_epoch"])
    if bool(training_profile["use_swa"]):
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=float(training_profile["swa_lr"]),
            anneal_strategy="linear",
            anneal_epochs=5,
        )
        print(f"SWA 已启用: start_epoch={swa_start}, swa_lr={training_profile['swa_lr']}")

    best_model_path = os.path.join(cfg.PATHS.model_save_dir, "best_model.pth")
    best_acc = -1.0

    # 记录训练过程中的 Loss 和 Accuracy
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("开始训练...")

    accum_steps = max(1, int(cfg.TRAINING.grad_accum_steps))
    print(f"梯度累积步数: {accum_steps} (等效 Batch Size: {cfg.TRAINING.batch_size * accum_steps})")

    for epoch in range(cfg.TRAINING.num_epochs):
        model.train()  # 设置为训练模式 (开启 Dropout 等)
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 创建单条覆盖式进度条（每个 epoch 仅一条）
        # ascii=False 启用 Unicode 方块样式；dynamic_ncols 让宽度自适应终端
        pbar = tqdm(
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1}/{cfg.TRAINING.num_epochs}]",
            ncols=100,
            dynamic_ncols=True,
            leave=False,
            ascii=False,
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, labels, lengths) in enumerate(train_loader):
            # 迁移数据到设备
            inputs = inputs.to(device)
            # 确保标签是 LongTensor 类型，多分类任务的要求
            labels = labels.long().to(device)

            # --- Mixup 数据混合（Sequence-level）---
            use_mixup = float(training_profile["mixup_alpha"]) > 0
            if use_mixup:
                inputs, labels_a, labels_b, lengths, lam = mixup_data(
                    inputs,
                    labels,
                    lengths,
                    alpha=float(training_profile["mixup_alpha"]),
                )
            else:
                labels_a = labels_b = labels
                lam = 1.0

            # --- 前向传播 ---
            outputs = model(inputs, lengths)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            scaled_loss = loss / accum_steps

            # --- 反向传播和优化 ---
            scaled_loss.backward()  # 计算梯度

            should_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                if cfg.TRAINING.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.TRAINING.grad_clip_max_norm
                    )
                optimizer.step()  # 更新参数
                if ema_tracker is not None and (epoch + 1) >= ema_start_epoch:
                    ema_tracker.update(model)
                optimizer.zero_grad(set_to_none=True)  # 清空梯度

            # --- 统计训练指标 ---
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测值 (最大概率对应的索引)
            total += labels_a.size(0)
            # Mixup 下用主标签近似统计准确率
            correct += (predicted == labels_a).sum().item()

            # 在进度条上实时更新损失和准确率信息
            current_loss = running_loss / total
            current_acc = 100 * correct / total
            pbar.set_postfix(
                ordered_dict={
                    "Loss": f"{current_loss:.4f}",
                    "Acc": f"{current_acc:.2f}%",
                },
                refresh=False,
            )  # refresh=False 避免强制刷新
            pbar.update(1)

        pbar.close()

        # 计算 Epoch 平均 Loss 和准确率
        epoch_loss = running_loss / total
        train_acc = 100 * correct / total

        # --- 验证环节 ---
        # 每一轮训练结束后，在验证集上评估模型性能
        eval_model = model
        if ema_tracker is not None and (epoch + 1) >= ema_start_epoch:
            eval_model = ema_tracker.module

        val_loss, val_acc = validate(
            eval_model,
            val_loader,
            criterion,
            device,
            use_tta_hflip=bool(training_profile["use_eval_tta_hflip"]),
        )

        selection_acc = val_acc
        if bool(training_profile["use_eval_tta_hflip"]):
            _, selection_acc = _evaluate_no_tta_val_acc(eval_model, val_loader, criterion, device)

        # 打印本轮训练结果
        print(f"Epoch [{epoch + 1}/{cfg.TRAINING.num_epochs}] 结果:")
        print(f"  训练集 Loss: {epoch_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  验证集 Loss: {val_loss:.4f} | 准确率: {val_acc:.2f}%")
        if bool(training_profile["use_eval_tta_hflip"]):
            print(f"  验证集准确率(无TTA门禁): {selection_acc:.2f}%")

        # --- 保存当前训练最优模型 ---
        if selection_acc > best_acc:
            best_acc = selection_acc
            torch.save(eval_model.state_dict(), best_model_path)
            print(f"  检测到更好的模型，已保存至 {best_model_path}")

        # --- 每五轮保存一次模型 ---
        if (epoch + 1) % cfg.TRAINING.save_every_n_epochs == 0:
            periodic_save_path = os.path.join(
                cfg.PATHS.model_save_dir, f"lstm_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), periodic_save_path)
            print(f"  定期保存模型至 {periodic_save_path}")

        # --- 更新 Loss 图片 ---
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy Curve")

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.PATHS.model_save_dir, "training_metrics.png"))
        plt.close()
        print("  训练曲线已更新")

        # --- 更新学习率 ---
        in_swa_phase = swa_model is not None and (epoch + 1) > swa_start
        if in_swa_phase:
            # SWA 阶段：使用 SWALR 调度器（固定小学习率）
            swa_model.update_parameters(model)
            swa_scheduler.step()  # type: ignore[union-attr]
        elif scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)  # type: ignore[arg-type]  # ReduceLROnPlateau: 基于验证集 Loss
            else:
                scheduler.step(epoch + 1)  # type: ignore[arg-type]  # CosineAnnealing: 基于 epoch 计数
        current_lr = optimizer.param_groups[0]["lr"]
        swa_status = " [SWA]" if in_swa_phase else ""
        print(f"  当前学习率: {current_lr:.6f}{swa_status}")

        if early_stopper is not None:
            monitored_value = (
                selection_acc if cfg.TRAINING.early_stopping_metric == "val_acc" else val_loss
            )
            if early_stopper.step(monitored_value):
                print(
                    f"  EarlyStopping 触发：{cfg.TRAINING.early_stopping_metric} 连续 {cfg.TRAINING.early_stopping_patience} 轮无改善"
                )
                break

    # --- SWA 收尾：保存 SWA 平均模型 ---
    if swa_model is not None:
        swa_save_path = os.path.join(cfg.PATHS.model_save_dir, "swa_model.pth")
        # 提取 SWA 平均后的 state_dict（去掉 AveragedModel 的 "module." 前缀）
        swa_state = swa_model.module.state_dict()
        torch.save(swa_state, swa_save_path)
        print(f"SWA 平均模型已保存至 {swa_save_path}")

    print("训练结束。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 BiLSTM / BiLSTM+Attention 模型")
    parser.add_argument(
        "--overfit-debug",
        action="store_true",
        help="启用过拟合诊断模式（关闭增强与正则，验证模型可拟合能力）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _configure_windows_console()
    args = parse_args()
    train(overfit_debug=args.overfit_debug)
