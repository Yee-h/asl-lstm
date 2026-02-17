import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import io
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
from src.model.training_utils import EarlyStopping, set_global_seed


def train():
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

    # --- 加载数据集 ---
    print("正在加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print("数据集加载完成。")

    # --- 初始化模型 ---
    # 根据配置选择 BiLSTM 或 BiLSTM+Attention 模型
    # 并将模型参数迁移到指定设备 (CPU/GPU)
    model = get_model(use_attention=cfg.MODEL.use_attention).to(device)
    model_type = "BiLSTM+Attention" if cfg.MODEL.use_attention else "BiLSTM"
    print(f"模型架构: {model_type}")
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 定义损失函数和优化器 ---
    # CrossEntropyLoss 适用于多分类任务
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.MODEL.label_smoothing)
    # 使用 Adam 优化器，学习率从配置文件获取 (加入 L2 正则化以减轻过拟合)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAINING.learning_rate,
        weight_decay=cfg.TRAINING.weight_decay,
    )

    # 学习率调度器：当验证集 Loss 连续 patience 轮不下降时，自动将学习率乘以 factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # 监控指标为 Loss，越小越好
        factor=cfg.TRAINING.scheduler_factor,  # 学习率衰减因子
        patience=cfg.TRAINING.scheduler_patience,  # 容忍多少轮 Loss 不下降
        min_lr=cfg.TRAINING.scheduler_min_lr,  # 最小学习率
    )

    early_stopper = None
    if cfg.TRAINING.early_stopping_enabled:
        metric_mode = "max" if cfg.TRAINING.early_stopping_metric == "val_acc" else "min"
        early_stopper = EarlyStopping(
            mode=metric_mode,
            patience=cfg.TRAINING.early_stopping_patience,
            min_delta=cfg.TRAINING.early_stopping_min_delta,
        )

    # --- 训练循环 ---
    best_acc = 0.0  # 记录验证集上的最高准确率

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

            # --- 前向传播 ---
            outputs = model(inputs, lengths)
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
                optimizer.zero_grad(set_to_none=True)  # 清空梯度

            # --- 统计训练指标 ---
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测值 (最大概率对应的索引)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 打印本轮训练结果
        print(f"Epoch [{epoch + 1}/{cfg.TRAINING.num_epochs}] 结果:")
        print(f"  训练集 Loss: {epoch_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  验证集 Loss: {val_loss:.4f} | 准确率: {val_acc:.2f}%")

        # --- 保存最佳模型 ---
        # 如果当前轮次的验证集准确率是最高的，则保存模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(cfg.PATHS.model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  检测到更好的模型，已保存至 {save_path}")

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

        # --- 更新学习率 (基于验证集 Loss) ---
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  当前学习率: {current_lr:.6f}")

        if early_stopper is not None:
            monitored_value = (
                val_acc if cfg.TRAINING.early_stopping_metric == "val_acc" else val_loss
            )
            if early_stopper.step(monitored_value):
                print(
                    f"  EarlyStopping 触发：{cfg.TRAINING.early_stopping_metric} 连续 {cfg.TRAINING.early_stopping_patience} 轮无改善"
                )
                break

    print("训练结束。")


if __name__ == "__main__":
    _configure_windows_console()
    train()
