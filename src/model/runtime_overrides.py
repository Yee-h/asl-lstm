from __future__ import annotations

import os
from dataclasses import replace

import src.config as cfg


def _validate_positive_int(name: str, value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值: {value}")
    return parsed


def _validate_positive_float(name: str, value: float) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值: {value}")
    return parsed


def _validate_non_negative_float(name: str, value: float) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} 不能小于 0，当前值: {value}")
    return parsed


def _validate_probability(name: str, value: float) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} 必须在 [0, 1) 区间，当前值: {value}")
    return parsed


def apply_runtime_overrides(
    *,
    seed: int | None = None,
    run_tag: str | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    label_smoothing: float | None = None,
    mixup_alpha: float | None = None,
) -> dict[str, str]:
    """在运行时覆盖训练配置，便于实验快速迭代。"""
    training_updates: dict[str, int | float] = {}
    model_updates: dict[str, float] = {}

    if epochs is not None:
        training_updates["num_epochs"] = _validate_positive_int("epochs", epochs)
    if learning_rate is not None:
        training_updates["learning_rate"] = _validate_positive_float("learning_rate", learning_rate)
    if weight_decay is not None:
        training_updates["weight_decay"] = _validate_non_negative_float(
            "weight_decay", weight_decay
        )
    if mixup_alpha is not None:
        training_updates["mixup_alpha"] = _validate_non_negative_float("mixup_alpha", mixup_alpha)

    if dropout is not None:
        model_updates["dropout"] = _validate_probability("dropout", dropout)
    if label_smoothing is not None:
        model_updates["label_smoothing"] = _validate_probability("label_smoothing", label_smoothing)

    if seed is not None:
        training_updates["seed"] = int(seed)

    if training_updates:
        cfg.TRAINING = replace(cfg.TRAINING, **training_updates)

    if model_updates:
        cfg.MODEL = replace(cfg.MODEL, **model_updates)

    if run_tag is not None:
        normalized_run_tag = run_tag.strip()
        if not normalized_run_tag:
            raise ValueError("run_tag 不能为空字符串")
        run_dir = os.path.join(cfg.PATHS.model_save_dir, normalized_run_tag)
        os.makedirs(run_dir, exist_ok=True)
        cfg.PATHS = replace(cfg.PATHS, model_save_dir=run_dir)

    return {
        "seed": str(cfg.TRAINING.seed),
        "epochs": str(cfg.TRAINING.num_epochs),
        "model_save_dir": cfg.PATHS.model_save_dir,
    }
