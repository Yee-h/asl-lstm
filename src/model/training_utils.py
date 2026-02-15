import random
import os

import numpy as np
import torch


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = False,
) -> None:
    """Set random seeds and determinism flags for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic and benchmark:
        benchmark = False

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

    torch.use_deterministic_algorithms(
        bool(use_deterministic_algorithms),
        warn_only=True,
    )


class EarlyStopping:
    """Simple early stopping utility based on a monitored scalar metric."""

    def __init__(self, mode: str = "max", patience: int = 10, min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best_value: float | None = None
        self.bad_epochs = 0

    def _is_improved(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        return value < (self.best_value - self.min_delta)

    def step(self, value: float) -> bool:
        """
        Update tracker with latest metric value.

        Returns True when training should stop.
        """
        current = float(value)
        if self._is_improved(current):
            self.best_value = current
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
