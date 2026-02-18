import os
import sys
import unittest

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.model.training_utils as training_utils
from src.model.training_utils import EarlyStopping, ModelEMA, set_global_seed


class TestEarlyStopping(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self):
        stopper = EarlyStopping(mode="max", patience=2, min_delta=0.0)

        self.assertFalse(stopper.step(0.60))
        self.assertFalse(stopper.step(0.59))
        self.assertTrue(stopper.step(0.58))

    def test_improvement_resets_counter(self):
        stopper = EarlyStopping(mode="min", patience=2, min_delta=0.0)

        self.assertFalse(stopper.step(2.0))
        self.assertFalse(stopper.step(2.1))
        self.assertFalse(stopper.step(1.9))
        self.assertFalse(stopper.step(1.95))
        self.assertTrue(stopper.step(2.05))

    def test_set_global_seed_can_control_cudnn_flags(self):
        set_global_seed(
            42,
            deterministic=True,
            benchmark=False,
            use_deterministic_algorithms=False,
        )

        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_model_ema_updates_floating_params(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        ema = ModelEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))

        ema.update(model)

        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        self.assertTrue(torch.allclose(ema.module.weight, expected, atol=1e-6))

    def test_champion_gate_helper_removed(self):
        self.assertFalse(hasattr(training_utils, "should_promote_model"))


if __name__ == "__main__":
    unittest.main()
