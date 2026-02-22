import os
import sys
import tempfile
import unittest
from dataclasses import replace


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.model import runtime_overrides
from src.model.model_lstm import get_model


class TestRuntimeOverrides(unittest.TestCase):
    def setUp(self):
        self._training_backup = cfg.TRAINING
        self._model_backup = cfg.MODEL
        self._paths_backup = cfg.PATHS

    def tearDown(self):
        cfg.TRAINING = self._training_backup
        cfg.MODEL = self._model_backup
        cfg.PATHS = self._paths_backup

    def test_apply_runtime_overrides_updates_config_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.PATHS = replace(cfg.PATHS, model_save_dir=tmpdir)

            runtime_overrides.apply_runtime_overrides(
                seed=123,
                run_tag="exp_demo",
                epochs=12,
                learning_rate=1e-3,
                weight_decay=1e-4,
                dropout=0.28,
                label_smoothing=0.02,
                mixup_alpha=0.1,
            )

            self.assertEqual(cfg.TRAINING.seed, 123)
            self.assertEqual(cfg.TRAINING.num_epochs, 12)
            self.assertAlmostEqual(cfg.TRAINING.learning_rate, 1e-3)
            self.assertAlmostEqual(cfg.TRAINING.weight_decay, 1e-4)
            self.assertAlmostEqual(cfg.TRAINING.mixup_alpha, 0.1)
            self.assertAlmostEqual(cfg.MODEL.dropout, 0.28)
            self.assertAlmostEqual(cfg.MODEL.label_smoothing, 0.02)
            self.assertEqual(os.path.basename(cfg.PATHS.model_save_dir), "exp_demo")
            self.assertTrue(os.path.isdir(cfg.PATHS.model_save_dir))

    def test_apply_runtime_overrides_rejects_invalid_dropout(self):
        original_dropout = cfg.MODEL.dropout

        with self.assertRaises(ValueError):
            runtime_overrides.apply_runtime_overrides(dropout=1.2)

        self.assertEqual(cfg.MODEL.dropout, original_dropout)

    def test_runtime_dropout_override_applies_to_new_model(self):
        runtime_overrides.apply_runtime_overrides(dropout=0.28)

        model = get_model(use_attention=True)

        self.assertAlmostEqual(model.dropout_fc.p, 0.28)
        expected_lstm_dropout = 0.28 if cfg.MODEL.num_layers > 1 else 0.0
        self.assertAlmostEqual(model.lstm.dropout, expected_lstm_dropout)


if __name__ == "__main__":
    unittest.main()
