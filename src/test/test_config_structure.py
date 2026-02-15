import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg


class TestConfigStructure(unittest.TestCase):
    def test_grouped_config_objects_exist(self):
        self.assertTrue(hasattr(cfg, "PATHS"))
        self.assertTrue(hasattr(cfg, "SEQUENCE"))
        self.assertTrue(hasattr(cfg, "PREPROCESS"))
        self.assertTrue(hasattr(cfg, "MODEL"))
        self.assertTrue(hasattr(cfg, "AUGMENTATION"))
        self.assertTrue(hasattr(cfg, "TRAINING"))
        self.assertTrue(hasattr(cfg, "INFERENCE"))
        self.assertTrue(hasattr(cfg, "UI"))

    def test_legacy_alias_removed(self):
        self.assertFalse(hasattr(cfg, "BATCH_SIZE"))
        self.assertFalse(hasattr(cfg, "LEARNING_RATE"))
        self.assertFalse(hasattr(cfg, "TRAIN_DATA_PATH"))
        self.assertFalse(hasattr(cfg, "USE_ATTENTION"))

    def test_grouped_values_have_expected_type(self):
        self.assertIsInstance(cfg.TRAINING.batch_size, int)
        self.assertIsInstance(cfg.TRAINING.learning_rate, float)
        self.assertIsInstance(cfg.PATHS.train_data_path, str)
        self.assertIsInstance(cfg.MODEL.use_attention, bool)

    def test_training_strategy_fields_exist(self):
        self.assertGreaterEqual(cfg.TRAINING.grad_accum_steps, 1)
        self.assertGreaterEqual(cfg.TRAINING.early_stopping_patience, 1)


if __name__ == "__main__":
    unittest.main()
