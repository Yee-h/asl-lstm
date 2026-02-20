import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.model.train_lstm import build_training_profile


class TestTrainingProfile(unittest.TestCase):
    def test_overfit_debug_profile_disables_regularization_and_augmentation(self):
        profile = build_training_profile(overfit_debug=True)

        self.assertEqual(profile["label_smoothing"], 0.0)
        self.assertEqual(profile["weight_decay"], 0.0)
        self.assertFalse(profile["train_augment"])
        self.assertFalse(profile["use_weighted_sampler"])
        self.assertTrue(profile["disable_dropout"])
        self.assertFalse(profile["use_val_scheduler"])
        self.assertFalse(profile["use_early_stopping"])
        self.assertFalse(profile["use_ema"])
        self.assertFalse(profile["use_eval_tta_hflip"])
        self.assertFalse(profile["use_swa"])

    def test_default_profile_keeps_current_strategy(self):
        profile = build_training_profile(overfit_debug=False)

        self.assertIn("label_smoothing", profile)
        self.assertIn("weight_decay", profile)
        self.assertIn("train_augment", profile)
        self.assertIn("use_weighted_sampler", profile)
        self.assertIn("disable_dropout", profile)
        self.assertIn("use_val_scheduler", profile)
        self.assertIn("use_early_stopping", profile)
        self.assertIn("use_ema", profile)
        self.assertIn("use_eval_tta_hflip", profile)
        self.assertIn("use_swa", profile)
        self.assertIn("swa_start_epoch", profile)
        self.assertIn("swa_lr", profile)


if __name__ == "__main__":
    unittest.main()
