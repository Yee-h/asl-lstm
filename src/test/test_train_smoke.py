import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.model import train_lstm


class TestTrainSmoke(unittest.TestCase):
    def _build_tiny_loader(self) -> DataLoader:
        sample_count = 4
        inputs = torch.randn(
            sample_count,
            cfg.SEQUENCE.max_frames,
            cfg.SEQUENCE.input_size,
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        lengths = torch.tensor([cfg.SEQUENCE.max_frames] * sample_count, dtype=torch.long)
        dataset = TensorDataset(inputs, labels, lengths)
        return DataLoader(dataset, batch_size=2, shuffle=False)

    def test_train_entry_minimal_epoch_smoke(self):
        train_loader = self._build_tiny_loader()
        val_loader = self._build_tiny_loader()
        test_loader = self._build_tiny_loader()

        original_num_epochs = cfg.TRAINING.num_epochs
        original_save_every = cfg.TRAINING.save_every_n_epochs
        original_device = cfg.TRAINING.device
        original_model_dir = cfg.PATHS.model_save_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            object.__setattr__(cfg.TRAINING, "num_epochs", 1)
            object.__setattr__(cfg.TRAINING, "save_every_n_epochs", 1)
            object.__setattr__(cfg.TRAINING, "device", "cpu")
            object.__setattr__(cfg.PATHS, "model_save_dir", tmpdir)

            try:
                with patch.object(
                    train_lstm,
                    "get_dataloaders",
                    return_value=(train_loader, val_loader, test_loader),
                ):
                    train_lstm.train()

                periodic_model = os.path.join(tmpdir, "lstm_epoch_1.pth")
                metrics_png = os.path.join(tmpdir, "training_metrics.png")
                self.assertTrue(os.path.exists(periodic_model))
                self.assertTrue(os.path.exists(metrics_png))
            finally:
                object.__setattr__(cfg.TRAINING, "num_epochs", original_num_epochs)
                object.__setattr__(cfg.TRAINING, "save_every_n_epochs", original_save_every)
                object.__setattr__(cfg.TRAINING, "device", original_device)
                object.__setattr__(cfg.PATHS, "model_save_dir", original_model_dir)

    def test_train_entry_overfit_debug_smoke(self):
        train_loader = self._build_tiny_loader()
        val_loader = self._build_tiny_loader()
        test_loader = self._build_tiny_loader()

        original_num_epochs = cfg.TRAINING.num_epochs
        original_save_every = cfg.TRAINING.save_every_n_epochs
        original_device = cfg.TRAINING.device
        original_model_dir = cfg.PATHS.model_save_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            object.__setattr__(cfg.TRAINING, "num_epochs", 1)
            object.__setattr__(cfg.TRAINING, "save_every_n_epochs", 1)
            object.__setattr__(cfg.TRAINING, "device", "cpu")
            object.__setattr__(cfg.PATHS, "model_save_dir", tmpdir)

            try:
                with patch.object(
                    train_lstm,
                    "get_dataloaders",
                    return_value=(train_loader, val_loader, test_loader),
                ):
                    train_lstm.train(overfit_debug=True)

                periodic_model = os.path.join(tmpdir, "lstm_epoch_1.pth")
                metrics_png = os.path.join(tmpdir, "training_metrics.png")
                self.assertTrue(os.path.exists(periodic_model))
                self.assertTrue(os.path.exists(metrics_png))
            finally:
                object.__setattr__(cfg.TRAINING, "num_epochs", original_num_epochs)
                object.__setattr__(cfg.TRAINING, "save_every_n_epochs", original_save_every)
                object.__setattr__(cfg.TRAINING, "device", original_device)
                object.__setattr__(cfg.PATHS, "model_save_dir", original_model_dir)


if __name__ == "__main__":
    unittest.main()
