import os
import sys
import unittest

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.model.tta import build_hflip_tta_batch


class TestTTAUtils(unittest.TestCase):
    def test_hflip_tta_preserves_shape(self):
        inputs = torch.randn(2, 7, cfg.SEQUENCE.input_size)
        flipped = build_hflip_tta_batch(inputs)
        self.assertEqual(tuple(flipped.shape), tuple(inputs.shape))

    def test_hflip_tta_swaps_hand_landmarks(self):
        c = cfg.SEQUENCE.landmark_dim
        v = cfg.SEQUENCE.num_landmarks
        sample = torch.zeros((1, 1, c * v), dtype=torch.float32)
        view = sample.view(1, 1, c, v)

        view[0, 0, 0, 25] = 1.0
        view[0, 0, 0, 46] = 2.0

        flipped = build_hflip_tta_batch(sample)
        out = flipped.view(1, 1, c, v)

        self.assertAlmostEqual(float(out[0, 0, 0, 25]), -2.0, places=6)
        self.assertAlmostEqual(float(out[0, 0, 0, 46]), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
