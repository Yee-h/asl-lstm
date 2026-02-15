import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataloader import compute_sample_weights


class TestSamplingStrategy(unittest.TestCase):
    def test_minority_class_gets_higher_weight(self):
        labels = [0, 0, 0, 1]
        weights = compute_sample_weights(labels, power=1.0)

        self.assertEqual(len(weights), 4)
        self.assertGreater(weights[-1], weights[0])

    def test_returns_empty_for_empty_labels(self):
        weights = compute_sample_weights([], power=1.0)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(weights.size, 0)


if __name__ == "__main__":
    unittest.main()
