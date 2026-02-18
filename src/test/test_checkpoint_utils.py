import os
import sys
import unittest

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.model.checkpoint_utils import average_state_dicts


class TestCheckpointUtils(unittest.TestCase):
    def test_average_state_dicts(self):
        a = {
            "w": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "b": torch.tensor([2.0], dtype=torch.float32),
        }
        b = {
            "w": torch.tensor([3.0, 5.0], dtype=torch.float32),
            "b": torch.tensor([4.0], dtype=torch.float32),
        }

        avg = average_state_dicts([a, b])
        self.assertTrue(torch.allclose(avg["w"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(avg["b"], torch.tensor([3.0])))


if __name__ == "__main__":
    unittest.main()
