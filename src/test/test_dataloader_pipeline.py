import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataloader import preprocess_keypoints, swap_left_right_keypoints


class TestDataloaderPipeline(unittest.TestCase):
    def test_preprocess_keypoints_xy_to_xydxdy_and_padding(self):
        data = np.array(
            [
                [[1.0, 2.0], [0.5, 1.0]],
                [[2.0, 4.0], [1.0, 2.0]],
                [[4.0, 7.0], [2.0, 3.0]],
            ],
            dtype=np.float32,
        )

        out, valid_len = preprocess_keypoints(data, max_frames=5)

        self.assertEqual(valid_len, 3)
        self.assertEqual(tuple(out.shape), (5, 8))

        frame0 = out[0].numpy().reshape(4, 2)
        frame1 = out[1].numpy().reshape(4, 2)

        np.testing.assert_allclose(frame0[2], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(frame0[3], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(frame1[2], [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(frame1[3], [0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(
            out[3].numpy(), np.zeros(8, dtype=np.float32), atol=1e-6
        )

    def test_preprocess_keypoints_keep_existing_4_channels(self):
        data = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0], [8.0]],
            ],
            dtype=np.float32,
        )

        out, valid_len = preprocess_keypoints(data, max_frames=3)

        self.assertEqual(valid_len, 2)
        self.assertEqual(tuple(out.shape), (3, 4))
        np.testing.assert_allclose(out[0].numpy(), [1.0, 2.0, 3.0, 4.0], atol=1e-6)
        np.testing.assert_allclose(out[1].numpy(), [5.0, 6.0, 7.0, 8.0], atol=1e-6)
        np.testing.assert_allclose(
            out[2].numpy(), np.zeros(4, dtype=np.float32), atol=1e-6
        )

    def test_swap_left_right_keypoints(self):
        data = np.zeros((1, 4, 135), dtype=np.float32)

        data[0, 0, 25] = 1.0
        data[0, 0, 46] = 2.0

        data[0, 0, 5] = 10.0
        data[0, 0, 2] = 20.0

        swapped = swap_left_right_keypoints(data)

        self.assertEqual(swapped[0, 0, 25], 2.0)
        self.assertEqual(swapped[0, 0, 46], 1.0)
        self.assertEqual(swapped[0, 0, 5], 20.0)
        self.assertEqual(swapped[0, 0, 2], 10.0)


if __name__ == "__main__":
    unittest.main()
