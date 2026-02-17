import os
import sys
import tempfile
import unittest

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.core.hdf5_schema import validate_required_fields
from src.data_process.preprocess_wlasl import create_hdf5_file


class TestPreprocessSmoke(unittest.TestCase):
    def test_create_hdf5_file_outputs_valid_structure(self):
        channels = len(cfg.SEQUENCE.base_feature_channels)
        sample_data = np.zeros(
            (cfg.SEQUENCE.max_frames, channels, cfg.SEQUENCE.num_landmarks),
            dtype=np.float32,
        )
        sample_mask = np.ones((cfg.SEQUENCE.max_frames, cfg.SEQUENCE.num_landmarks), dtype=np.uint8)

        payload = {
            "video_demo": {
                "data": sample_data,
                "mask": sample_mask,
                "quality": 1.0,
                "length": cfg.SEQUENCE.max_frames,
                "label": "book",
                "video_name": "video_demo.mp4",
                "width": 640,
                "height": 480,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "WLASL100_135-Train.hdf5")
            create_hdf5_file(
                out_path,
                payload,
                split="train",
                subset=cfg.PATHS.dataset_scale,
            )

            with h5py.File(out_path, "r") as h5_obj:
                self.assertIn("0", h5_obj.keys())
                group = h5_obj["0"]
                validate_required_fields(group.keys())
                self.assertEqual(group["data"].shape[2], cfg.SEQUENCE.num_landmarks)
                self.assertEqual(group["length"][()], cfg.SEQUENCE.max_frames)


if __name__ == "__main__":
    unittest.main()
