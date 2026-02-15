import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.data_process.read_data_struct import _build_file_path


class TestReadDataStructPaths(unittest.TestCase):
    def test_build_file_path_points_to_processed_dataset(self):
        path = _build_file_path("100", "Train")
        expected_tail = (
            Path("dataset") / "processed" / "WLASL100" / "WLASL100_135-Train.hdf5"
        )
        self.assertTrue(str(path).endswith(str(expected_tail)), msg=f"Got path: {path}")


if __name__ == "__main__":
    unittest.main()
