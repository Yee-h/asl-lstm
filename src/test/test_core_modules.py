import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.core.hdf5_schema import (
    REQUIRED_SAMPLE_FIELDS,
    missing_required_fields,
    validate_required_fields,
)
from src.core.labels import load_id_to_label_map, load_label_to_id_map
from src.core.labels import load_id_to_label_map_compat
from src.core.paths import dataset_subset_dir, hdf5_file_path, label_map_file_path


class TestCoreModules(unittest.TestCase):
    def _write_temp_json(self, payload: dict) -> str:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".json", delete=False
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            return tmp.name

    def test_paths_module_builders(self):
        root = Path("dataset") / "processed"
        subset_dir = dataset_subset_dir("100", data_root=root)
        self.assertEqual(subset_dir, root / "WLASL100")

        hdf5_path = hdf5_file_path("100", "Train", data_root=root)
        self.assertEqual(hdf5_path, root / "WLASL100" / "WLASL100_135-Train.hdf5")

        label_path = label_map_file_path("100", data_root=root)
        self.assertEqual(label_path, root / "WLASL100" / "wlasl_100_maplabels.json")

    def test_labels_module_contract_loader(self):
        json_path = self._write_temp_json(
            {
                "id_to_label": {"0": "book", "1": "drink"},
                "label_to_id": {"book": 0, "drink": 1},
            }
        )
        try:
            label_to_id = load_label_to_id_map(json_path)
            id_to_label = load_id_to_label_map(json_path)
            self.assertEqual(label_to_id, {"book": 0, "drink": 1})
            self.assertEqual(id_to_label, {0: "book", 1: "drink"})
        finally:
            os.remove(json_path)

    def test_labels_module_rejects_legacy_mapping(self):
        json_path = self._write_temp_json(
            {
                "id_to_label": {"book": 0},
                "label_to_id": {"0": "book"},
            }
        )
        try:
            with self.assertRaisesRegex(ValueError, "旧格式"):
                load_label_to_id_map(json_path)
        finally:
            os.remove(json_path)

    def test_labels_compat_loader_migrates_legacy_file(self):
        json_path = self._write_temp_json(
            {
                "id_to_label": {"book": 0, "drink": 1},
                "label_to_id": {"0": "book", "1": "drink"},
            }
        )
        try:
            id_to_label = load_id_to_label_map_compat(json_path)
            self.assertEqual(id_to_label, {0: "book", 1: "drink"})

            converted = load_label_to_id_map(json_path)
            self.assertEqual(converted, {"book": 0, "drink": 1})

            backup_file = Path(json_path + ".legacy.bak")
            self.assertTrue(backup_file.exists())
            backup_file.unlink()
        finally:
            os.remove(json_path)

    def test_hdf5_schema_validator(self):
        complete = list(REQUIRED_SAMPLE_FIELDS)
        self.assertEqual(missing_required_fields(complete), [])
        validate_required_fields(complete)

        incomplete = ["data", "label"]
        missing = missing_required_fields(incomplete)
        self.assertIn("length", missing)
        with self.assertRaisesRegex(ValueError, "必填字段"):
            validate_required_fields(incomplete)


if __name__ == "__main__":
    unittest.main()
