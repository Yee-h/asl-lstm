import json
import os
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.model.dataloader import load_label_to_id_map


class TestLabelMapContract(unittest.TestCase):
    def _write_temp_json(self, payload: dict) -> str:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".json", delete=False
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            return tmp.name

    def test_valid_contract_is_loaded(self):
        json_path = self._write_temp_json(
            {
                "id_to_label": {"0": "book", "1": "drink"},
                "label_to_id": {"book": 0, "drink": 1},
            }
        )

        try:
            label_to_id = load_label_to_id_map(json_path)
            self.assertEqual(label_to_id, {"book": 0, "drink": 1})
        finally:
            os.remove(json_path)

    def test_invalid_contract_raises_clear_error(self):
        json_path = self._write_temp_json(
            {
                "id_to_label": {"0": "book"},
                "label_to_id": {"book": "0"},
            }
        )

        try:
            with self.assertRaisesRegex(ValueError, "label_to_id"):
                load_label_to_id_map(json_path)
        finally:
            os.remove(json_path)

    def test_legacy_format_is_rejected(self):
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


if __name__ == "__main__":
    unittest.main()
