import os
import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)


class TestNoLegacyCfgAccess(unittest.TestCase):
    def test_no_legacy_uppercase_cfg_access_outside_config(self):
        pattern = re.compile(r"cfg\.[A-Z][A-Z0-9]*_[A-Z0-9_]*")
        src_root = Path(PROJECT_ROOT) / "src"

        excluded = {
            str((src_root / "config.py").resolve()),
            str((src_root / "test" / "test_no_legacy_cfg_access.py").resolve()),
        }

        offenders = []
        for file_path in src_root.rglob("*.py"):
            resolved = str(file_path.resolve())
            if resolved in excluded:
                continue

            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(file_path.relative_to(src_root)))

        self.assertEqual(offenders, [], msg=f"Legacy cfg access found: {offenders}")


if __name__ == "__main__":
    unittest.main()
