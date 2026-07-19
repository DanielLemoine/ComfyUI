from __future__ import annotations

import json
from subprocess import CompletedProcess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.inventory import collect_preflight


class CollectPreflightTests(unittest.TestCase):
    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_writes_required_evidence(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        object_info_fixture = json.loads(
            (PROJECT_DIR / "tests" / "fixtures" / "object_info.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=object_info_fixture,
            )

            self.assertEqual(result.object_info_path.name, "object_info.json")
            self.assertTrue(
                {
                    path.name for path in result.artifact_dir.iterdir()
                }
                >= {
                    "environment.json",
                    "model_inventory.json",
                    "custom_nodes.json",
                    "object_info.json",
                    "official_template_inventory.md",
                    "preflight_report.md",
                }
            )


if __name__ == "__main__":
    unittest.main()
