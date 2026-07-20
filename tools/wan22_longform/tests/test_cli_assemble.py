from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform import cli  # noqa: E402


class AssembleCliTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_assemble_executes_native_plan_and_reports_validated_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source clip's.mkv"
            self._write_video(source)
            review = root / "review.mp4"
            master_ffv1 = root / "master.mkv"
            master_prores = root / "master.mov"
            decision_log = root / "boundary-decisions.json"
            stdout = io.StringIO()

            with patch.object(
                sys,
                "argv",
                [
                    "wan22-longform",
                    "assemble",
                    "--input",
                    str(source),
                    "--review-mp4",
                    str(review),
                    "--edit-master-ffv1",
                    str(master_ffv1),
                    "--edit-master-prores",
                    str(master_prores),
                    "--decision-log",
                    str(decision_log),
                    "--diagnostic-only",
                ],
            ), contextlib.redirect_stdout(stdout):
                cli.main()

            result = json.loads(stdout.getvalue())
            self.assertEqual(result["expected_frame_count"], 5)
            self.assertEqual(result["expected_duration"], "5/16")
            self.assertTrue(all(path.is_file() for path in (review, master_ffv1, master_prores)))
            self.assertEqual(json.loads(decision_log.read_text(encoding="utf-8"))["decisions"], [])
            self.assertEqual(
                review.with_suffix(".concat.txt").read_text(encoding="utf-8"),
                "file '" + source.resolve().as_posix().replace("'", r"'\''") + "'\n",
            )

    @staticmethod
    def _write_video(path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:size=16x8:rate=16:duration=10",
                "-frames:v",
                "5",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
