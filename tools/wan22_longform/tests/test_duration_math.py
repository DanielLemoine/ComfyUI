from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.assembly import (  # noqa: E402
    AssemblyError,
    duration_seconds,
    validate_output_duration,
)


class DurationMathTests(unittest.TestCase):
    def test_duration_is_exact_fraction_of_frame_count_and_fps(self) -> None:
        self.assertEqual(duration_seconds(81, Fraction(16, 1)), Fraction(81, 16))

    def test_duration_rejects_non_positive_inputs(self) -> None:
        with self.assertRaisesRegex(AssemblyError, "frame_count"):
            duration_seconds(0, Fraction(16, 1))
        with self.assertRaisesRegex(AssemblyError, "fps"):
            duration_seconds(81, Fraction(0, 1))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_probed_normalized_output_duration_accepts_one_frame_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "normalized.mkv"
            self._write_video(output, frame_count=54, fps=Fraction(16, 1))

            spec = validate_output_duration(
                output,
                duration_seconds(81, Fraction(24, 1)) + Fraction(1, 16),
                output_fps=Fraction(16, 1),
            )

        self.assertEqual(spec.frame_count, 54)
        self.assertEqual(spec.fps, Fraction(16, 1))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_probed_output_duration_rejects_more_than_one_frame_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "normalized.mkv"
            self._write_video(output, frame_count=54, fps=Fraction(16, 1))

            with self.assertRaisesRegex(AssemblyError, "more than one output frame"):
                validate_output_duration(
                    output,
                    duration_seconds(81, Fraction(24, 1)) + Fraction(2, 16),
                    output_fps=Fraction(16, 1),
                )

    @staticmethod
    def _write_video(path: Path, *, frame_count: int, fps: Fraction) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=green:size=16x8:rate={fps}:duration=10",
                "-frames:v",
                str(frame_count),
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
