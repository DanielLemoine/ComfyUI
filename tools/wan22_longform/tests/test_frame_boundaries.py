from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.frames import (  # noqa: E402
    FrameError,
    candidate_start_index,
    create_contact_sheet,
    extract_candidate_frames,
)


class FrameBoundaryTests(unittest.TestCase):
    def test_tail_selection_uses_length_minus_count(self) -> None:
        self.assertEqual(
            candidate_start_index(frame_count=81, count=5, where="tail"), 76
        )

    def test_head_selection_starts_at_zero(self) -> None:
        self.assertEqual(
            candidate_start_index(frame_count=81, count=5, where="head"), 0
        )

    def test_short_or_invalid_candidate_requests_fail_clearly(self) -> None:
        cases = (
            ({"frame_count": 4, "count": 5, "where": "tail"}, "shorter"),
            ({"frame_count": 5, "count": 0, "where": "head"}, "positive"),
            ({"frame_count": 5, "count": 5, "where": "middle"}, "head or tail"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(FrameError, message):
                    candidate_start_index(**arguments)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_ffmpeg_extracts_exact_head_and_tail_frames_and_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "fixture.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=8",
                    "-frames:v",
                    "8",
                    "-c:v",
                    "ffv1",
                    str(video),
                ],
                check=True,
            )

            head = extract_candidate_frames(video, 3, "head", root / "head")
            tail = extract_candidate_frames(video, 3, "tail", root / "tail")
            contact_sheet = create_contact_sheet([*head, *tail], root / "contact-sheet.png")

            self.assertEqual([path.name for path in head], ["frame-000000.png", "frame-000001.png", "frame-000002.png"])
            self.assertEqual([path.name for path in tail], ["frame-000005.png", "frame-000006.png", "frame-000007.png"])
            self.assertTrue(all(path.stat().st_size > 0 for path in [*head, *tail]))
            self.assertGreater(contact_sheet.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
