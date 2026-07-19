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
    AssemblyTargets,
    compare_frame_hashes,
    duration_seconds,
    plan_assembly,
    validate_assembly_outputs,
    validate_output_duration,
)
from wan22_longform.ffmpeg import MediaSpec, probe_media, run_ffmpeg  # noqa: E402


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

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_plan_uses_normalized_frame_count_and_runs_final_duration_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self._media(root / "left.mkv", frame_count=5, fps=Fraction(16, 1))
            right = self._media(root / "right.mkv", frame_count=5, fps=Fraction(24, 1))
            targets = AssemblyTargets(
                review_mp4=root / "review.mp4",
                edit_master_ffv1=root / "master.mkv",
                edit_master_prores=root / "master.mov",
            )
            plan = plan_assembly(
                [left, right],
                targets,
                boundary_decisions=[compare_frame_hashes("left", "right", perceptual_distance=9)],
            )

            self.assertEqual(plan.expected_frame_count, 9)
            self.assertEqual(plan.expected_duration, Fraction(9, 16))
            self.assertNotEqual(
                plan.expected_duration,
                duration_seconds(left.frame_count, left.fps)
                + duration_seconds(right.frame_count, right.fps),
            )
            checks = [operation for operation in plan.operations if operation.kind == "validate_duration"]
            normalization = [operation for operation in plan.operations if operation.kind == "normalize"]
            final_outputs = [
                operation
                for operation in plan.operations
                if operation.kind in {"review_mp4", "edit_master_ffv1", "edit_master_prores"}
            ]
            self.assertEqual(len(checks), 3)
            self.assertTrue(
                all(
                    any(
                        "fps=16:round=down:eof_action=pass" in argument
                        for argument in operation.command
                    )
                    for operation in normalization
                )
            )
            self.assertTrue(
                all(operation.expected_duration == plan.expected_duration for operation in checks)
            )
            self.assertTrue(
                all(plan.operations.index(check) > plan.operations.index(final_outputs[-1]) for check in checks)
            )

            self._write_video(targets.review_mp4, frame_count=9, fps=Fraction(16, 1))
            self._write_video(targets.edit_master_ffv1, frame_count=9, fps=Fraction(16, 1))
            self._write_video(targets.edit_master_prores, frame_count=9, fps=Fraction(16, 1))
            checked = validate_assembly_outputs(plan)

        self.assertEqual(len(checked), 3)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_plan_owned_duration_validation_rejects_two_frame_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self._media(root / "left.mkv", frame_count=5, fps=Fraction(16, 1))
            right = self._media(root / "right.mkv", frame_count=5, fps=Fraction(24, 1))
            targets = AssemblyTargets(
                review_mp4=root / "review.mp4",
                edit_master_ffv1=root / "master.mkv",
                edit_master_prores=root / "master.mov",
            )
            plan = plan_assembly(
                [left, right],
                targets,
                boundary_decisions=[compare_frame_hashes("left", "right", perceptual_distance=9)],
            )
            self._write_video(targets.review_mp4, frame_count=7, fps=Fraction(16, 1))

            with self.assertRaisesRegex(AssemblyError, "more than one output frame"):
                validate_assembly_outputs(plan)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_normalization_operations_match_explicit_target_frame_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.mkv"
            right_path = root / "right.mkv"
            self._write_video(left_path, frame_count=5, fps=Fraction(16, 1), color="green")
            self._write_video(right_path, frame_count=5, fps=Fraction(24, 1), color="red")
            targets = AssemblyTargets(
                review_mp4=root / "review.mp4",
                edit_master_ffv1=root / "master.mkv",
                edit_master_prores=root / "master.mov",
            )
            plan = plan_assembly(
                [probe_media(left_path), probe_media(right_path)],
                targets,
                boundary_decisions=[compare_frame_hashes("left", "right", perceptual_distance=9)],
            )

            normalizations = [operation for operation in plan.operations if operation.kind == "normalize"]
            for operation in normalizations:
                run_ffmpeg(operation.command)
            normalized_frame_count = sum(probe_media(operation.output).frame_count for operation in normalizations)

        self.assertEqual(normalized_frame_count, plan.expected_frame_count)

    @staticmethod
    def _write_video(
        path: Path,
        *,
        frame_count: int,
        fps: Fraction,
        color: str = "green",
    ) -> None:
        codec = "ffv1" if path.suffix == ".mkv" else "libx264"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size=16x8:rate={fps}:duration=10",
                "-frames:v",
                str(frame_count),
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _media(path: Path, *, frame_count: int, fps: Fraction) -> MediaSpec:
        path.write_bytes(b"fixture")
        return MediaSpec(
            path=path,
            width=1280,
            height=720,
            fps=fps,
            time_base=Fraction(1, fps),
            pixel_format="yuv420p",
            codec="h264",
            profile="High",
            color_space="bt709",
            color_transfer="bt709",
            color_primaries="bt709",
            audio=None,
            frame_count=frame_count,
        )


if __name__ == "__main__":
    unittest.main()
