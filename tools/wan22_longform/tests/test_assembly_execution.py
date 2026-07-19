from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import wan22_longform.assembly as assembly  # noqa: E402
from wan22_longform.ffmpeg import probe_media  # noqa: E402


class AssemblyExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_executor_derives_16_to_24_timeline_from_actual_normalized_inputs(self) -> None:
        plan, targets = self._plan(reference_fps=Fraction(24, 1), source_fps=Fraction(16, 1))
        executor = getattr(assembly, "execute_assembly_plan", None)

        self.assertIsNotNone(executor, "a production assembly executor is required")
        result = executor(plan, decision_log=self.root / "boundary-decisions.json")

        self.assertEqual(result.expected_frame_count, 29)
        self.assertEqual(result.expected_duration, Fraction(29, 24))
        self.assertEqual(len(result.validated_outputs), 3)
        self.assertTrue(
            all(
                spec.frame_count == 29 and spec.fps == Fraction(24, 1)
                for spec in result.validated_outputs
            )
        )
        decision_log = json.loads((self.root / "boundary-decisions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(decision_log["decisions"]), 3)
        self.assertTrue(all(path.is_file() for path in (targets.review_mp4, targets.edit_master_ffv1, targets.edit_master_prores)))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_executor_derives_24_to_16_timeline_from_actual_normalized_inputs(self) -> None:
        plan, _ = self._plan(reference_fps=Fraction(16, 1), source_fps=Fraction(24, 1))
        executor = getattr(assembly, "execute_assembly_plan", None)

        self.assertIsNotNone(executor, "a production assembly executor is required")
        result = executor(plan, decision_log=self.root / "boundary-decisions.json")

        self.assertEqual(result.expected_frame_count, 17)
        self.assertEqual(result.expected_duration, Fraction(17, 16))
        self.assertTrue(
            all(
                spec.frame_count == 17 and spec.fps == Fraction(16, 1)
                for spec in result.validated_outputs
            )
        )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_failed_duration_validation_never_releases_rife(self) -> None:
        source = self._write_video("source.mkv", Fraction(16, 1))
        targets = assembly.AssemblyTargets(
            review_mp4=self.root / "review.mp4",
            edit_master_ffv1=self.root / "master.mkv",
            edit_master_prores=self.root / "master.mov",
            rife_review_mp4=self.root / "rife.mp4",
        )
        plan = assembly.plan_assembly(
            [source],
            targets,
            request_rife=True,
            qc_approved=True,
        )
        bad_operations = tuple(
            replace(
                operation,
                command=(*operation.command[:-1], "-frames:v", "1", operation.command[-1]),
            )
            if operation.kind == "review_mp4"
            else operation
            for operation in plan.operations
        )

        with self.assertRaisesRegex(assembly.AssemblyError, "more than one output frame"):
            assembly.execute_assembly_plan(
                replace(plan, operations=bad_operations),
                decision_log=self.root / "boundary-decisions.json",
            )

        self.assertTrue((self.root / "boundary-decisions.json").is_file())
        self.assertFalse(targets.rife_review_mp4.exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_executor_rejects_a_decision_log_that_collides_with_an_output(self) -> None:
        source = self._write_video("source.mkv", Fraction(16, 1))
        targets = assembly.AssemblyTargets(
            review_mp4=self.root / "review.mp4",
            edit_master_ffv1=self.root / "master.mkv",
            edit_master_prores=self.root / "master.mov",
        )
        plan = assembly.plan_assembly([source], targets)

        with self.assertRaisesRegex(assembly.AssemblyError, "decision log.*collide"):
            assembly.execute_assembly_plan(plan, decision_log=targets.review_mp4)

        self.assertFalse(targets.review_mp4.exists())

    def _plan(
        self,
        *,
        reference_fps: Fraction,
        source_fps: Fraction,
    ) -> tuple[assembly.AssemblyPlan, assembly.AssemblyTargets]:
        inputs = [
            self._write_video("reference.mkv", reference_fps),
            self._write_video("source-1.mkv", source_fps),
            self._write_video("source-2.mkv", source_fps),
            self._write_video("source-3.mkv", source_fps),
        ]
        targets = assembly.AssemblyTargets(
            review_mp4=self.root / "review.mp4",
            edit_master_ffv1=self.root / "master.mkv",
            edit_master_prores=self.root / "master.mov",
        )
        return (
            assembly.plan_assembly(
                inputs,
                targets,
                boundary_decisions=[
                    assembly.compare_frame_hashes("left", "right", perceptual_distance=9)
                    for _ in range(3)
                ],
            ),
            targets,
        )

    def _write_video(self, name: str, fps: Fraction):
        path = self.root / name
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
        return probe_media(path)


if __name__ == "__main__":
    unittest.main()
