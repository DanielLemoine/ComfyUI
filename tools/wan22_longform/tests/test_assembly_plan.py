from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import json
from fractions import Fraction
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.assembly import (  # noqa: E402
    AssemblyError,
    AssemblyTargets,
    compare_boundary,
    compare_frame_hashes,
    plan_assembly,
    write_boundary_decision,
)
from wan22_longform.ffmpeg import MediaSpec, probe_media  # noqa: E402


class AssemblyPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_exact_duplicate_is_trimmed_once(self) -> None:
        decision = compare_frame_hashes("same", "same", perceptual_distance=0)

        self.assertEqual(decision.trim_right_frames, 1)
        self.assertFalse(decision.requires_review)
        self.assertEqual(decision.reason, "exact duplicate boundary frame")

    def test_similar_frame_is_not_silently_trimmed(self) -> None:
        decision = compare_frame_hashes("different", "different", perceptual_distance=3)

        self.assertEqual(decision.trim_right_frames, 0)
        self.assertTrue(decision.requires_review)
        self.assertIn("perceptually similar", decision.reason)

    def test_boundary_decision_log_is_append_only(self) -> None:
        decision = compare_frame_hashes("same", "same", perceptual_distance=0)
        destination = self.root / "reviews" / "boundary.json"

        self.assertEqual(write_boundary_decision(decision, destination), destination)
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["trim_right_frames"], 1)
        with self.assertRaises(FileExistsError):
            write_boundary_decision(decision, destination)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_probe_reads_exact_video_and_audio_policy_fields(self) -> None:
        fixture = self.root / "fixture.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=16x8:rate=16:duration=1",
                "-frames:v",
                "16",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-video_track_timescale",
                "16",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-an",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        spec = probe_media(fixture)

        self.assertEqual(spec.width, 16)
        self.assertEqual(spec.height, 8)
        self.assertEqual(spec.fps, Fraction(16, 1))
        self.assertEqual(spec.time_base, Fraction(1, 16))
        self.assertEqual(spec.pixel_format, "yuv420p")
        self.assertEqual(spec.codec, "h264")
        self.assertEqual(spec.profile, "High")
        self.assertEqual(spec.color_space, "bt709")
        self.assertEqual(spec.color_transfer, "bt709")
        self.assertEqual(spec.color_primaries, "bt709")
        self.assertIsNone(spec.audio)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_exact_boundary_fixture_trims_right_once(self) -> None:
        left = self.root / "left.mkv"
        right = self.root / "right.mkv"
        for destination in (left, right):
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:size=16x8:rate=16:duration=1",
                    "-frames:v",
                    "16",
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        decision = compare_boundary(left, right)

        self.assertEqual(decision.trim_right_frames, 1)
        self.assertFalse(decision.requires_review)

    def test_incompatible_inputs_receive_normalization_before_concat(self) -> None:
        left = self._media("left.mp4", width=1280, fps=Fraction(16, 1))
        right = self._media("right.mp4", width=1024, fps=Fraction(24, 1))

        plan = plan_assembly(
            [left, right],
            self._targets(),
            boundary_decisions=[self._distinct_boundary()],
        )

        self.assertTrue(plan.normalize_first)
        self.assertEqual(plan.operations[0].kind, "normalize")
        self.assertEqual(plan.operations[1].kind, "normalize")
        self.assertEqual(plan.operations[2].kind, "write_concat_manifest")
        self.assertEqual(plan.operations[3].kind, "concat")
        self.assertTrue(
            all(operation.output.suffix == ".mkv" for operation in plan.operations[:2])
        )
        self.assertEqual(
            {
                operation.kind
                for operation in plan.operations
                if operation.kind in {"review_mp4", "edit_master_ffv1", "edit_master_prores"}
            },
            {"review_mp4", "edit_master_ffv1", "edit_master_prores"},
        )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_verified_exact_duplicate_boundary_trims_right_input_before_concat(self) -> None:
        left = self._ffv1_media("left.mkv", "blue")
        right = self._ffv1_media("right.mkv", "blue")
        targets = self._targets()
        decision = compare_boundary(left.path, right.path)

        plan = plan_assembly(
            [left, right],
            targets,
            boundary_decisions=[decision],
        )

        trim = next(operation for operation in plan.operations if operation.kind == "trim_boundary")
        concat = next(operation for operation in plan.operations if operation.kind == "concat")
        self.assertEqual(trim.command[trim.command.index("-vf") + 1], "trim=start_frame=1,setpts=PTS-STARTPTS")
        self.assertIn(trim.output, concat.inputs)
        self.assertNotIn(right.path, concat.inputs)
        self.assertEqual(plan.expected_duration, Fraction(31, 16))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_forged_exact_trim_decision_blocks_automatic_assembly(self) -> None:
        left = self._ffv1_media("left.mkv", "blue")
        right = self._ffv1_media("right.mkv", "red")
        targets = self._targets()
        forged = compare_frame_hashes("forged", "forged", perceptual_distance=0)

        with self.assertRaisesRegex(AssemblyError, "requires review"):
            plan_assembly([left, right], targets, boundary_decisions=[forged])

        self.assertFalse(targets.review_mp4.with_suffix(".concat.txt").exists())

    def test_review_required_boundary_blocks_automatic_assembly_without_trimming(self) -> None:
        left = self._media("left.mp4")
        right = self._media("right.mp4")
        targets = self._targets()
        similar = compare_frame_hashes("different", "different", perceptual_distance=3)

        with self.assertRaisesRegex(AssemblyError, "requires review"):
            plan_assembly([left, right], targets, boundary_decisions=[similar])

        self.assertFalse(targets.review_mp4.with_suffix(".concat.txt").exists())

    def test_plan_writes_safely_quoted_concat_manifest_before_concat(self) -> None:
        left = self._media("left clip.mp4")
        right = self._media("right clip's.mp4")
        targets = self._targets()

        plan = plan_assembly(
            [left, right],
            targets,
            boundary_decisions=[self._distinct_boundary()],
        )

        manifest = targets.review_mp4.with_suffix(".concat.txt")
        write_operation = next(
            operation for operation in plan.operations if operation.kind == "write_concat_manifest"
        )
        concat = next(operation for operation in plan.operations if operation.kind == "concat")
        self.assertEqual(write_operation.output, manifest)
        self.assertLess(plan.operations.index(write_operation), plan.operations.index(concat))
        self.assertEqual(
            manifest.read_text(encoding="utf-8"),
            "file '"
            + left.path.resolve().as_posix()
            + "'\nfile '"
            + right.path.resolve().as_posix().replace("'", r"'\''")
            + "'\n",
        )

    def test_existing_target_is_never_overwritten(self) -> None:
        left = self._media("left.mp4")
        targets = self._targets()
        targets.review_mp4.write_bytes(b"existing output")

        with self.assertRaisesRegex(AssemblyError, "already exists"):
            plan_assembly([left], targets)

    def test_existing_native_target_does_not_leave_a_concat_manifest(self) -> None:
        left = self._media("left.mp4")
        targets = self._targets()
        manifest = targets.review_mp4.with_suffix(".concat.txt")
        native = manifest.with_name(f"{manifest.stem}-native.mkv")
        native.write_bytes(b"existing native output")

        with self.assertRaisesRegex(AssemblyError, "native output already exists"):
            plan_assembly([left], targets)

        self.assertFalse(manifest.exists())

    def test_rife_is_only_scheduled_after_approved_native_assembly(self) -> None:
        left = self._media("left.mp4")
        targets = self._targets(rife_review_mp4=self.root / "rife-review.mp4")

        with self.assertRaisesRegex(AssemblyError, "approved QC"):
            plan_assembly([left], targets, request_rife=True, qc_approved=False)

        plan = plan_assembly([left], targets, request_rife=True, qc_approved=True)

        self.assertEqual(plan.operations[-1].kind, "rife")
        self.assertEqual(plan.operations[-1].inputs, (targets.review_mp4,))

    @staticmethod
    def _distinct_boundary():
        return compare_frame_hashes("left", "right", perceptual_distance=9)

    def _ffv1_media(self, name: str, color: str) -> MediaSpec:
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
                f"color=c={color}:size=16x8:rate=16:duration=1",
                "-frames:v",
                "16",
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

    def _media(
        self,
        name: str,
        *,
        width: int = 1280,
        fps: Fraction = Fraction(16, 1),
    ) -> MediaSpec:
        path = self.root / name
        path.write_bytes(b"fixture")
        return MediaSpec(
            path=path,
            width=width,
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
            frame_count=81,
        )

    def _targets(self, *, rife_review_mp4: Path | None = None) -> AssemblyTargets:
        return AssemblyTargets(
            review_mp4=self.root / "review.mp4",
            edit_master_ffv1=self.root / "master.mkv",
            edit_master_prores=self.root / "master.mov",
            rife_review_mp4=rife_review_mp4,
        )


if __name__ == "__main__":
    unittest.main()
