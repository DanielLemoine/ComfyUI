from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.cli import DeploymentError, deploy_workflows, main  # noqa: E402


class CliTests(unittest.TestCase):
    def test_validate_project_does_not_mutate_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_example_manifest(root)
            before = source.read_bytes()

            self.assertEqual(main(["validate-project", str(source)]), 0)

            self.assertEqual(source.read_bytes(), before)

    def test_deploy_rejects_existing_target_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "workflow.json").write_text("{}", encoding="utf-8")
            target = root / "wan22_longform"
            target.mkdir()

            with self.assertRaisesRegex(DeploymentError, "already exists"):
                deploy_workflows(source_dir, target, force=False)

    def test_deploy_copies_only_known_workflows_when_force_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "workflow.json").write_text("new", encoding="utf-8")
            target = root / "wan22_longform"
            target.mkdir()
            (target / "operator-note.txt").write_text("keep", encoding="utf-8")

            deployed = deploy_workflows(source_dir, target, force=True)

            self.assertEqual(deployed, (target / "workflow.json",))
            self.assertEqual((target / "workflow.json").read_text(encoding="utf-8"), "new")
            self.assertEqual((target / "operator-note.txt").read_text(encoding="utf-8"), "keep")

    def test_deploy_rejects_a_target_inside_the_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "workflow.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(DeploymentError, "inside the source"):
                deploy_workflows(source_dir, source_dir / "nested", force=True)

    def test_deploy_rejects_a_symbolic_link_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "workflow.json").write_text("{}", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            target = root / "wan22_longform"
            try:
                os.symlink(outside, target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(DeploymentError, "symbolic link"):
                deploy_workflows(source_dir, target, force=True)

    def test_assemble_shot_rejects_a_path_like_manifest_shot_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_example_manifest(root)
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            source["shots"][0]["id"] = "../outside"
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "single path component"):
                main(["assemble-shot", str(manifest), "../outside"])

    def test_all_operator_commands_parse(self) -> None:
        commands = (
            ["preflight", "--comfy-root", "C:/ComfyUI"],
            ["validate-project", "project.yaml"],
            ["render-segment", "project.yaml", "S010", "S010_C001"],
            ["render-bridge", "project.yaml", "B010"],
            ["render-shot", "project.yaml", "S010"],
            ["qc-contact-sheet", "attempt"],
            ["accept", "attempt", "--note", "approved"],
            ["reject", "attempt", "--note", "reject"],
            ["retry", "attempt", "--note", "retry"],
            ["assemble-shot", "project.yaml", "S010"],
            ["assemble-project", "project.yaml"],
            ["status", "project.yaml"],
            ["resume", "project.yaml"],
            ["deploy-workflows", "--target", "C:/ComfyUI/user/default/workflows/wan22_longform"],
        )

        for command in commands:
            with self.subTest(command=command[0]):
                with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
                    main([*command, "--help"])
                self.assertEqual(raised.exception.code, 0)

    @staticmethod
    def _write_example_manifest(root: Path) -> Path:
        opening = root / "opening.png"
        opening.write_bytes(b"opening")
        manifest = root / "project.yaml"
        source = {
            "schema_version": 1,
            "project_id": "cli_fixture",
            "title": "CLI fixture",
            "mode": "cinematic",
            "target_seconds": 2,
            "output_root": str(root / "outputs"),
            "outputs": {
                "review_mp4": str(root / "outputs" / "review.mp4"),
                "edit_master_ffv1": str(root / "outputs" / "master.mkv"),
                "edit_master_prores": str(root / "outputs" / "master.mov"),
            },
            "preset": "P0_IDENTITY_BASELINE",
            "models": {
                "high": "high.safetensors",
                "low": "low.safetensors",
                "vae": "vae.safetensors",
                "text_encoder": "text-encoder.safetensors",
            },
            "model_files": [
                "high.safetensors",
                "low.safetensors",
                "vae.safetensors",
                "text-encoder.safetensors",
            ],
            "workflow_api": str(PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"),
            "bridge_workflow_api": str(PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"),
            "workflow_hashes": {
                "segment_api": hashlib.sha256(
                    (PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json").read_bytes()
                ).hexdigest(),
                "bridge_api": hashlib.sha256(
                    (PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json").read_bytes()
                ).hexdigest(),
            },
            "environment_snapshot": {
                "captured_at": "2026-07-19T00:00:00Z",
                "platform": "fixture",
                "python": "3.11.6",
                "gpu": "fixture",
            },
            "render": {
                "workflow": "wan22_segment_i2v_native_api.json",
                "width": 640,
                "height": 640,
                "frames": 17,
                "generation_fps": 16,
                "review_mp4_codec": "h264",
                "master_codec": "ffv1",
                "seed_base": 1,
                "seed_increment": 17,
            },
            "request": {
                "positive": "neutral fully clothed adult",
                "negative": "low quality",
                "width": 640,
                "height": 640,
                "frames": 17,
            },
            "inputs": {"opening_frame": str(opening)},
            "attempts_dir": str(root / "attempts"),
            "policy": {"automatic_continuation": False},
            "continuation": {"strategy": "selected_tail", "reset_limit": 2},
            "qc": {"candidate_count": 2, "retry_limit": 1},
            "loras": {
                "identity": {"mode": "none"},
                "vbvr": {"enabled": False},
                "motion": {"enabled": False},
                "corrective": {"enabled": False},
                "permissiveness": {
                    "mystic": {"enabled": False},
                    "wan_general": {"enabled": False},
                },
            },
            "shots": [
                {
                    "id": "S010",
                    "target_seconds": 2,
                    "anchor_image": str(opening),
                    "segments": [
                        {
                            "id": "S010_C001",
                            "action": "A neutral adult pauses naturally.",
                            "expected_seconds": 1,
                            "seed_offset": 0,
                        }
                    ],
                }
            ],
            "bridges": [
                {
                    "id": "B010",
                    "shot_id": "S010",
                    "strategy": "flf2v",
                    "purpose": "technical_smoke",
                    "base_source_image": str(opening),
                    "frames": 33,
                }
            ],
            "assembly_order": [{"shot_id": "S010", "segment_id": "S010_C001"}],
        }
        manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        return manifest
