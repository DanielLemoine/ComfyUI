from __future__ import annotations

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
            "preset": "P0_IDENTITY_BASELINE",
            "models": {"high": "high.safetensors", "low": "low.safetensors"},
            "model_files": ["high.safetensors", "low.safetensors"],
            "workflow_api": str(PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"),
            "bridge_workflow_api": str(PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"),
            "request": {
                "positive": "neutral fully clothed adult",
                "negative": "low quality",
                "seed": 1,
                "width": 640,
                "height": 640,
                "frames": 17,
            },
            "inputs": {"opening_frame": str(opening)},
            "attempts_dir": str(root / "attempts"),
            "shots": [{"id": "S010", "segments": [{"id": "S010_C001"}]}],
            "bridges": [
                {
                    "id": "B010",
                    "shot_id": "S010",
                    "base_source_image": str(opening),
                    "frames": 33,
                }
            ],
        }
        manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        return manifest
