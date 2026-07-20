from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.cli import DeploymentError, deploy_workflows, main  # noqa: E402
from wan22_longform.inventory import PreflightResult  # noqa: E402


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

    def test_preflight_emits_machine_readable_gate_status_and_nonzero_for_blocked(self) -> None:
        result = PreflightResult(
            artifact_dir=Path("C:/evidence"),
            object_info_path=Path("C:/evidence/object_info.json"),
            active_workflow_dir=None,
            native_i2v_template=None,
            native_flf_template=None,
            status="BLOCKED",
            blockers=("CUDA proof is unavailable",),
            workflow_candidates=(),
        )
        output = io.StringIO()

        with patch("wan22_longform.cli.collect_preflight", return_value=result), redirect_stdout(output):
            exit_code = main(["preflight", "--comfy-root", "C:/ComfyUI"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "artifact_dir": str(Path("C:/evidence")),
                "blockers": ["CUDA proof is unavailable"],
                "status": "BLOCKED",
            },
        )

    def test_preflight_returns_zero_only_when_ready(self) -> None:
        result = PreflightResult(
            artifact_dir=Path("C:/evidence"),
            object_info_path=Path("C:/evidence/object_info.json"),
            active_workflow_dir=None,
            native_i2v_template=None,
            native_flf_template=None,
            status="READY",
            blockers=(),
            workflow_candidates=(),
        )

        with patch("wan22_longform.cli.collect_preflight", return_value=result), redirect_stdout(io.StringIO()):
            exit_code = main(["preflight", "--comfy-root", "C:/ComfyUI"])

        self.assertEqual(exit_code, 0)

    def test_ready_preflight_binds_object_info_before_validate_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_example_manifest(root)
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            source.pop("object_info")
            source.pop("object_info_sha256")
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
            snapshot = root / "preflight" / "object_info.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(
                (PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json").read_bytes()
            )
            result = PreflightResult(
                artifact_dir=snapshot.parent,
                object_info_path=snapshot,
                active_workflow_dir=None,
                native_i2v_template=None,
                native_flf_template=None,
                status="READY",
                blockers=(),
                workflow_candidates=(),
            )

            with patch("wan22_longform.cli.collect_preflight", return_value=result), redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(
                    main(
                        [
                            "preflight",
                            "--comfy-root",
                            str(root),
                            "--project",
                            str(manifest),
                            "--bind-project",
                        ]
                    ),
                    0,
                )

            bound = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                bound["object_info"], str(snapshot.relative_to(manifest.parent))
            )
            self.assertEqual(
                bound["object_info_sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest()
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate-project", str(manifest)]), 0)

    def test_blocked_preflight_does_not_bind_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_example_manifest(root)
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            source.pop("object_info")
            source.pop("object_info_sha256")
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
            before = manifest.read_bytes()
            result = PreflightResult(
                artifact_dir=root / "preflight",
                object_info_path=root / "preflight" / "object_info.json",
                active_workflow_dir=None,
                native_i2v_template=None,
                native_flf_template=None,
                status="BLOCKED",
                blockers=("fixture blocker",),
                workflow_candidates=(),
            )

            with patch("wan22_longform.cli.collect_preflight", return_value=result), redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(
                    main(
                        [
                            "preflight",
                            "--comfy-root",
                            str(root),
                            "--project",
                            str(manifest),
                            "--bind-project",
                        ]
                    ),
                    1,
                )

            self.assertEqual(manifest.read_bytes(), before)

    def test_preflight_cli_blocks_a_hash_pinned_graph_with_invalid_native_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_example_manifest(root)
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            invalid_graph = json.loads(
                (PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json").read_text(
                    encoding="utf-8"
                )
            )
            invalid_graph["7"]["inputs"]["unsupported_native_input"] = 1
            invalid_graph_path = root / "invalid-segment-api.json"
            invalid_graph_path.write_text(
                json.dumps(invalid_graph), encoding="utf-8", newline="\n"
            )
            source["workflow_api"] = str(invalid_graph_path)
            source["workflow_hashes"]["segment_api"] = hashlib.sha256(
                invalid_graph_path.read_bytes()
            ).hexdigest()
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")

            object_info = json.loads(
                (PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json").read_text(
                    encoding="utf-8"
                )
            )
            high = source["models"]["high"]
            low = source["models"]["low"]
            vae = source["models"]["vae"]
            text_encoder = source["models"]["text_encoder"]
            inventory = {
                "roots": [
                    {"kind": "diffusion_models", "files": [high, low]},
                    {"kind": "vae", "files": [vae]},
                    {"kind": "text_encoders", "files": [text_encoder]},
                ]
            }
            available = {"available": True, "detail": "fixture"}
            environment = {
                "runtime_interpreter": available,
                "comfyui_revision": available,
                "frontend_version": available,
                "pytorch_version": available,
                "cuda_version": available,
            }
            canonical_i2v = root / "video_wan2_2_14B_i2v.json"
            canonical_flf = root / "video_wan2_2_14B_flf2v.json"
            canonical_i2v.write_text("{}", encoding="utf-8")
            canonical_flf.write_text("{}", encoding="utf-8")
            output = io.StringIO()

            with (
                patch(
                    "wan22_longform.inventory._object_info",
                    return_value=(object_info, "fixture"),
                ),
                patch("wan22_longform.inventory._model_inventory", return_value=inventory),
                patch("wan22_longform.inventory._custom_nodes", return_value={"nodes": []}),
                patch("wan22_longform.inventory._environment", return_value=environment),
                patch(
                    "wan22_longform.inventory._official_templates",
                    return_value=(
                        canonical_i2v,
                        canonical_flf,
                        [canonical_i2v, canonical_flf],
                        "verified",
                        "verified",
                    ),
                ),
                patch(
                    "wan22_longform.inventory._active_workflow_dir",
                    return_value=(None, ()),
                ),
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "preflight",
                        "--comfy-root",
                        str(root),
                        "--artifact-dir",
                        str(root / "preflight"),
                        "--project",
                        str(manifest),
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertTrue(
                any(
                    "segment API workflow is incompatible with captured local /object_info"
                    in blocker
                    and "unsupported_native_input" in blocker
                    for blocker in payload["blockers"]
                )
            )

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
            "target_seconds": 1.0625,
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
            "model_roots": [],
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
                "width": 640,
                "height": 640,
                "frames": 17,
                "generation_fps": 16,
                "review_mp4_codec": "h264",
                "master_codec": "ffv1",
                "seed_base": 1,
            },
            "request": {
                "positive": "neutral fully clothed adult",
                "negative": "low quality",
                "width": 640,
                "height": 640,
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
                    "target_seconds": 1.0625,
                    "anchor_image": str(opening),
                    "segments": [
                        {
                            "id": "S010_C001",
                            "action": "A neutral adult pauses naturally.",
                            "expected_seconds": 1.0625,
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
        model_roots = {
            "diffusion_models": root / "models" / "diffusion_models",
            "vae": root / "models" / "vae",
            "text_encoders": root / "models" / "text_encoders",
        }
        for directory in model_roots.values():
            directory.mkdir(parents=True, exist_ok=True)
        (model_roots["diffusion_models"] / "high.safetensors").write_bytes(b"high")
        (model_roots["diffusion_models"] / "low.safetensors").write_bytes(b"low")
        (model_roots["vae"] / "vae.safetensors").write_bytes(b"vae")
        (model_roots["text_encoders"] / "text-encoder.safetensors").write_bytes(
            b"text"
        )
        source["model_roots"] = [str(directory) for directory in model_roots.values()]
        object_info = root / "object_info.json"
        object_info.write_bytes(
            (PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json").read_bytes()
        )
        source["object_info"] = str(object_info)
        source["object_info_sha256"] = hashlib.sha256(object_info.read_bytes()).hexdigest()
        manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        return manifest
