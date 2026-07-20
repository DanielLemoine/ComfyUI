from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from subprocess import CompletedProcess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.errors import PreflightError
from wan22_longform import inventory
from wan22_longform.config import ProjectConfig, load_project
from wan22_longform.inventory import collect_preflight


class RedirectingObjectInfoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/object_info")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class CollectPreflightTests(unittest.TestCase):
    def _object_info_fixture(self) -> dict[str, object]:
        return json.loads(
            (PROJECT_DIR / "tests" / "fixtures" / "object_info.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _package_template_dir(temporary_path: Path) -> Path:
        return (
            temporary_path
            / "ComfyUI"
            / "venv"
            / "Lib"
            / "site-packages"
            / "comfyui_workflow_templates_json"
            / "templates"
        )

    @staticmethod
    def _write_registered_flf_manifest(
        template_dir: Path, asset_hash: str
    ) -> None:
        CollectPreflightTests._write_registered_manifest(
            template_dir,
            {"video_wan2_2_14B_flf2v": asset_hash},
        )

    @staticmethod
    def _write_registered_manifest(
        template_dir: Path, asset_hashes: dict[str, str]
    ) -> None:
        manifest_path = (
            template_dir.parent.parent
            / "comfyui_workflow_templates_core"
            / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "templates": [
                        {
                            "id": template_id,
                            "bundle": "media-video",
                            "assets": [
                                {
                                    "filename": f"{template_id}.json",
                                    "sha256": asset_hash,
                                }
                            ],
                        }
                        for template_id, asset_hash in asset_hashes.items()
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _ready_fixture(
        self, temporary_path: Path, *, runtime: bool = True
    ) -> tuple[Path, ProjectConfig, str, str]:
        comfy_root = temporary_path / "ComfyUI"
        template_dir = self._package_template_dir(temporary_path)
        template_dir.mkdir(parents=True)
        i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
        i2v_template.write_text(
            json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
            encoding="utf-8",
        )
        flf_template = template_dir / "video_wan2_2_14B_flf2v.json"
        flf_template.write_text(
            json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
            encoding="utf-8",
        )
        i2v_hash = hashlib.sha256(i2v_template.read_bytes()).hexdigest()
        flf_hash = hashlib.sha256(flf_template.read_bytes()).hexdigest()
        self._write_registered_manifest(
            template_dir,
            {
                "video_wan2_2_14B_i2v": i2v_hash,
                "video_wan2_2_14B_flf2v": flf_hash,
            },
        )
        model_names = {
            "high": "high.safetensors",
            "low": "low.safetensors",
            "vae": "vae.safetensors",
            "text_encoder": "text.safetensors",
        }
        model_kinds = {
            "high": "diffusion_models",
            "low": "diffusion_models",
            "vae": "vae",
            "text_encoder": "text_encoders",
        }
        for role, name in model_names.items():
            model_dir = comfy_root / "models" / model_kinds[role]
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / name).write_bytes(name.encode("utf-8"))
        if runtime:
            interpreter = comfy_root / "venv" / "Scripts" / "python.exe"
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_bytes(b"fixture interpreter")
        project_path = temporary_path / "project.yaml"
        project_path.write_text("project_id: preflight-fixture\n", encoding="utf-8")
        project = ProjectConfig(
            path=project_path,
            source={"project_id": "preflight-fixture", "models": model_names},
        )
        return comfy_root, project, i2v_hash, flf_hash

    def test_canonical_flf_template_identity_is_pinned(self) -> None:
        self.assertEqual(
            inventory._CANONICAL_FLF_TEMPLATE_ID, "video_wan2_2_14B_flf2v"
        )
        self.assertEqual(
            inventory._CANONICAL_FLF_TEMPLATE_FILENAME,
            "video_wan2_2_14B_flf2v.json",
        )
        self.assertEqual(
            inventory._CANONICAL_FLF_TEMPLATE_SHA256,
            "9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf",
        )

    def test_canonical_i2v_template_identity_is_pinned(self) -> None:
        self.assertEqual(
            getattr(inventory, "_CANONICAL_I2V_TEMPLATE_ID", None),
            "video_wan2_2_14B_i2v",
        )
        self.assertEqual(
            getattr(inventory, "_CANONICAL_I2V_TEMPLATE_FILENAME", None),
            "video_wan2_2_14B_i2v.json",
        )
        self.assertEqual(
            getattr(inventory, "_CANONICAL_I2V_TEMPLATE_SHA256", None),
            "6eea9b627b10fcfaf3e75a43aad2c58d8daabdbf72b32ede1602c668cac376bb",
        )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_writes_required_evidence(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        object_info_fixture = self._object_info_fixture()
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

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_blocks_empty_or_incomplete_object_info(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        cases = (
            ({}, "object_info is empty or unavailable"),
            (
                {"WanImageToVideo": {}},
                "required object-info schema WanFirstLastFrameToVideo is unavailable",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            for object_info, expected_blocker in cases:
                with self.subTest(object_info=object_info):
                    result = collect_preflight(
                        comfy_root=temporary_path / "ComfyUI",
                        comfy_url=None,
                        artifact_dir=temporary_path / "artifacts" / str(len(object_info)),
                        object_info=object_info,
                    )

                    self.assertEqual(result.status, "BLOCKED")
                    self.assertTrue(
                        any(expected_blocker in blocker for blocker in result.blockers)
                    )
                    self.assertIn(
                        expected_blocker,
                        (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
                    )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_blocks_empty_or_malformed_native_schema(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        cases = (
            {
                "WanImageToVideo": {},
                "WanFirstLastFrameToVideo": {},
            },
            {
                "WanImageToVideo": {"input": {"required": []}},
                "WanFirstLastFrameToVideo": {"input": {}},
            },
        )
        expected_blockers = (
            "required object-info schema WanImageToVideo lacks a non-empty input.required mapping",
            "required object-info schema WanFirstLastFrameToVideo lacks a non-empty input.required mapping",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            for case_index, object_info in enumerate(cases):
                with self.subTest(object_info=object_info):
                    result = collect_preflight(
                        comfy_root=temporary_path / "ComfyUI",
                        comfy_url=None,
                        artifact_dir=temporary_path / "artifacts" / str(case_index),
                        object_info=object_info,
                    )

                    self.assertEqual(result.status, "BLOCKED")
                    for expected_blocker in expected_blockers:
                        self.assertTrue(
                            any(expected_blocker in blocker for blocker in result.blockers)
                        )
                        self.assertIn(
                            expected_blocker,
                            (result.artifact_dir / "preflight_report.md").read_text(
                                encoding="utf-8"
                            ),
                        )

    def test_collect_preflight_rejects_local_object_info_redirects(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectingObjectInfoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory)
                with self.assertRaisesRegex(PreflightError, "redirects are not allowed"):
                    collect_preflight(
                        comfy_root=temporary_path / "ComfyUI",
                        comfy_url=f"http://127.0.0.1:{server.server_port}",
                        artifact_dir=temporary_path / "artifacts" / "preflight",
                    )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    @patch("wan22_longform.inventory.subprocess.run")
    @patch("wan22_longform.inventory.build_opener")
    def test_collect_preflight_blocks_unavailable_object_info(self, build_opener, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        build_opener.return_value.open.side_effect = OSError("fixture unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url="http://127.0.0.1:8188",
                artifact_dir=temporary_path / "artifacts" / "preflight",
            )

            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("object_info is empty or unavailable" in blocker for blocker in result.blockers)
            )
            self.assertIn(
                "fixture unavailable",
                (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
            )

    @patch("wan22_longform.inventory.subprocess.run")
    @patch("wan22_longform.inventory.json.load")
    @patch("wan22_longform.inventory.build_opener")
    def test_collect_preflight_blocks_malformed_local_object_info(
        self, build_opener, load, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        load.side_effect = json.JSONDecodeError("fixture malformed JSON", "{", 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url="http://127.0.0.1:8188",
                artifact_dir=temporary_path / "artifacts" / "preflight",
            )

            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("object_info is empty or unavailable" in blocker for blocker in result.blockers)
            )
            self.assertEqual(
                json.loads(result.object_info_path.read_text(encoding="utf-8")), {}
            )
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
            self.assertIn(
                "was not valid JSON",
                (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_unregistered_flf_in_workflow_templates(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = temporary_path / "ComfyUI" / "workflow_templates"
            template_dir.mkdir(parents=True)
            misleading = template_dir / "aaa_wan_i2v_official.json"
            misleading.write_text(
                json.dumps({"type": "WanImageToVideo"}), encoding="utf-8"
            )
            valid_i2v = template_dir / "zzz_wan_i2v_native.json"
            valid_i2v.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}), encoding="utf-8"
            )
            valid_flf = template_dir / "wan_flf_native.json"
            valid_flf.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_i2v_template)
            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertIn(
                "official native Wan FLF template could not be verified",
                (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_same_node_flf_from_standard_template_roots(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        standard_roots = (
            Path("blueprints"),
            Path("workflow_templates"),
            Path("web") / "assets" / "workflow_templates",
        )
        for relative_root in standard_roots:
            with self.subTest(relative_root=relative_root):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    template_dir = temporary_path / "ComfyUI" / relative_root
                    template_dir.mkdir(parents=True)
                    i2v_template = template_dir / "wan_i2v_native.json"
                    i2v_template.write_text(
                        json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                        encoding="utf-8",
                    )
                    fake_flf = template_dir / "unregistered_same_node_flf.json"
                    fake_flf.write_text(
                        json.dumps(
                            {"nodes": [{"type": "WanFirstLastFrameToVideo"}]}
                        ),
                        encoding="utf-8",
                    )

                    result = collect_preflight(
                        comfy_root=temporary_path / "ComfyUI",
                        comfy_url=None,
                        artifact_dir=temporary_path / "artifacts" / "preflight",
                        object_info=self._object_info_fixture(),
                    )

                    self.assertIsNone(result.native_i2v_template)
                    self.assertIsNone(result.native_flf_template)
                    self.assertEqual(result.status, "BLOCKED")
                    self.assertTrue(
                        any(
                            "registered canonical Wan FLF package asset" in blocker
                            for blocker in result.blockers
                        )
                    )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_wan_i2v_blueprint_subgraph(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            blueprint_dir = temporary_path / "ComfyUI" / "blueprints"
            blueprint_dir.mkdir(parents=True)
            i2v_blueprint = blueprint_dir / "Image to Video (Wan 2.2).json"
            i2v_blueprint.write_text(
                json.dumps(
                    {
                        "nodes": [{"id": 1, "type": "uuid-subgraph-id"}],
                        "definitions": {
                            "subgraphs": [
                                {"nodes": [{"id": 2, "type": "WanImageToVideo"}]}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_i2v_template)
            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any(
                    "official native Wan I2V template could not be verified" in blocker
                    for blocker in result.blockers
                )
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_package_shaped_blueprint_tree_cannot_qualify_as_registered(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = (
                temporary_path
                / "ComfyUI"
                / "blueprints"
                / "comfyui_workflow_templates_json"
                / "templates"
            )
            template_dir.mkdir(parents=True)
            i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
            i2v_template.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            i2v_hash = hashlib.sha256(i2v_template.read_bytes()).hexdigest()
            self._write_registered_manifest(
                template_dir, {"video_wan2_2_14B_i2v": i2v_hash}
            )

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ):
                result = collect_preflight(
                    comfy_root=temporary_path / "ComfyUI",
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                )

            self.assertIsNone(result.native_i2v_template)
            self.assertTrue(
                any("registered canonical Wan I2V" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_records_runtime_revisions_and_filters_non_nodes(self, run) -> None:
        run.return_value = CompletedProcess([], 1, "", "unavailable fixture")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            comfy_root = temporary_path / "ComfyUI"
            for name in ("real-node", "__pycache__", "tests"):
                (comfy_root / "custom_nodes" / name).mkdir(parents=True)
            for profile in ("alice", "default"):
                (comfy_root / "user" / profile / "workflows").mkdir(parents=True)

            result = collect_preflight(
                comfy_root=comfy_root,
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            environment = json.loads(
                (result.artifact_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(environment).intersection(
                    {"comfyui_revision", "frontend_version", "pytorch_version", "cuda_version"}
                ),
                {"comfyui_revision", "frontend_version", "pytorch_version", "cuda_version"},
            )
            custom_nodes = json.loads(
                (result.artifact_dir / "custom_nodes.json").read_text(encoding="utf-8")
            )
            self.assertEqual([node["name"] for node in custom_nodes["nodes"]], ["real-node"])
            self.assertIn("revision", custom_nodes["nodes"][0])
            self.assertIsNone(result.active_workflow_dir)
            self.assertEqual(
                {path.parent.name for path in result.workflow_candidates},
                {"alice", "default"},
            )

    def test_preflight_blocks_without_project_model_role_proof(self) -> None:
        blockers = inventory._preflight_blockers(
            self._object_info_fixture(),
            Path("i2v.json"),
            Path("flf.json"),
            "verified fixture",
            "verified fixture",
            {"roots": [], "required_roles": {}},
            None,
        )

        self.assertTrue(any("project model-role proof" in blocker for blocker in blockers))

    def test_preflight_blocks_a_required_model_absent_from_local_inventory(self) -> None:
        project = ProjectConfig(
            path=Path("project.yaml"),
            source={
                "models": {
                    "high": "high.safetensors",
                    "low": "low.safetensors",
                    "vae": "vae.safetensors",
                    "text_encoder": "text.safetensors",
                }
            },
        )
        blockers = inventory._preflight_blockers(
            self._object_info_fixture(),
            Path("i2v.json"),
            Path("flf.json"),
            "verified fixture",
            "verified fixture",
            {
                "roots": [
                    {
                        "files": [
                            "high.safetensors",
                            "low.safetensors",
                            "text.safetensors",
                        ]
                    }
                ],
                "required_roles": {},
            },
            project,
        )

        self.assertTrue(
            any("model vae is absent" in blocker for blocker in blockers)
        )

    def test_required_model_roles_support_the_frozen_manifest_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "project.yaml"
            manifest.write_text(
                """models:
  high: high.safetensors
  low: low.safetensors
  vae: vae.safetensors
  text_encoder: text.safetensors
""",
                encoding="utf-8",
            )

            roles = inventory._required_model_roles(load_project(manifest))

        self.assertEqual(
            roles,
            {
                "high": "high.safetensors",
                "low": "low.safetensors",
                "vae": "vae.safetensors",
                "text_encoder": "text.safetensors",
            },
        )

    def test_preflight_rejects_models_found_only_in_wrong_root_kinds(self) -> None:
        project = ProjectConfig(
            path=Path("project.yaml"),
            source={
                "models": {
                    "high": "high.safetensors",
                    "low": "low.safetensors",
                    "vae": "vae.safetensors",
                    "text_encoder": "text.safetensors",
                }
            },
        )
        blockers = inventory._model_role_blockers(
            {
                "roots": [
                    {
                        "kind": "vae",
                        "files": [
                            "high.safetensors",
                            "low.safetensors",
                            "vae.safetensors",
                            "text.safetensors",
                        ],
                    }
                ]
            },
            project,
        )

        self.assertTrue(any("model high is absent" in blocker for blocker in blockers))
        self.assertTrue(any("model low is absent" in blocker for blocker in blockers))
        self.assertFalse(any("model vae is absent" in blocker for blocker in blockers))
        self.assertTrue(
            any("model text_encoder is absent" in blocker for blocker in blockers)
        )

    def test_model_inventory_includes_default_roots_and_configured_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comfy_root = Path(temporary_directory) / "ComfyUI"
            default_root = comfy_root / "models" / "vae"
            default_root.mkdir(parents=True)
            extra_root = Path(temporary_directory) / "extra-diffusion"
            extra_root.mkdir()
            (comfy_root / "extra_model_paths.yaml").write_text(
                "\n".join(
                    (
                        "fixture:",
                        f"  base_path: {extra_root.parent}",
                        f"  diffusion_models: {extra_root.name}",
                    )
                ),
                encoding="utf-8",
            )

            roots = inventory._configured_model_roots(comfy_root)

            self.assertIn(("vae", default_root.resolve()), roots)
            self.assertIn(("diffusion_models", extra_root.resolve()), roots)

    def test_model_inventory_parses_canonical_block_scalars_and_multiple_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comfy_root = root / "ComfyUI"
            default_root = comfy_root / "models" / "text_encoders"
            default_root.mkdir(parents=True)
            primary_root = root / "shared" / "diffusion_models"
            secondary_root = root / "shared" / "vae"
            primary_root.mkdir(parents=True)
            secondary_root.mkdir(parents=True)
            first = comfy_root / "extra_model_paths.yaml"
            first.parent.mkdir(parents=True, exist_ok=True)
            first.write_text(
                """primary:
  base_path: ../shared
  is_default: true
  diffusion_models: |
    diffusion_models
""",
                encoding="utf-8",
            )
            second = root / "additional-model-paths.yaml"
            second.write_text(
                """secondary:
  base_path: shared
  vae:
    - vae
""",
                encoding="utf-8",
            )

            roots = inventory._configured_model_roots(
                comfy_root, extra_model_path_configs=(first, second)
            )

            self.assertIn(("text_encoders", default_root.resolve()), roots)
            self.assertIn(("diffusion_models", primary_root.resolve()), roots)
            self.assertIn(("vae", secondary_root.resolve()), roots)

    def test_model_inventory_ignores_stale_extra_model_path_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comfy_root = root / "ComfyUI"
            (comfy_root / "models" / "vae").mkdir(parents=True)
            stale_root = root / "stale-diffusion"
            stale_root.mkdir()
            (comfy_root / "extra_model_paths.backup.yaml").write_text(
                f"""stale:
  base_path: {root}
  diffusion_models: {stale_root.name}
""",
                encoding="utf-8",
            )

            roots = inventory._configured_model_roots(comfy_root)

        self.assertNotIn(("diffusion_models", stale_root.resolve()), roots)

    def test_model_inventory_rejects_missing_explicit_active_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comfy_root = Path(temporary_directory) / "ComfyUI"
            comfy_root.mkdir()
            missing = Path(temporary_directory) / "active-extra-model-paths.yaml"

            with self.assertRaisesRegex(PreflightError, "explicit extra_model_paths config"):
                inventory._configured_model_roots(
                    comfy_root, extra_model_path_configs=(missing,)
                )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_preflight_blocks_without_a_trustworthy_comfy_runtime(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            comfy_root, project, i2v_hash, flf_hash = self._ready_fixture(
                temporary_path, runtime=False
            )

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ), patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256", flf_hash
            ):
                result = collect_preflight(
                    comfy_root=comfy_root,
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                    project=project,
                )

            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("runtime interpreter" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_runtime_queries_use_the_comfy_environment_not_the_helper_interpreter(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            comfy_root, project, i2v_hash, flf_hash = self._ready_fixture(temporary_path)
            runtime = comfy_root / "venv" / "Scripts" / "python.exe"
            helper = temporary_path / "helper" / "python.exe"

            with patch("sys.executable", str(helper)), patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ), patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256", flf_hash
            ):
                result = collect_preflight(
                    comfy_root=comfy_root,
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                    project=project,
                )

            environment = json.loads(
                (result.artifact_dir / "environment.json").read_text(encoding="utf-8")
            )
            python_commands = [
                call.args[0]
                for call in run.call_args_list
                if "-c" in call.args[0]
            ]
            self.assertTrue(python_commands)
            self.assertTrue(
                all(Path(command[0]) == runtime.resolve() for command in python_commands)
            )
            self.assertEqual(
                Path(environment["python"]["executable"]), runtime.resolve()
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_cpu_only_runtime_cuda_sentinel_blocks_ready(self, run) -> None:
        def command_result(command, **_kwargs):
            if any("torch.version.cuda" in str(part) for part in command):
                return CompletedProcess(command, 0, "unavailable\n", "")
            return CompletedProcess(command, 0, "fixture output", "")

        run.side_effect = command_result
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            comfy_root, project, i2v_hash, flf_hash = self._ready_fixture(temporary_path)

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ), patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256", flf_hash
            ):
                result = collect_preflight(
                    comfy_root=comfy_root,
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                    project=project,
                )

            environment = json.loads(
                (result.artifact_dir / "environment.json").read_text(encoding="utf-8")
            )
            self.assertFalse(environment["cuda_version"]["available"])
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("cuda_version" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_unavailable_custom_node_revision_blocks_ready(self, run) -> None:
        def command_result(command, **_kwargs):
            if "custom_nodes" in " ".join(str(part) for part in command):
                return CompletedProcess(command, 1, "", "not a git repository")
            return CompletedProcess(command, 0, "fixture output", "")

        run.side_effect = command_result
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            comfy_root, project, i2v_hash, flf_hash = self._ready_fixture(temporary_path)
            (comfy_root / "custom_nodes" / "fixture-node").mkdir(parents=True)

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ), patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256", flf_hash
            ):
                result = collect_preflight(
                    comfy_root=comfy_root,
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                    project=project,
                )

            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("custom node fixture-node revision" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_an_i2v_package_asset_missing_its_manifest_entry(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
            i2v_template.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            self._write_registered_flf_manifest(template_dir, "0" * 64)

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_i2v_template)
            self.assertTrue(
                any("I2V JSON has no registered" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_changed_registered_i2v_bytes(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
            i2v_template.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}], "changed": True}),
                encoding="utf-8",
            )
            registered_hash = hashlib.sha256(b"official fixture bytes").hexdigest()
            self._write_registered_manifest(
                template_dir, {"video_wan2_2_14B_i2v": registered_hash}
            )

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256",
                registered_hash,
            ):
                result = collect_preflight(
                    comfy_root=temporary_path / "ComfyUI",
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                )

            self.assertIsNone(result.native_i2v_template)
            self.assertTrue(
                any("package I2V JSON SHA-256" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_blocks_unregistered_package_flf_template(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
            flf_template = template_dir / "video_wan2_2_14B_flf2v.json"
            i2v_template.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            flf_template.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_i2v_template)
            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("registered ComfyUI manifest entry" in blocker for blocker in result.blockers)
            )
            self.assertIn(
                "registered ComfyUI manifest entry",
                (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_blocks_tampered_registered_package_flf_template(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            (template_dir / "video_wan2_2_14B_i2v.json").write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            flf_template = template_dir / "video_wan2_2_14B_flf2v.json"
            flf_template.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )
            self._write_registered_flf_manifest(
                template_dir,
                "9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any("SHA-256" in blocker for blocker in result.blockers)
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_blocks_registered_flf_with_unpinned_manifest_hash(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            (template_dir / "video_wan2_2_14B_i2v.json").write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            flf_template = template_dir / "video_wan2_2_14B_flf2v.json"
            flf_template.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )
            self._write_registered_flf_manifest(
                template_dir,
                hashlib.sha256(flf_template.read_bytes()).hexdigest(),
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any(
                    "manifest SHA-256" in blocker and "pinned official hash" in blocker
                    for blocker in result.blockers
                )
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_recognizes_verified_registered_wan_template_package(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            i2v_template = template_dir / "video_wan2_2_14B_i2v.json"
            i2v_template.write_text(
                json.dumps({"nodes": [{"type": "WanImageToVideo"}]}),
                encoding="utf-8",
            )
            flf_template = template_dir / "video_wan2_2_14B_flf2v.json"
            flf_template.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )
            expected_hash = hashlib.sha256(flf_template.read_bytes()).hexdigest()
            i2v_hash = hashlib.sha256(i2v_template.read_bytes()).hexdigest()
            self._write_registered_manifest(
                template_dir,
                {
                    "video_wan2_2_14B_i2v": i2v_hash,
                    "video_wan2_2_14B_flf2v": expected_hash,
                },
            )
            model_names = {
                "high": "high.safetensors",
                "low": "low.safetensors",
                "vae": "vae.safetensors",
                "text_encoder": "text.safetensors",
            }
            model_kinds = {
                "high": "diffusion_models",
                "low": "diffusion_models",
                "vae": "vae",
                "text_encoder": "text_encoders",
            }
            for role, name in model_names.items():
                model_dir = temporary_path / "ComfyUI" / "models" / model_kinds[role]
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / name).write_bytes(name.encode("utf-8"))
            interpreter = (
                temporary_path
                / "ComfyUI"
                / "venv"
                / "Scripts"
                / "python.exe"
            )
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_bytes(b"fixture interpreter")
            project_path = temporary_path / "project.yaml"
            project_path.write_text("project_id: preflight-fixture\n", encoding="utf-8")
            project = ProjectConfig(
                path=project_path,
                source={"project_id": "preflight-fixture", "models": model_names},
            )

            with patch(
                "wan22_longform.inventory._CANONICAL_I2V_TEMPLATE_SHA256", i2v_hash
            ), patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256", expected_hash
            ):
                result = collect_preflight(
                    comfy_root=temporary_path / "ComfyUI",
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                    project=project,
                )

            self.assertEqual(result.native_i2v_template, i2v_template.resolve())
            self.assertEqual(result.native_flf_template, flf_template.resolve())
            self.assertEqual(result.status, "READY")
            self.assertEqual(result.blockers, ())

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_prefers_the_canonical_wan_flf_template(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            template_dir = self._package_template_dir(temporary_path)
            template_dir.mkdir(parents=True)
            (template_dir / "gsl_starter_1_2.json").write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )
            canonical_flf = template_dir / "video_wan2_2_14B_flf2v.json"
            canonical_flf.write_text(
                json.dumps({"nodes": [{"type": "WanFirstLastFrameToVideo"}]}),
                encoding="utf-8",
            )
            expected_hash = hashlib.sha256(canonical_flf.read_bytes()).hexdigest()
            self._write_registered_flf_manifest(template_dir, expected_hash)

            with patch(
                "wan22_longform.inventory._CANONICAL_FLF_TEMPLATE_SHA256",
                expected_hash,
            ):
                result = collect_preflight(
                    comfy_root=temporary_path / "ComfyUI",
                    comfy_url=None,
                    artifact_dir=temporary_path / "artifacts" / "preflight",
                    object_info=self._object_info_fixture(),
                )

            self.assertEqual(result.native_flf_template, canonical_flf.resolve())

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_does_not_treat_user_workflows_as_official_templates(
        self, run
    ) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            user_workflow_dir = (
                temporary_path / "ComfyUI" / "user" / "default" / "workflows"
            )
            user_workflow_dir.mkdir(parents=True)
            (user_workflow_dir / "wan_flf2v.json").write_text(
                json.dumps(
                    {"nodes": [{"type": "WanFirstLastFrameToVideo"}]}
                ),
                encoding="utf-8",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_rejects_ltx_blueprint_subgraph(self, run) -> None:
        run.return_value = CompletedProcess([], 0, "fixture output", "")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            blueprint_dir = temporary_path / "ComfyUI" / "blueprints"
            blueprint_dir.mkdir(parents=True)
            ltx_blueprint = blueprint_dir / "First-Last-Frame to Video.json"
            ltx_blueprint.write_text(
                json.dumps(
                    {
                        "nodes": [{"id": 1, "type": "uuid-subgraph-id"}],
                        "definitions": {
                            "subgraphs": [
                                {"nodes": [{"id": 2, "type": "LTXVConditioning"}]}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = collect_preflight(
                comfy_root=temporary_path / "ComfyUI",
                comfy_url=None,
                artifact_dir=temporary_path / "artifacts" / "preflight",
                object_info=self._object_info_fixture(),
            )

            self.assertIsNone(result.native_i2v_template)
            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any(
                    "official native Wan I2V template could not be verified" in blocker
                    for blocker in result.blockers
                )
            )
            self.assertTrue(
                any(
                    "official native Wan FLF template could not be verified" in blocker
                    for blocker in result.blockers
                )
            )


if __name__ == "__main__":
    unittest.main()
