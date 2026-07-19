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
                            "id": "video_wan2_2_14B_flf2v",
                            "bundle": "media-video",
                            "assets": [
                                {
                                    "filename": "video_wan2_2_14B_flf2v.json",
                                    "sha256": asset_hash,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

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
    def test_collect_preflight_rejects_misleading_template_filename(self, run) -> None:
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

            self.assertEqual(result.native_i2v_template, valid_i2v.resolve())
            self.assertEqual(result.native_flf_template, valid_flf.resolve())
            self.assertEqual(result.status, "READY")
            self.assertEqual(result.blockers, ())
            self.assertIn(
                "- READY: all required native schemas and templates were verified.",
                (result.artifact_dir / "preflight_report.md").read_text(encoding="utf-8"),
            )

    @patch("wan22_longform.inventory.subprocess.run")
    def test_collect_preflight_recognizes_wan_i2v_blueprint_subgraph(self, run) -> None:
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

            self.assertEqual(result.native_i2v_template, i2v_blueprint.resolve())
            self.assertIsNone(result.native_flf_template)
            self.assertEqual(result.status, "BLOCKED")
            self.assertTrue(
                any(
                    "official native Wan FLF template could not be verified" in blocker
                    for blocker in result.blockers
                )
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

            self.assertEqual(result.native_i2v_template, i2v_template.resolve())
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
