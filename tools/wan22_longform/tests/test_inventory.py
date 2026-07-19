from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
