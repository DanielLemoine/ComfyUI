from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.comfy_client import (  # noqa: E402
    ComfyClient,
    ComfyClientError,
    HistoryOutput,
    HistoryResult,
)
from wan22_longform.config import ProjectConfig  # noqa: E402
from wan22_longform.project import AttemptState  # noqa: E402
from wan22_longform.qc import read_qc  # noqa: E402
from wan22_longform.render import render_segment  # noqa: E402


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
        self.requests.append((method, url, body, headers))
        return json.dumps(self.responses.pop(0)).encode("utf-8")


class FakeRenderClient:
    def __init__(self, video: Path) -> None:
        self.video = video
        self.submitted: list[dict[str, dict[str, object]]] = []
        self.waited: list[str] = []
        self.uploaded: list[Path] = []

    def upload_image(self, path: Path) -> str:
        self.uploaded.append(path)
        return "uploaded-opening.png"

    def submit(self, graph: dict[str, dict[str, object]]) -> str:
        self.submitted.append(graph)
        return "fixture-prompt"

    def wait(self, prompt_id: str) -> HistoryResult:
        self.waited.append(prompt_id)
        return HistoryResult(
            prompt_id=prompt_id,
            outputs=(
                HistoryOutput(
                    node_id="14",
                    filename="fixture.mp4",
                    subfolder="video",
                    type="output",
                    local_path=self.video,
                ),
            ),
        )

    def fetch_output(self, output: HistoryOutput, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output.local_path, destination)
        return destination


class ComfyClientTests(unittest.TestCase):
    def test_submit_and_wait_use_only_the_configured_loopback_origin(self) -> None:
        transport = FakeTransport(
            [
                {"prompt_id": "prompt-123"},
                {
                    "prompt-123": {
                        "status": {"completed": True},
                        "outputs": {
                            "14": {
                                "gifs": [
                                    {
                                        "filename": "segment.mp4",
                                        "subfolder": "clips",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            ]
        )
        client = ComfyClient(
            "http://127.0.0.1:8188", transport=transport, poll_interval=0
        )

        prompt_id = client.submit({"1": {"class_type": "Fixture", "inputs": {}}})
        history = client.wait(prompt_id)

        self.assertEqual(prompt_id, "prompt-123")
        self.assertEqual(history.outputs[0].filename, "segment.mp4")
        self.assertEqual(
            [request[1] for request in transport.requests],
            [
                "http://127.0.0.1:8188/prompt",
                "http://127.0.0.1:8188/history/prompt-123",
            ],
        )
        submitted = json.loads(transport.requests[0][2])
        self.assertEqual(submitted["prompt"]["1"]["class_type"], "Fixture")

    def test_upload_image_uses_local_multipart_endpoint(self) -> None:
        transport = FakeTransport([{"name": "opening.png", "subfolder": "", "type": "input"}])
        client = ComfyClient("http://[::1]:8188", transport=transport)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image = Path(temporary_directory) / "opening.png"
            image.write_bytes(b"fixture-image")

            uploaded_name = client.upload_image(image)

        method, url, body, headers = transport.requests[0]
        self.assertEqual(uploaded_name, "opening.png")
        self.assertEqual((method, url), ("POST", "http://[::1]:8188/upload/image"))
        self.assertIn(b"fixture-image", body)
        self.assertTrue(headers["Content-Type"].startswith("multipart/form-data; boundary="))

    def test_non_loopback_or_ambiguous_origins_are_rejected(self) -> None:
        for url in (
            "https://example.com:8188",
            "http://192.168.1.10:8188",
            "http://user:password@127.0.0.1:8188",
            "http://127.0.0.1:8188/base",
            "http://127.0.0.1:8188?proxy=http://example.com",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ComfyClientError, "loopback|origin"):
                    ComfyClient(url, transport=FakeTransport([]))

    def test_history_prompt_id_cannot_change_the_local_request_target(self) -> None:
        client = ComfyClient(
            "http://127.0.0.1:8188", transport=FakeTransport([]), poll_interval=0
        )

        for prompt_id in ("../prompt", "prompt?redirect=http://example.com", "prompt#other"):
            with self.subTest(prompt_id=prompt_id):
                with self.assertRaisesRegex(ComfyClientError, "path component"):
                    client.wait(prompt_id)

    def test_history_rejects_traversal_subfolder_before_view_request(self) -> None:
        transport = FakeTransport(
            [
                {
                    "prompt-123": {
                        "status": {"completed": True},
                        "outputs": {
                            "14": {
                                "gifs": [
                                    {
                                        "filename": "segment.mp4",
                                        "subfolder": "../outside",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                }
            ]
        )
        client = ComfyClient("http://127.0.0.1:8188", transport=transport)

        with self.assertRaisesRegex(ComfyClientError, "subfolder"):
            client.wait("prompt-123")

        self.assertEqual(
            [request[1] for request in transport.requests],
            ["http://127.0.0.1:8188/history/prompt-123"],
        )

    def test_history_rejects_unexpected_output_type_before_view_request(self) -> None:
        transport = FakeTransport(
            [
                {
                    "prompt-123": {
                        "status": {"completed": True},
                        "outputs": {
                            "14": {
                                "gifs": [
                                    {
                                        "filename": "segment.mp4",
                                        "subfolder": "clips",
                                        "type": "temp",
                                    }
                                ]
                            }
                        },
                    }
                }
            ]
        )
        client = ComfyClient("http://127.0.0.1:8188", transport=transport)

        with self.assertRaisesRegex(ComfyClientError, "output type"):
            client.wait("prompt-123")

        self.assertEqual(
            [request[1] for request in transport.requests],
            ["http://127.0.0.1:8188/history/prompt-123"],
        )


class RenderSegmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "history-video.mp4"
        self.video.write_bytes(b"fixture-video")
        self.opening = self.root / "opening.png"
        self.opening.write_bytes(b"opening-frame")
        self.workflow = PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"
        self.manifest = self.root / "project.yaml"
        source = {
            "preset": "P0_IDENTITY_BASELINE",
            "models": {
                "high": "configured-high.safetensors",
                "low": "configured-low.safetensors",
            },
            "workflow_api": str(self.workflow),
            "request": {
                "positive": "configured positive prompt",
                "negative": "configured negative prompt",
                "seed": 424242,
                "width": 832,
                "height": 480,
                "frames": 81,
            },
            "inputs": {"opening_frame": str(self.opening)},
            "attempts_dir": str(self.root / "attempts"),
            "model_files": [
                "configured-high.safetensors",
                "configured-low.safetensors",
            ],
            "candidate_count": 5,
        }
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        self.project = ProjectConfig(path=self.manifest, source=source)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_render_initializes_qc_needs_review_and_hash_linked_artifacts(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            start = 0 if where == "head" else 76
            for index in range(start, start + count):
                frame = destination / f"frame-{index:06d}.png"
                frame.write_bytes(f"{where}-{index}".encode("utf-8"))
                frames.append(frame)
            return frames

        def fake_contact_sheet(_frames: list[Path], destination: Path) -> Path:
            destination.write_bytes(b"contact-sheet")
            return destination

        extract.side_effect = fake_extract
        contact_sheet.side_effect = fake_contact_sheet
        client = FakeRenderClient(self.video)
        manifest_before = self.manifest.read_bytes()

        attempt = render_segment(self.project, "S010", "S010_C001", client)

        self.assertEqual(attempt.state, AttemptState.NEEDS_REVIEW)
        qc = read_qc(attempt.path / "qc.yaml")
        self.assertEqual(qc["status"], "needs_review")
        self.assertIsNone(qc["selected_continuation_frame"])
        self.assertFalse(qc["automatic_continuation_authorized"])
        self.assertEqual(len(qc["candidate_frames"]["head"]), 5)
        self.assertEqual(len(qc["candidate_frames"]["tail"]), 5)
        self.assertTrue((attempt.path / "submission-request.json").is_file())
        self.assertTrue((attempt.path / "configured-workflow-api.json").is_file())
        self.assertEqual(client.waited, ["fixture-prompt"])
        self.assertEqual(client.uploaded, [self.opening])
        metadata = json.loads(
            (attempt.path / "render-metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["details"]["prompt_id"], "fixture-prompt")
        self.assertEqual(
            metadata["outputs"]["segment"]["sha256"],
            hashlib.sha256(b"fixture-video").hexdigest(),
        )
        self.assertIn("contact_sheet", metadata["outputs"])
        self.assertEqual(len(client.submitted), 1)
        graph = client.submitted[0]
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "configured-high.safetensors")
        self.assertEqual(graph["2"]["inputs"]["unet_name"], "configured-low.safetensors")
        self.assertEqual(graph["7"]["inputs"]["text"], "configured positive prompt")
        self.assertEqual(graph["8"]["inputs"]["text"], "configured negative prompt")
        self.assertEqual(graph["9"]["inputs"]["image"], "uploaded-opening.png")
        self.assertEqual(graph["10"]["inputs"]["width"], 832)
        self.assertEqual(graph["10"]["inputs"]["height"], 480)
        self.assertEqual(graph["10"]["inputs"]["length"], 81)
        self.assertEqual(graph["11"]["inputs"]["noise_seed"], 424242)
        self.assertEqual(graph["12"]["inputs"]["noise_seed"], 424242)
        self.assertEqual(
            json.loads((attempt.path / "configured-workflow-api.json").read_text(encoding="utf-8")),
            graph,
        )
        self.assertEqual(
            json.loads((attempt.path / "submission-request.json").read_text(encoding="utf-8")),
            {"prompt": graph},
        )
        self.assertEqual(self.manifest.read_bytes(), manifest_before)

    def test_invalid_model_policy_stops_before_submission(self) -> None:
        source = dict(self.project.source)
        source["model_files"] = [source["models"]["high"]]
        project = ProjectConfig(path=self.manifest, source=source)
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(ValueError, "configured low model is missing"):
            render_segment(project, "S020", "S020_C001", client)

        self.assertEqual(client.submitted, [])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_retry_keeps_the_first_attempt_immutable(self, extract, contact_sheet) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(count):
                path = destination / f"frame-{index:06d}.png"
                path.write_bytes(f"{where}-{index}".encode())
                paths.append(path)
            return paths

        extract.side_effect = fake_extract
        def fake_contact_sheet(_frames: list[Path], destination: Path) -> Path:
            destination.write_bytes(b"sheet")
            return destination

        contact_sheet.side_effect = fake_contact_sheet
        client = FakeRenderClient(self.video)

        first = render_segment(self.project, "S030", "S030_C001", client)
        first_qc = (first.path / "qc.yaml").read_bytes()
        retry = render_segment(self.project, "S030", "S030_C001", client)

        self.assertNotEqual(first.path, retry.path)
        self.assertEqual((first.path / "qc.yaml").read_bytes(), first_qc)
        self.assertEqual(retry.parent_attempt, first.path)


if __name__ == "__main__":
    unittest.main()
