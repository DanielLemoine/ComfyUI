from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from fractions import Fraction
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
from wan22_longform.ffmpeg import MediaSpec  # noqa: E402
from wan22_longform.metadata import RenderMetadata, write_metadata  # noqa: E402
from wan22_longform.project import (  # noqa: E402
    Attempt,
    AttemptState,
    accepted_attempt_evidence,
    create_attempt,
    inspect_attempt_integrity,
    load_attempt,
    transition_attempt,
)
from wan22_longform import cli  # noqa: E402
from wan22_longform.qc import initialize_qc, read_qc  # noqa: E402
from wan22_longform import render as render_module  # noqa: E402
from wan22_longform.render import (  # noqa: E402
    RenderError,
    render_bridge,
    render_segment,
    resume_attempt,
)


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
    def __init__(
        self,
        video: Path,
        *,
        fail_uploads: int = 0,
        fail_submits: int = 0,
        fail_waits: int = 0,
    ) -> None:
        self.video = video
        self.submitted: list[dict[str, dict[str, object]]] = []
        self.waited: list[str] = []
        self.uploaded: list[Path] = []
        self.fetched: list[Path] = []
        self.fail_uploads = fail_uploads
        self.fail_submits = fail_submits
        self.fail_waits = fail_waits

    def upload_image(self, path: Path) -> str:
        self.uploaded.append(path)
        if self.fail_uploads:
            self.fail_uploads -= 1
            raise TimeoutError("fixture upload failure")
        return "uploaded-opening.png"

    def submit(self, graph: dict[str, dict[str, object]]) -> str:
        self.submitted.append(graph)
        if self.fail_submits:
            self.fail_submits -= 1
            raise TimeoutError("fixture submit response failure")
        return "fixture-prompt"

    def wait(self, prompt_id: str) -> HistoryResult:
        self.waited.append(prompt_id)
        if self.fail_waits:
            self.fail_waits -= 1
            raise TimeoutError("fixture wait timeout")
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
        self.fetched.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output.local_path, destination)
        return destination


def fixture_media(path: Path, *, frame_count: int, fps: int = 16) -> MediaSpec:
    return MediaSpec(
        path=path,
        width=832,
        height=480,
        fps=Fraction(fps, 1),
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


def write_submission_fixture(attempt: Attempt, *, kind: str = "segment") -> dict[str, str]:
    def write(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    input_upload = attempt.path / "input-upload.json"
    configured = attempt.path / "configured-workflow-api.json"
    request = attempt.path / "submission-request.json"
    provenance = attempt.path / "submission-provenance.json"
    write(input_upload, {})
    configured_payload = json.loads(
        (attempt.path / "workflow-api.json").read_text(encoding="utf-8")
    )
    write(configured, configured_payload)
    write(request, {"prompt": configured_payload})
    source_manifest = next(attempt.path.glob("source-manifest.*"))
    write(
        provenance,
        {
            "version": 1,
            "input_upload": hashlib.sha256(input_upload.read_bytes()).hexdigest(),
            "request": hashlib.sha256(request.read_bytes()).hexdigest(),
            "source_manifest": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "base_workflow": hashlib.sha256(
                (attempt.path / "workflow-api.json").read_bytes()
            ).hexdigest(),
            "workflow": hashlib.sha256(configured.read_bytes()).hexdigest(),
        },
    )
    request_evidence = {
        "path": str(request.resolve()),
        "sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    provenance_evidence = {
        "path": str(provenance.resolve()),
        "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
    }
    write(
        attempt.path / "submission-intent.json",
        {
            "submission_request": request_evidence,
            "submission_provenance": provenance_evidence,
        },
    )
    write(
        attempt.path / "queue.json",
        {
            "kind": kind,
            "prompt_id": "fixture-prompt",
            "submission_request": request_evidence,
            "submission_provenance": provenance_evidence,
        },
    )
    write(attempt.path / "history.json", {"prompt_id": "fixture-prompt", "history": {}})
    return request_evidence


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
        self.model_root = self.root / "models"
        for kind, names in {
            "diffusion_models": (
                "configured-high.safetensors",
                "configured-low.safetensors",
            ),
            "vae": ("wan_2.1_vae.safetensors",),
            "text_encoders": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors",),
        }.items():
            root = self.model_root / kind
            root.mkdir(parents=True, exist_ok=True)
            for name in names:
                (root / name).write_bytes(name.encode("utf-8"))
        self.workflow = PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"
        self.object_info = self.root / "object_info.json"
        self.object_info.write_bytes(
            (PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json").read_bytes()
        )
        self.manifest = self.root / "project.yaml"
        source = {
            "schema_version": 1,
            "project_id": "render-segment-fixture",
            "title": "Render segment fixture",
            "mode": "continuous",
            "target_seconds": 20.25,
            "output_root": str(self.root / "outputs"),
            "outputs": {
                "review_mp4": str(self.root / "outputs" / "review.mp4"),
                "edit_master_ffv1": str(self.root / "outputs" / "master.mkv"),
                "edit_master_prores": str(self.root / "outputs" / "master.mov"),
            },
            "preset": "P0_IDENTITY_BASELINE",
            "models": {
                "high": "configured-high.safetensors",
                "low": "configured-low.safetensors",
                "vae": "wan_2.1_vae.safetensors",
                "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            "workflow_api": str(self.workflow),
            "bridge_workflow_api": str(self.workflow),
            "workflow_hashes": {
                "segment_api": hashlib.sha256(self.workflow.read_bytes()).hexdigest(),
                "bridge_api": hashlib.sha256(self.workflow.read_bytes()).hexdigest(),
            },
            "object_info": str(self.object_info),
            "object_info_sha256": hashlib.sha256(self.object_info.read_bytes()).hexdigest(),
            "environment_snapshot": {
                "captured_at": "2026-07-19T00:00:00Z",
                "platform": "fixture",
                "python": "3.11.6",
                "gpu": "fixture",
            },
            "render": {
                "workflow": "wan22_segment_i2v_native_api.json",
                "width": 832,
                "height": 480,
                "frames": 81,
                "generation_fps": 16,
                "review_mp4_codec": "h264",
                "master_codec": "ffv1",
                "seed_base": 424242,
            },
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
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            ],
            "model_roots": [
                str(self.model_root / "diffusion_models"),
                str(self.model_root / "vae"),
                str(self.model_root / "text_encoders"),
            ],
            "policy": {"automatic_continuation": False},
            "continuation": {"strategy": "selected_tail", "reset_limit": 3},
            "qc": {"candidate_count": 5, "retry_limit": 2},
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
                    "id": shot_id,
                    "target_seconds": 5.0625,
                    "anchor_image": str(self.opening),
                    "segments": [
                        {
                            "id": f"{shot_id}_C001",
                            "action": "A neutral adult pauses naturally.",
                            "expected_seconds": 5.0625,
                            "seed_offset": index * 17,
                        }
                    ],
                }
                for index, shot_id in enumerate(("S010", "S020", "S030", "S040"))
            ],
            "bridges": [],
            "assembly_order": [
                {"shot_id": shot_id, "segment_id": f"{shot_id}_C001"}
                for shot_id in ("S010", "S020", "S030", "S040")
            ],
        }
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        self.project = ProjectConfig(path=self.manifest, source=source)
        self._probe_media_patch = patch(
            "wan22_longform.render.probe_media",
            side_effect=lambda path: fixture_media(path, frame_count=81),
        )
        self._probe_media_patch.start()

    def _accepted_upstream(self, *, suffix: str = "") -> tuple[Attempt, Path, Path]:
        head = self.root / f"accepted-head{suffix}.png"
        tail = self.root / f"accepted-tail{suffix}.png"
        head.write_bytes(f"head{suffix}".encode())
        tail.write_bytes(f"tail{suffix}".encode())
        upstream = create_attempt(self.project, "S010", "S010_C001")
        for state in (
            AttemptState.RENDERING,
            AttemptState.RENDERED,
            AttemptState.NEEDS_REVIEW,
        ):
            upstream = transition_attempt(upstream, state, "fixture")
        request_evidence = write_submission_fixture(upstream)
        write_metadata(
            upstream,
            RenderMetadata(
                outputs={"segment": self.video},
                details={
                    "prompt_id": "fixture-prompt",
                    "submission_request": request_evidence,
                },
            ),
        )
        sheet = upstream.path / "sheet.png"
        sheet.write_bytes(b"sheet")
        initialize_qc(
            upstream,
            video=self.video,
            head_frames=[head],
            tail_frames=[tail],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        upstream = transition_attempt(
            upstream,
            AttemptState.ACCEPTED,
            "accepted fixture",
            selected_continuation_frame=tail,
        )
        return upstream, head, tail

    def tearDown(self) -> None:
        self._probe_media_patch.stop()
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
        self.assertEqual(metadata["details"]["media_timing"]["actual"]["frame_count"], 81)
        self.assertEqual(metadata["details"]["media_timing"]["actual"]["fps"], "16")
        self.assertEqual(
            metadata["outputs"]["segment"]["sha256"],
            hashlib.sha256(b"fixture-video").hexdigest(),
        )
        self.assertIn("contact_sheet", metadata["outputs"])
        self.assertEqual(len(client.submitted), 1)
        graph = client.submitted[0]
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "configured-high.safetensors")
        self.assertEqual(graph["2"]["inputs"]["unet_name"], "configured-low.safetensors")
        self.assertEqual(graph["5"]["inputs"]["clip_name"], "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        self.assertEqual(graph["6"]["inputs"]["vae_name"], "wan_2.1_vae.safetensors")
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

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_non_default_generation_fps_is_submitted_and_sealed_in_timing_evidence(
        self, extract, contact_sheet
    ) -> None:
        source = deepcopy(self.project.source)
        seconds = 81 / 24
        source["render"]["generation_fps"] = 24
        source["target_seconds"] = seconds * 4
        for shot in source["shots"]:
            shot["target_seconds"] = seconds
            shot["segments"][0]["expected_seconds"] = seconds
        project = ProjectConfig(path=self.manifest, source=source)

        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(count):
                path = destination / f"{where}-{index}.png"
                path.write_bytes(b"candidate")
                paths.append(path)
            return paths

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)

        with patch(
            "wan22_longform.render.probe_media",
            side_effect=lambda path: fixture_media(path, frame_count=81, fps=24),
        ):
            attempt = render_segment(project, "S010", "S010_C001", FakeRenderClient(self.video))

        graph = json.loads((attempt.path / "configured-workflow-api.json").read_text(encoding="utf-8"))
        metadata = json.loads((attempt.path / "render-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(graph["14"]["inputs"]["fps"], 24.0)
        self.assertEqual(metadata["details"]["media_timing"]["expected"]["fps"], "24")
        self.assertEqual(metadata["details"]["media_timing"]["actual"]["fps"], "24")

    def test_timing_drift_blocks_metadata_and_qc_before_review(self) -> None:
        with patch(
            "wan22_longform.render.probe_media",
            side_effect=lambda path: fixture_media(path, frame_count=79),
        ):
            with self.assertRaisesRegex(RenderError, "more than one frame"):
                render_segment(self.project, "S010", "S010_C001", FakeRenderClient(self.video))

        attempt = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        self.assertFalse((attempt.path / "render-metadata.json").exists())
        self.assertFalse((attempt.path / "qc.yaml").exists())

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_segment_resume_replaces_a_nonzero_partial_output_before_metadata(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(count):
                path = destination / f"{where}-{index}.png"
                path.write_bytes(b"candidate")
                paths.append(path)
            return paths

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        client = FakeRenderClient(self.video, fail_waits=1)

        with self.assertRaises(TimeoutError):
            render_segment(self.project, "S010", "S010_C001", client)

        failed = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        partial = failed.path / "outputs" / "segment.mp4"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial-output")

        resumed = resume_attempt(self.project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(partial.read_bytes(), b"fixture-video")
        self.assertEqual(len(client.fetched), 1)
        metadata = json.loads((resumed.path / "render-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["outputs"]["segment"]["sha256"],
            hashlib.sha256(b"fixture-video").hexdigest(),
        )

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_candidate_publish_interruption_is_cleaned_and_resumable(
        self, extract, contact_sheet
    ) -> None:
        extract.side_effect = lambda _video, count, where, destination: [
            self._candidate(destination, where, index) for index in range(count)
        ]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        client = FakeRenderClient(self.video)
        original_replace = render_module.os.replace

        def interrupt_sheet_publish(source: Path | str, destination: Path | str) -> None:
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.name == "contact-sheet.png"
                and destination_path.name == "contact-sheet.png"
            ):
                raise RuntimeError("fixture interruption after candidate frame publish")
            original_replace(source, destination)

        with patch(
            "wan22_longform.render.os.replace", side_effect=interrupt_sheet_publish
        ):
            with self.assertRaisesRegex(RuntimeError, "candidate frame publish"):
                render_segment(self.project, "S010", "S010_C001", client)

        failed = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        self.assertTrue((failed.path / "candidate-frames").is_dir())
        self.assertFalse((failed.path / "contact-sheet.png").exists())
        self.assertFalse(list(failed.path.glob(".candidate-stage-*")))
        self.assertFalse((failed.path / "render-metadata.json").exists())

        resumed = resume_attempt(self.project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertTrue((resumed.path / "contact-sheet.png").is_file())
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(client.waited, ["fixture-prompt", "fixture-prompt"])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_history_publish_interruption_is_resumable_without_duplicate_submission(
        self, extract, contact_sheet
    ) -> None:
        extract.side_effect = lambda _video, count, where, destination: [
            self._candidate(destination, where, index) for index in range(count)
        ]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        client = FakeRenderClient(self.video)
        original_write = render_module._write_or_verify_json

        def interrupt_after_history(path: Path, payload: object) -> Path:
            written = original_write(path, payload)
            if path.name == "history.json":
                raise RuntimeError("fixture interruption after history publication")
            return written

        with patch(
            "wan22_longform.render._write_or_verify_json",
            side_effect=interrupt_after_history,
        ):
            with self.assertRaisesRegex(RuntimeError, "history publication"):
                render_segment(self.project, "S010", "S010_C001", client)

        failed = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        self.assertEqual(failed.state, AttemptState.RENDERING)
        self.assertTrue((failed.path / "history.json").is_file())
        self.assertFalse((failed.path / "outputs" / "segment.mp4").exists())
        self.assertFalse(list(failed.path.glob(".history.json.*.staging")))

        resumed = resume_attempt(self.project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(client.waited, ["fixture-prompt", "fixture-prompt"])

    def test_invalid_model_policy_stops_before_submission(self) -> None:
        source = dict(self.project.source)
        source["model_files"] = [source["models"]["high"]]
        project = ProjectConfig(path=self.manifest, source=source)
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(ValueError, "models.low"):
            render_segment(project, "S020", "S020_C001", client)

        self.assertEqual(client.submitted, [])

    def test_missing_vae_or_text_encoder_stops_before_uploading_or_submitting(self) -> None:
        for role in ("vae", "text_encoder"):
            with self.subTest(role=role):
                source = dict(self.project.source)
                source["model_files"] = [
                    name for name in source["model_files"] if name != source["models"][role]
                ]
                client = FakeRenderClient(self.video)

                with self.assertRaisesRegex(ValueError, f"models.{role}"):
                    render_segment(
                        ProjectConfig(path=self.manifest, source=source),
                        "S020",
                        "S020_C001",
                        client,
                    )

                self.assertEqual(client.uploaded, [])
                self.assertEqual(client.submitted, [])

    def test_declared_model_absent_from_disk_stops_before_uploading_or_submitting(self) -> None:
        (self.model_root / "diffusion_models" / "configured-low.safetensors").unlink()
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(
            RenderError, "role-correct on-disk model root.*configured-low"
        ):
            render_segment(self.project, "S020", "S020_C001", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_accepted_submission_chain_tampering_blocks_status_continuation_flf_and_assembly(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(f"{where}-{index}".encode("utf-8"))
                frames.append(frame)
            return frames

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        attempt = render_segment(self.project, "S010", "S010_C001", FakeRenderClient(self.video))
        tail = Path(read_qc(attempt.path / "qc.yaml")["candidate_frames"]["tail"][0]["path"])
        accepted = transition_attempt(
            attempt,
            AttemptState.ACCEPTED,
            "approved fixture",
            selected_continuation_frame=tail,
        )
        provenance_path = accepted.path / "submission-provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        artifacts = {
            "source manifest": next(accepted.path.glob("source-manifest.*")),
            "workflow snapshot": accepted.path / "workflow-api.json",
            "request snapshot": accepted.path / "request.json",
            "input upload": accepted.path / "input-upload.json",
            "configured workflow": accepted.path / "configured-workflow-api.json",
            "submission request": accepted.path / "submission-request.json",
            "submission provenance": provenance_path,
        }
        mutations = {
            **{
                name: (path, lambda target: target.write_bytes(target.read_bytes() + b"\n"))
                for name, path in artifacts.items()
            },
            **{
                f"submission provenance {name} hash": (
                    provenance_path,
                    lambda target, key=name: target.write_text(
                        json.dumps({**provenance, key: "0" * 64}, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    ),
                )
                for name in (
                    "input_upload",
                    "request",
                    "source_manifest",
                    "base_workflow",
                    "workflow",
                )
            },
        }
        continuation_shot = {
            "segments": [{"id": "S010_C001"}, {"id": "S010_C002"}],
        }
        continuation_segment = {"id": "S010_C002"}
        for label, (path, mutate) in mutations.items():
            with self.subTest(label=label):
                original = path.read_bytes()
                mutate(path)
                status, failures = inspect_attempt_integrity(self.project, accepted)
                self.assertEqual(status, "failed")
                self.assertTrue(failures)
                with self.assertRaises(Exception):
                    accepted_attempt_evidence(self.project, accepted)
                self.assertEqual(
                    cli._attempt_payload(accepted, self.project)["state"],
                    "integrity_failed",
                )
                with self.assertRaises(RenderError):
                    render_module._continuation_selection(
                        self.project,
                        "S010",
                        continuation_shot,
                        continuation_segment,
                        "S010_C001",
                    )
                with self.assertRaises(RenderError):
                    render_module._accepted_bridge_endpoint(self.project, tail, "tail")
                with self.assertRaises(Exception):
                    cli._accepted_project_attempts(self.project)
                path.write_bytes(original)

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_render_uses_qc_candidate_count_for_frame_extraction_and_qc(
        self, extract, contact_sheet
    ) -> None:
        source = dict(self.project.source)
        source["qc"] = {"candidate_count": 2, "retry_limit": 2}
        project = ProjectConfig(path=self.manifest, source=source)

        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(b"frame")
                frames.append(frame)
            return frames

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)

        attempt = render_segment(project, "S010", "S010_C001", FakeRenderClient(self.video))

        self.assertEqual([call.args[1] for call in extract.call_args_list], [2, 2])
        qc = read_qc(attempt.path / "qc.yaml")
        self.assertEqual(len(qc["candidate_frames"]["head"]), 2)
        self.assertEqual(len(qc["candidate_frames"]["tail"]), 2)

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

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_renders_the_existing_planned_attempt_without_creating_another(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(b"frame")
                frames.append(frame)
            return frames

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        planned = create_attempt(
            self.project,
            "S040",
            "S040_C001",
            selected_inputs={
                "opening_frame": {
                    "path": str(self.opening),
                    "sha256": hashlib.sha256(self.opening.read_bytes()).hexdigest(),
                    "source": "project_fallback",
                }
            },
        )
        client = FakeRenderClient(self.video)

        resumed = resume_attempt(self.project, planned, client)

        self.assertEqual(resumed.path, planned.path)
        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(len(list(planned.path.parent.iterdir())), 1)

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_segment_recovery_keeps_prepared_evidence_and_never_duplicates_submission(
        self, extract, contact_sheet
    ) -> None:
        extract.side_effect = lambda _video, count, where, destination: [
            self._candidate(destination, where, index) for index in range(count)
        ]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        cases = (
            ("upload", {"fail_uploads": 1}),
            ("submit", {"fail_submits": 1}),
            ("wait", {"fail_waits": 1}),
        )
        for stage, failures in cases:
            with self.subTest(stage=stage):
                client = FakeRenderClient(self.video, **failures)
                with self.assertRaises(TimeoutError):
                    render_segment(self.project, "S010", "S010_C001", client)
                attempt_path = sorted(
                    (self.root / "attempts" / "S010" / "S010_C001").iterdir()
                )[-1]
                failed = load_attempt(attempt_path)
                self.assertTrue((failed.path / "input-upload.json").is_file())
                if stage == "upload":
                    resumed = resume_attempt(self.project, failed, client)
                    self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
                    self.assertEqual(len(client.submitted), 1)
                elif stage == "submit":
                    self.assertTrue((failed.path / "submission-intent.json").is_file())
                    with self.assertRaisesRegex(RenderError, "unknown submit outcome"):
                        resume_attempt(self.project, failed, client)
                    self.assertEqual(len(client.submitted), 1)
                else:
                    self.assertTrue((failed.path / "queue.json").is_file())
                    resumed = resume_attempt(self.project, failed, client)
                    self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
                    self.assertEqual(len(client.submitted), 1)
                    self.assertEqual(client.waited, ["fixture-prompt", "fixture-prompt"])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_finishes_after_metadata_write_interruption_without_repolling(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(f"{where}-{index}".encode("utf-8"))
                frames.append(frame)
            return frames

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        client = FakeRenderClient(self.video)
        original_transition = render_module.transition_attempt

        def interrupt_after_metadata(attempt, target, note, **kwargs):
            if target is AttemptState.RENDERED:
                raise RuntimeError("fixture interruption after metadata")
            return original_transition(attempt, target, note, **kwargs)

        with patch(
            "wan22_longform.render.transition_attempt",
            side_effect=interrupt_after_metadata,
        ):
            with self.assertRaisesRegex(RuntimeError, "after metadata"):
                render_segment(self.project, "S010", "S010_C001", client)

        failed = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        self.assertEqual(failed.state, AttemptState.RENDERING)
        self.assertTrue((failed.path / "render-metadata.json").is_file())
        self.assertTrue(cli._is_safely_resumable_attempt(self.project, failed))

        resumed = resume_attempt(self.project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(client.waited, ["fixture-prompt"])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_finishes_after_qc_write_interruption_without_repolling(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(f"{where}-{index}".encode("utf-8"))
                frames.append(frame)
            return frames

        extract.side_effect = fake_extract
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(destination)
        client = FakeRenderClient(self.video)
        original_transition = render_module.transition_attempt

        def interrupt_after_qc(attempt, target, note, **kwargs):
            if target is AttemptState.NEEDS_REVIEW:
                raise RuntimeError("fixture interruption after QC")
            return original_transition(attempt, target, note, **kwargs)

        with patch(
            "wan22_longform.render.transition_attempt",
            side_effect=interrupt_after_qc,
        ):
            with self.assertRaisesRegex(RuntimeError, "after QC"):
                render_segment(self.project, "S010", "S010_C001", client)

        failed = load_attempt(
            next((self.root / "attempts" / "S010" / "S010_C001").iterdir())
        )
        self.assertEqual(failed.state, AttemptState.RENDERED)
        self.assertTrue((failed.path / "qc.yaml").is_file())
        self.assertTrue(cli._is_safely_resumable_attempt(self.project, failed))

        resumed = resume_attempt(self.project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(client.waited, ["fixture-prompt"])

    def test_resume_rejects_a_planned_attempt_from_another_project(self) -> None:
        planned = create_attempt(
            self.project,
            "S040",
            "S040_C001",
            selected_inputs={
                "opening_frame": {
                    "path": str(self.opening),
                    "sha256": hashlib.sha256(self.opening.read_bytes()).hexdigest(),
                    "source": "project_fallback",
                }
            },
        )
        other_manifest = self.root / "other-project.yaml"
        other_source = dict(self.project.source)
        other_source["project_id"] = "other-project"
        other_manifest.write_text(yaml.safe_dump(other_source, sort_keys=True), encoding="utf-8")
        other_project = ProjectConfig(path=other_manifest, source=other_source)

        with self.assertRaisesRegex(RuntimeError, "project lineage"):
            resume_attempt(other_project, planned, FakeRenderClient(self.video))

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_rehashes_all_planned_snapshot_evidence_before_upload(
        self, extract, contact_sheet
    ) -> None:
        extract.return_value = [self.opening]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(
            destination
        )
        mutations = {
            "source manifest": lambda attempt: next(
                attempt.path.glob("source-manifest.*")
            ).write_bytes(
                next(attempt.path.glob("source-manifest.*")).read_bytes() + b"\n"
            ),
            "workflow": lambda attempt: (attempt.path / "workflow-api.json").write_bytes(
                (attempt.path / "workflow-api.json").read_bytes() + b"\n"
            ),
            "request": lambda attempt: (attempt.path / "request.json").write_text(
                json.dumps({"tampered": True}), encoding="utf-8"
            ),
            "provenance": lambda attempt: (attempt.path / "provenance.json").write_bytes(
                (attempt.path / "provenance.json").read_bytes() + b"\n"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                planned = create_attempt(
                    self.project,
                    "S040",
                    "S040_C001",
                    selected_inputs={
                        "opening_frame": {
                            "path": str(self.opening),
                            "sha256": hashlib.sha256(
                                self.opening.read_bytes()
                            ).hexdigest(),
                            "source": "project_fallback",
                        }
                    },
                )
                mutate(planned)
                client = FakeRenderClient(self.video)

                with self.assertRaisesRegex(RenderError, "planned attempt"):
                    resume_attempt(self.project, planned, client)

                self.assertEqual(client.uploaded, [])
                self.assertEqual(client.submitted, [])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_revalidates_upstream_accepted_evidence_before_upload(
        self, extract, contact_sheet
    ) -> None:
        extract.return_value = [self.opening]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(
            destination
        )
        mutations = {
            "provenance": lambda upstream: (
                upstream.path / "provenance.json"
            ).write_bytes(
                (upstream.path / "provenance.json").read_bytes() + b"\n"
            ),
            "lineage": lambda upstream: (
                upstream.path / "attempt.json"
            ).write_text(
                json.dumps(
                    {
                        **json.loads(
                            (upstream.path / "attempt.json").read_text(encoding="utf-8")
                        ),
                        "lineage": {
                            "project_id": "tampered-project",
                            "source_manifest_sha256": "0" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            ),
        }
        for index, (label, mutate) in enumerate(mutations.items()):
            with self.subTest(label=label):
                upstream, _head, tail = self._accepted_upstream(suffix=f"-{index}")
                planned = create_attempt(
                    self.project,
                    "S040",
                    "S040_C001",
                    selected_inputs={
                        "opening_frame": {
                            "path": str(tail),
                            "sha256": hashlib.sha256(tail.read_bytes()).hexdigest(),
                            "source": "accepted_tail",
                            "upstream_attempt": str(upstream.path),
                            "candidate_kind": "tail",
                        }
                    },
                )
                mutate(upstream)
                client = FakeRenderClient(self.video)

                with self.assertRaisesRegex(RenderError, "accepted"):
                    resume_attempt(self.project, planned, client)

                self.assertEqual(client.uploaded, [])
                self.assertEqual(client.submitted, [])

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_resume_revalidates_exact_upstream_candidate_kind_before_upload(
        self, extract, contact_sheet
    ) -> None:
        extract.return_value = [self.opening]
        contact_sheet.side_effect = lambda _frames, destination: self._write_sheet(
            destination
        )
        upstream, head, _tail = self._accepted_upstream(suffix="-kind")
        planned = create_attempt(
            self.project,
            "S040",
            "S040_C001",
            selected_inputs={
                "opening_frame": {
                    "path": str(head),
                    "sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
                    "source": "accepted_qc_candidate",
                    "upstream_attempt": str(upstream.path),
                    "candidate_kind": "tail",
                }
            },
        )
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(RenderError, "accepted"):
            resume_attempt(self.project, planned, client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_continuation_rejects_changed_accepted_attempt_evidence(self) -> None:
        upstream = create_attempt(self.project, "S010", "S010_C001")
        for state in (
            AttemptState.RENDERING,
            AttemptState.RENDERED,
            AttemptState.NEEDS_REVIEW,
        ):
            upstream = transition_attempt(upstream, state, "fixture")
        request_evidence = write_submission_fixture(upstream)
        write_metadata(
            upstream,
            RenderMetadata(
                outputs={"segment": self.video},
                details={
                    "prompt_id": "fixture-prompt",
                    "submission_request": request_evidence,
                },
            ),
        )
        sheet = upstream.path / "sheet.png"
        sheet.write_bytes(b"sheet")
        initialize_qc(
            upstream,
            video=self.video,
            head_frames=[self.opening],
            tail_frames=[self.opening],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        upstream = transition_attempt(
            upstream,
            AttemptState.ACCEPTED,
            "accepted fixture",
            selected_continuation_frame=self.opening,
        )
        provenance = upstream.path / "provenance.json"
        provenance.write_bytes(provenance.read_bytes() + b"\n")
        shot = {
            "segments": [
                {"id": "S010_C001"},
                {"id": "S010_C002", "continue_from": "S010_C001"},
            ]
        }

        with self.assertRaisesRegex(RenderError, "immutable artifact evidence"):
            render_module._continuation_selection(
                self.project,
                "S010",
                shot,
                shot["segments"][1],
                "S010_C001",
            )

    @staticmethod
    def _candidate(destination: Path, where: str, index: int) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{where}-{index}.png"
        path.write_bytes(b"candidate")
        return path

    @staticmethod
    def _write_sheet(destination: Path) -> Path:
        destination.write_bytes(b"sheet")
        return destination


class RenderBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "history-video.mp4"
        self.video.write_bytes(b"fixture-video")
        self.first = self.root / "first.png"
        self.first.write_bytes(b"first-frame")
        self.last = self.root / "last.png"
        self.last.write_bytes(b"last-frame")
        self.model_root = self.root / "models"
        for kind, names in {
            "diffusion_models": (
                "configured-high.safetensors",
                "configured-low.safetensors",
            ),
            "vae": ("wan_2.1_vae.safetensors",),
            "text_encoders": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors",),
        }.items():
            root = self.model_root / kind
            root.mkdir(parents=True, exist_ok=True)
            for name in names:
                (root / name).write_bytes(name.encode("utf-8"))
        self.workflow = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"
        self.object_info = self.root / "object_info.json"
        self.object_info.write_bytes(
            (PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json").read_bytes()
        )
        self.manifest = self.root / "project.yaml"
        self.source = {
            "schema_version": 1,
            "project_id": "render-bridge-fixture",
            "title": "Render bridge fixture",
            "mode": "continuous",
            "target_seconds": 1.0625,
            "output_root": str(self.root / "outputs"),
            "outputs": {
                "review_mp4": str(self.root / "outputs" / "review.mp4"),
                "edit_master_ffv1": str(self.root / "outputs" / "master.mkv"),
                "edit_master_prores": str(self.root / "outputs" / "master.mov"),
            },
            "preset": "P0_IDENTITY_BASELINE",
            "models": {
                "high": "configured-high.safetensors",
                "low": "configured-low.safetensors",
                "vae": "wan_2.1_vae.safetensors",
                "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            "workflow_api": str(PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"),
            "bridge_workflow_api": str(self.workflow),
            "workflow_hashes": {
                "segment_api": hashlib.sha256(
                    (PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json").read_bytes()
                ).hexdigest(),
                "bridge_api": hashlib.sha256(self.workflow.read_bytes()).hexdigest(),
            },
            "object_info": str(self.object_info),
            "object_info_sha256": hashlib.sha256(self.object_info.read_bytes()).hexdigest(),
            "environment_snapshot": {
                "captured_at": "2026-07-19T00:00:00Z",
                "platform": "fixture",
                "python": "3.11.6",
                "gpu": "fixture",
            },
            "render": {
                "workflow": "wan22_segment_i2v_native_api.json",
                "width": 832,
                "height": 480,
                "frames": 17,
                "generation_fps": 16,
                "review_mp4_codec": "h264",
                "master_codec": "ffv1",
                "seed_base": 424242,
            },
            "request": {
                "positive": "configured positive prompt",
                "negative": "configured negative prompt",
                "seed": 424242,
                "width": 832,
                "height": 480,
            },
            "inputs": {
                "opening_frame": str(self.first),
                "bridge_first": str(self.first),
                "bridge_last": str(self.last),
            },
            "bridges": [
                {
                    "id": "B010",
                    "shot_id": "S010",
                    "strategy": "flf2v",
                    "purpose": "technical_smoke",
                    "first_image": str(self.first),
                    "last_image": str(self.last),
                    "frames": 33,
                }
            ],
            "attempts_dir": str(self.root / "attempts"),
            "model_files": [
                "configured-high.safetensors",
                "configured-low.safetensors",
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            ],
            "model_roots": [
                str(self.model_root / "diffusion_models"),
                str(self.model_root / "vae"),
                str(self.model_root / "text_encoders"),
            ],
            "policy": {"automatic_continuation": False},
            "continuation": {"strategy": "selected_tail", "reset_limit": 3},
            "qc": {"candidate_count": 2, "retry_limit": 2},
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
                    "anchor_image": str(self.first),
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
            "assembly_order": [{"shot_id": "S010", "segment_id": "S010_C001"}],
        }
        self.manifest.write_text(yaml.safe_dump(self.source, sort_keys=True), encoding="utf-8")
        self.project = ProjectConfig(path=self.manifest, source=self.source)
        self._probe_media_patch = patch(
            "wan22_longform.render.probe_media",
            side_effect=lambda path: fixture_media(path, frame_count=33),
        )
        self._probe_media_patch.start()

    def tearDown(self) -> None:
        self._probe_media_patch.stop()
        self.temporary_directory.cleanup()

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_bridge_uploads_both_explicit_endpoints_and_patches_native_flf_graph(
        self, extract, contact_sheet
    ) -> None:
        def fake_extract(
            _video: Path, count: int, where: str, destination: Path
        ) -> list[Path]:
            destination.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in range(count):
                frame = destination / f"{where}-{index}.png"
                frame.write_bytes(f"{where}-{index}".encode("utf-8"))
                frames.append(frame)
            return frames

        def fake_contact_sheet(_frames: list[Path], destination: Path) -> Path:
            destination.write_bytes(b"contact-sheet")
            return destination

        class EndpointClient(FakeRenderClient):
            def upload_image(self, path: Path) -> str:
                self.uploaded.append(path)
                return f"uploaded-{path.name}"

        extract.side_effect = fake_extract
        contact_sheet.side_effect = fake_contact_sheet
        client = EndpointClient(self.video)
        upstream = create_attempt(self.project, "S010", "S010_C001")
        for state in (AttemptState.RENDERING, AttemptState.RENDERED, AttemptState.NEEDS_REVIEW):
            upstream = transition_attempt(upstream, state, "fixture")
        sheet = self.root / "accepted-sheet.png"
        sheet.write_bytes(b"sheet")
        initialize_qc(
            upstream,
            video=self.video,
            head_frames=[self.last],
            tail_frames=[self.first],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        request_evidence = write_submission_fixture(upstream)
        write_metadata(
            upstream,
            RenderMetadata(
                outputs={"segment": self.video},
                details={
                    "prompt_id": "fixture-prompt",
                    "submission_request": request_evidence,
                },
            ),
        )
        transition_attempt(
            upstream,
            AttemptState.ACCEPTED,
            "accepted endpoint fixture",
            selected_continuation_frame=self.first,
        )

        attempt = render_bridge(self.project, "B010", client)

        self.assertEqual(attempt.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(client.uploaded, [self.first, self.last])
        self.assertEqual(len(client.submitted), 1)
        graph = client.submitted[0]
        self.assertEqual(graph["9"]["inputs"]["image"], "uploaded-first.png")
        self.assertEqual(graph["10"]["inputs"]["image"], "uploaded-last.png")
        self.assertEqual(graph["11"]["inputs"]["width"], 832)
        self.assertEqual(graph["11"]["inputs"]["height"], 480)
        self.assertEqual(graph["11"]["inputs"]["length"], 33)
        self.assertEqual(graph["12"]["inputs"]["noise_seed"], 424242)
        self.assertEqual(graph["13"]["inputs"]["noise_seed"], 424242)
        uploaded = json.loads((attempt.path / "input-upload.json").read_text(encoding="utf-8"))
        self.assertEqual(uploaded["first_image"]["sha256"], hashlib.sha256(b"first-frame").hexdigest())
        self.assertEqual(uploaded["last_image"]["sha256"], hashlib.sha256(b"last-frame").hexdigest())

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_bridge_resume_replaces_a_nonzero_partial_output_before_metadata(
        self, extract, contact_sheet
    ) -> None:
        extract.side_effect = lambda _video, count, where, destination: [
            self._candidate(destination, where, index) for index in range(count)
        ]
        contact_sheet.side_effect = lambda _frames, destination: self._sheet(destination)
        source = dict(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "purpose": "technical_smoke",
                "base_source_image": str(self.first),
                "frames": 33,
            }
        ]
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        project = ProjectConfig(path=self.manifest, source=source)
        client = FakeRenderClient(self.video, fail_waits=1)

        with self.assertRaises(TimeoutError):
            render_bridge(project, "B010", client)

        failed = load_attempt(next((self.root / "attempts" / "S010" / "B010").iterdir()))
        partial = failed.path / "outputs" / "bridge.mp4"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial-output")

        resumed = resume_attempt(project, failed, client)

        self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
        self.assertEqual(partial.read_bytes(), b"fixture-video")
        self.assertEqual(len(client.fetched), 1)
        metadata = json.loads((resumed.path / "render-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["outputs"]["bridge"]["sha256"],
            hashlib.sha256(b"fixture-video").hexdigest(),
        )

    @patch("wan22_longform.render.create_contact_sheet")
    @patch("wan22_longform.render.extract_candidate_frames")
    def test_bridge_recovery_keeps_prepared_evidence_and_never_duplicates_submission(
        self, extract, contact_sheet
    ) -> None:
        extract.side_effect = lambda _video, count, where, destination: [
            self._candidate(destination, where, index) for index in range(count)
        ]
        contact_sheet.side_effect = lambda _frames, destination: self._sheet(destination)
        source = dict(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "purpose": "technical_smoke",
                "base_source_image": str(self.first),
                "frames": 33,
            }
        ]
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        project = ProjectConfig(path=self.manifest, source=source)
        cases = (
            ("upload", {"fail_uploads": 1}),
            ("submit", {"fail_submits": 1}),
            ("wait", {"fail_waits": 1}),
        )
        for stage, failures in cases:
            with self.subTest(stage=stage):
                client = FakeRenderClient(self.video, **failures)
                with self.assertRaises(TimeoutError):
                    render_bridge(project, "B010", client)
                attempt_path = sorted(
                    (self.root / "attempts" / "S010" / "B010").iterdir()
                )[-1]
                failed = load_attempt(attempt_path)
                self.assertTrue((failed.path / "input-upload.json").is_file())
                if stage == "upload":
                    resumed = resume_attempt(project, failed, client)
                    self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
                    self.assertEqual(len(client.submitted), 1)
                elif stage == "submit":
                    self.assertTrue((failed.path / "submission-intent.json").is_file())
                    with self.assertRaisesRegex(RenderError, "unknown submit outcome"):
                        resume_attempt(project, failed, client)
                    self.assertEqual(len(client.submitted), 1)
                else:
                    self.assertTrue((failed.path / "queue.json").is_file())
                    resumed = resume_attempt(project, failed, client)
                    self.assertEqual(resumed.state, AttemptState.NEEDS_REVIEW)
                    self.assertEqual(len(client.submitted), 1)
                    self.assertEqual(client.waited, ["fixture-prompt", "fixture-prompt"])

    def test_bridge_rejects_changed_accepted_endpoint_evidence(self) -> None:
        upstream = create_attempt(self.project, "S010", "S010_C001")
        for state in (
            AttemptState.RENDERING,
            AttemptState.RENDERED,
            AttemptState.NEEDS_REVIEW,
        ):
            upstream = transition_attempt(upstream, state, "fixture")
        sheet = self.root / "tamper-sheet.png"
        sheet.write_bytes(b"sheet")
        initialize_qc(
            upstream,
            video=self.video,
            head_frames=[self.last],
            tail_frames=[self.first],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        request_evidence = write_submission_fixture(upstream)
        write_metadata(
            upstream,
            RenderMetadata(
                outputs={"segment": self.video},
                details={
                    "prompt_id": "fixture-prompt",
                    "submission_request": request_evidence,
                },
            ),
        )
        transition_attempt(
            upstream,
            AttemptState.ACCEPTED,
            "accepted endpoint fixture",
            selected_continuation_frame=self.first,
        )
        metadata = upstream.path / "render-metadata.json"
        metadata.write_bytes(metadata.read_bytes() + b"\n")

        with self.assertRaisesRegex(RenderError, "accepted evidence is invalid"):
            render_module._accepted_bridge_endpoint(self.project, self.first, "tail")

    def test_bridge_rejects_an_illegal_length_before_uploading_or_submitting(self) -> None:
        source = dict(self.source)
        source["bridges"] = [dict(self.source["bridges"][0], frames=18)]
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(RuntimeError, "17, 33, 49, 65, or 81"):
            render_bridge(ProjectConfig(path=self.manifest, source=source), "B010", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_external_control_policy_is_not_submitted_as_a_bridge_clip(self) -> None:
        source = dict(self.source)
        source["bridges"] = [
            {
                "id": "X010",
                "shot_id": "S010",
                "strategy": "external_control",
                "from_segment": "S010_C001",
                "to_segment": "S010_C001_NEXT",
            }
        ]
        source["shots"] = [
            {
                **source["shots"][0],
                "segments": [
                    *source["shots"][0]["segments"],
                    {
                        "id": "S010_C001_NEXT",
                        "action": "The neutral adult remains still.",
                        "expected_seconds": 1.0625,
                        "seed_offset": 1,
                    },
                ],
            }
        ]
        source["assembly_order"] = [
            {"shot_id": "S010", "segment_id": "S010_C001"},
            {"shot_id": "S010", "segment_id": "S010_C001_NEXT"},
        ]
        source["target_seconds"] = 2.125
        source["shots"][0]["target_seconds"] = 2.125
        client = FakeRenderClient(self.video)

        with self.assertRaisesRegex(RuntimeError, "only handles bridges with strategy flf2v"):
            render_bridge(ProjectConfig(path=self.manifest, source=source), "X010", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_bridge_base_source_image_requires_a_technical_smoke_purpose(self) -> None:
        source = dict(self.source)
        source["bridges"] = [
            {
                "id": "B011",
                "shot_id": "S010",
                "strategy": "flf2v",
                "base_source_image": str(self.first),
                "frames": 33,
            }
        ]
        client = FakeRenderClient(self.video)

        with patch("wan22_longform.render.create_contact_sheet") as contact_sheet, patch(
            "wan22_longform.render.extract_candidate_frames"
        ) as extract:
            extract.side_effect = lambda _video, count, where, destination: [
                self._candidate(destination, where, index) for index in range(count)
            ]
            contact_sheet.side_effect = lambda _frames, destination: self._sheet(destination)
            with self.assertRaisesRegex(ValueError, "purpose technical_smoke"):
                render_bridge(ProjectConfig(path=self.manifest, source=source), "B011", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_resume_rejects_an_unsafe_bridge_snapshot_before_submission(self) -> None:
        source = dict(self.source)
        source["workflow_api"] = str(self.workflow)
        source["bridges"] = [
            {
                "id": "B011",
                "shot_id": "S010",
                "strategy": "flf2v",
                "base_source_image": str(self.first),
                "frames": 33,
            }
        ]
        unsafe_project = ProjectConfig(path=self.manifest, source=source)
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        planned = create_attempt(
            unsafe_project,
            "S010",
            "B011",
            selected_inputs={
                "first_image": {
                    "path": str(self.first),
                    "sha256": hashlib.sha256(self.first.read_bytes()).hexdigest(),
                    "source": "technical_smoke_base",
                },
                "last_image": {
                    "path": str(self.first),
                    "sha256": hashlib.sha256(self.first.read_bytes()).hexdigest(),
                    "source": "technical_smoke_base",
                },
            },
        )
        client = FakeRenderClient(self.video)

        with patch("wan22_longform.render.render_bridge") as render_bridge_mock:
            with self.assertRaisesRegex(ValueError, "purpose technical_smoke"):
                resume_attempt(unsafe_project, planned, client)

        render_bridge_mock.assert_not_called()

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    @staticmethod
    def _candidate(destination: Path, where: str, index: int) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{where}-{index}.png"
        path.write_bytes(b"candidate")
        return path

    @staticmethod
    def _sheet(destination: Path) -> Path:
        destination.write_bytes(b"sheet")
        return destination


if __name__ == "__main__":
    unittest.main()
