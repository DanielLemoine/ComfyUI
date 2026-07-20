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
from wan22_longform.metadata import RenderMetadata, write_metadata  # noqa: E402
from wan22_longform.project import (  # noqa: E402
    Attempt,
    AttemptState,
    create_attempt,
    transition_attempt,
)
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
                "seed_increment": 17,
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
        write_metadata(upstream, RenderMetadata(outputs={"segment": self.video}))
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

        with self.assertRaisesRegex(ValueError, "models.low"):
            render_segment(project, "S020", "S020_C001", client)

        self.assertEqual(client.submitted, [])

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
        write_metadata(upstream, RenderMetadata(outputs={"segment": self.video}))
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
        self.workflow = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"
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
                "seed_increment": 17,
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

    def tearDown(self) -> None:
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
        write_metadata(upstream, RenderMetadata(outputs={"segment": self.video}))
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
        write_metadata(upstream, RenderMetadata(outputs={"segment": self.video}))
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
