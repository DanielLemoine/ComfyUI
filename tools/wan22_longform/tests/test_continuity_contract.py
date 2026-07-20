from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.cli import _handle_render_shot, main  # noqa: E402
from wan22_longform.comfy_client import HistoryOutput, HistoryResult  # noqa: E402
from wan22_longform.config import (  # noqa: E402
    ConfigError,
    ProjectConfig,
    load_project,
    validate_project_contract,
)
from wan22_longform.project import (  # noqa: E402
    Attempt,
    AttemptState,
    create_attempt,
    transition_attempt,
)
from wan22_longform.qc import initialize_qc  # noqa: E402
from wan22_longform.render import RenderError, render_bridge, render_segment, validate_project  # noqa: E402


class FakeRenderClient:
    def __init__(self, video: Path) -> None:
        self.video = video
        self.uploaded: list[Path] = []
        self.submitted: list[dict[str, object]] = []

    def upload_image(self, path: Path) -> str:
        self.uploaded.append(path)
        return f"uploaded-{path.name}"

    def submit(self, graph: dict[str, object]) -> str:
        self.submitted.append(graph)
        return "continuity-prompt"

    def wait(self, prompt_id: str) -> HistoryResult:
        return HistoryResult(
            prompt_id=prompt_id,
            outputs=(
                HistoryOutput(
                    node_id="14",
                    filename="continuity.mp4",
                    subfolder="video",
                    type="output",
                    local_path=self.video,
                ),
            ),
        )

    def fetch_output(self, output: HistoryOutput, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(output.local_path.read_bytes())
        return destination


class ContinuityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "fixture.mp4"
        self.video.write_bytes(b"fixture-video")
        self.fallback = self._artifact("fallback.png", b"fallback")
        self.anchor = self._artifact("anchor.png", b"anchor")
        self.second_anchor = self._artifact("second-anchor.png", b"second-anchor")
        self.override = self._artifact("override.png", b"override")
        self.segment_graph = PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"
        self.bridge_graph = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"
        self.manifest = self.root / "project.yaml"
        self.source = self._source()
        self._write_manifest(self.source)
        self.project = ProjectConfig(path=self.manifest, source=self.source)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_continuation_uses_accepted_tail_and_records_immutable_provenance(self) -> None:
        upstream, _head, tail = self._accepted_attempt("S010", "S010_C001")
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            downstream = render_segment(self.project, "S010", "S010_C002", client)

        self.assertEqual(client.uploaded, [tail])
        provenance = json.loads((downstream.path / "provenance.json").read_text(encoding="utf-8"))
        selected = provenance["selected_inputs"]["opening_frame"]
        self.assertEqual(selected["path"], str(tail))
        self.assertEqual(selected["sha256"], hashlib.sha256(tail.read_bytes()).hexdigest())
        self.assertEqual(selected["source"], "accepted_tail")
        self.assertEqual(selected["upstream_attempt"], str(upstream.path))
        upload = json.loads((downstream.path / "input-upload.json").read_text(encoding="utf-8"))
        self.assertEqual(upload["opening_frame"]["sha256"], selected["sha256"])
        self.assertEqual(upload["opening_frame"]["source"], "accepted_tail")

    def test_explicit_segment_opening_image_precedes_shot_anchor_and_project_fallback(self) -> None:
        source = copy.deepcopy(self.source)
        source["shots"][0]["segments"][0]["opening_image"] = str(self.override)
        project = self._project(source)
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            render_segment(project, "S010", "S010_C001", client)

        self.assertEqual(client.uploaded, [self.override])

    def test_strict_seed_family_applies_each_segment_offset(self) -> None:
        source = copy.deepcopy(self.source)
        source["request"]["seed"] = 999999
        source["shots"][0]["segments"][0]["seed_offset"] = 7
        project = self._project(source)
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            render_segment(project, "S010", "S010_C001", client)

        self.assertEqual(client.submitted[0]["11"]["inputs"]["noise_seed"], 407)
        self.assertEqual(client.submitted[0]["12"]["inputs"]["noise_seed"], 407)

    def test_unsafe_continuations_reject_before_attempt_or_upload(self) -> None:
        cases = (
            ("missing", self._missing_upstream),
            ("not accepted", self._unaccepted_upstream),
            ("changed", self._changed_tail),
            ("head", self._head_selected),
            ("ambiguous", self._ambiguous_upstream),
            ("cross shot", self._cross_shot_reference),
            ("forward", self._forward_reference),
        )
        for label, prepare in cases:
            with self.subTest(label=label):
                project = self._project(copy.deepcopy(self.source))
                project.source["attempts_dir"] = str(self.root / f"attempts-{label}")
                prepare(project)
                client = FakeRenderClient(self.video)

                with self._render_artifact_patches():
                    with self.assertRaises((RenderError, ValueError)):
                        render_segment(project, "S010", "S010_C002", client)

                self.assertEqual(client.uploaded, [])
                self.assertEqual(client.submitted, [])
                self.assertFalse(
                    (self.root / f"attempts-{label}" / "S010" / "S010_C002").exists()
                )

    def test_accept_rejects_a_head_candidate_as_continuation(self) -> None:
        review, head, _tail = self._review_attempt("S010", "S010_C001")

        with self.assertRaisesRegex(ValueError, "tail"):
            main(
                [
                    "accept",
                    str(review.path),
                    "--note",
                    "head is not a continuation tail",
                    "--continuation-frame",
                    str(head),
                ]
            )

        self.assertEqual(review.state, AttemptState.NEEDS_REVIEW)
        self.assertFalse((review.path / "decisions" / "0004.json").exists())

    def test_story_bridge_requires_accepted_tail_and_head_provenance(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "first_image": str(self.anchor),
                "last_image": str(self.second_anchor),
                "from_segment": "S010_C001",
                "to_segment": "S010_C002",
                "frames": 33,
            }
        ]
        source["assembly_order"] = [
            {"shot_id": "S010", "segment_id": "S010_C001"},
            {"shot_id": "S010", "segment_id": "B010"},
            {"shot_id": "S010", "segment_id": "S010_C002"},
            {"shot_id": "S020", "segment_id": "S020_C001"},
        ]
        project = self._project(source)
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            with self.assertRaisesRegex(RenderError, "accepted.*tail.*head"):
                render_bridge(project, "B010", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_story_bridge_uses_hash_verified_accepted_tail_and_head(self) -> None:
        upstream, _head, tail = self._accepted_attempt("S010", "S010_C001")
        destination, head, _destination_tail = self._accepted_attempt("S010", "S010_C002")
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "first_image": str(tail),
                "last_image": str(head),
                "from_segment": "S010_C001",
                "to_segment": "S010_C002",
                "frames": 33,
            }
        ]
        source["assembly_order"] = [
            {"shot_id": "S010", "segment_id": "S010_C001"},
            {"shot_id": "S010", "segment_id": "B010"},
            {"shot_id": "S010", "segment_id": "S010_C002"},
            {"shot_id": "S020", "segment_id": "S020_C001"},
        ]
        project = self._project(source)
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            bridge = render_bridge(project, "B010", client)

        self.assertEqual(client.uploaded, [tail, head])
        provenance = json.loads((bridge.path / "provenance.json").read_text(encoding="utf-8"))
        selected = provenance.get("selected_inputs")
        if not isinstance(selected, dict):
            self.fail("bridge attempt must record selected endpoint provenance")
        self.assertEqual(selected["first_image"]["candidate_kind"], "tail")
        self.assertEqual(selected["last_image"]["candidate_kind"], "head")
        self.assertEqual(selected["first_image"]["upstream_attempt"], str(upstream.path))
        self.assertEqual(selected["last_image"]["upstream_attempt"], str(destination.path))

    def test_story_bridge_rejects_accepted_endpoints_from_the_wrong_declared_segments(self) -> None:
        _source, source_head, source_tail = self._accepted_attempt("S010", "S010_C001")
        _destination, destination_head, destination_tail = self._accepted_attempt(
            "S010", "S010_C002"
        )
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "first_image": str(destination_tail),
                "last_image": str(source_head),
                "from_segment": "S010_C001",
                "to_segment": "S010_C002",
                "frames": 33,
            }
        ]
        source["assembly_order"] = [
            {"shot_id": "S010", "segment_id": "S010_C001"},
            {"shot_id": "S010", "segment_id": "B010"},
            {"shot_id": "S010", "segment_id": "S010_C002"},
            {"shot_id": "S020", "segment_id": "S020_C001"},
        ]
        project = self._project(source)
        client = FakeRenderClient(self.video)

        with self._render_artifact_patches():
            with self.assertRaisesRegex(RenderError, "declared source.*destination"):
                render_bridge(project, "B010", client)

        self.assertEqual(client.uploaded, [])
        self.assertEqual(client.submitted, [])

    def test_render_shot_refuses_dependent_segments_without_partial_queueing(self) -> None:
        args = Namespace(
            project=self.manifest,
            shot_id="S010",
            comfy_url="http://127.0.0.1:8188",
            timeout=1.0,
            poll_interval=0.0,
        )
        with patch("wan22_longform.cli.load_project", return_value=self.project), patch(
            "wan22_longform.cli.render_segment"
        ) as render_segment_mock, patch("wan22_longform.cli._print_json"):
            with self.assertRaisesRegex(ValueError, "dependent continuation"):
                _handle_render_shot(args)

        render_segment_mock.assert_not_called()

    def test_strict_manifest_rejects_representative_missing_contract_groups(self) -> None:
        mutations = {
            "schema version": lambda source: source.pop("schema_version"),
            "project identity": lambda source: source.pop("title"),
            "mode": lambda source: source.__setitem__("mode", "preview"),
            "render timing": lambda source: source["render"].pop("generation_fps"),
            "core model": lambda source: source["models"].pop("vae"),
            "per-shot anchor": lambda source: (
                source["shots"][0].pop("anchor_image"),
                source["inputs"].pop("opening_frame"),
            ),
            "segment duration": lambda source: source["shots"][0]["segments"][0].pop("expected_seconds"),
            "continuation policy": lambda source: source.pop("continuation"),
            "qc retry": lambda source: source["qc"].pop("retry_limit"),
            "outputs": lambda source: source.pop("outputs"),
            "workflow hashes": lambda source: source.pop("workflow_hashes"),
            "environment snapshot": lambda source: source.pop("environment_snapshot"),
            "technical smoke marker": lambda source: source.__setitem__(
                "bridges",
                [
                    {
                        "id": "B010",
                        "shot_id": "S010",
                        "strategy": "flf2v",
                        "base_source_image": str(self.anchor),
                        "frames": 33,
                    }
                ],
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                source = copy.deepcopy(self.source)
                mutate(source)

                with self.assertRaises((RenderError, ValueError)):
                    validate_project(self._project(source))

    def test_strict_manifest_rejects_conflicting_legacy_candidate_count(self) -> None:
        source = copy.deepcopy(self.source)
        source["candidate_count"] = source["qc"]["candidate_count"] + 1

        with self.assertRaisesRegex(ConfigError, "candidate_count.*qc.candidate_count"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_requires_declared_output_roles_inside_output_root(self) -> None:
        source = copy.deepcopy(self.source)
        source["outputs"]["review_mp4"] = str(self.root / "outside-review.mp4")

        with self.assertRaisesRegex(ConfigError, "outputs.review_mp4.*output_root"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_rejects_unknown_bridge_shot(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B999",
                "shot_id": "S999",
                "strategy": "direct",
            }
        ]

        with self.assertRaisesRegex(ConfigError, "bridge B999 shot_id"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_rejects_bridge_id_collision_with_a_same_shot_segment(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "S010_C001",
                "shot_id": "S010",
                "strategy": "direct",
            }
        ]

        with self.assertRaisesRegex(ConfigError, "collides with configured segment"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_requires_story_flf_source_and_destination_segments(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "first_image": str(self.anchor),
                "last_image": str(self.second_anchor),
                "frames": 33,
            }
        ]

        with self.assertRaisesRegex(ConfigError, "from_segment"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_requires_story_flf_bridge_directly_between_declared_segments(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "first_image": str(self.anchor),
                "last_image": str(self.second_anchor),
                "from_segment": "S010_C001",
                "to_segment": "S010_C002",
                "frames": 33,
            }
        ]
        source["assembly_order"] = [
            {"shot_id": "S010", "segment_id": "S010_C001"},
            {"shot_id": "S010", "segment_id": "S010_C002"},
            {"shot_id": "S010", "segment_id": "B010"},
            {"shot_id": "S020", "segment_id": "S020_C001"},
        ]

        with self.assertRaisesRegex(ConfigError, "directly between"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_excludes_technical_smoke_bridge_from_production_order(self) -> None:
        source = copy.deepcopy(self.source)
        source["bridges"] = [
            {
                "id": "B010",
                "shot_id": "S010",
                "strategy": "flf2v",
                "purpose": "technical_smoke",
                "base_source_image": str(self.anchor),
                "frames": 33,
            }
        ]
        source["assembly_order"].append({"shot_id": "S010", "segment_id": "B010"})

        with self.assertRaisesRegex(ConfigError, "technical_smoke"):
            validate_project_contract(self._project(source))

    def test_strict_manifest_rejects_assembly_item_for_the_wrong_shot(self) -> None:
        source = copy.deepcopy(self.source)
        source["assembly_order"] = [
            {"shot_id": "S020", "segment_id": "S010_C001"},
        ]

        with self.assertRaisesRegex(ConfigError, "assembly_order.*S020.*S010_C001"):
            validate_project_contract(self._project(source))

    def test_example_timeline_excludes_technical_smoke_and_matches_declared_durations(self) -> None:
        project = load_project(PROJECT_DIR / "projects" / "example" / "project.yaml")
        source = project.source
        render = source["render"]
        fps = render["generation_fps"]
        segment_duration = render["frames"] / fps

        durations: dict[tuple[str, str], float] = {}
        technical_smoke_ids: set[str] = set()
        for shot in source["shots"]:
            shot_id = shot["id"]
            expected_shot_duration = 0.0
            for segment in shot["segments"]:
                segment_id = segment["id"]
                self.assertAlmostEqual(segment["expected_seconds"], segment_duration, delta=1 / fps)
                durations[(shot_id, segment_id)] = segment["expected_seconds"]
                expected_shot_duration += segment["expected_seconds"]
            self.assertAlmostEqual(shot["target_seconds"], expected_shot_duration, delta=1 / fps)

        for bridge in source["bridges"]:
            if bridge.get("purpose") == "technical_smoke":
                technical_smoke_ids.add(bridge["id"])
            else:
                durations[(bridge["shot_id"], bridge["id"])] = bridge["frames"] / fps

        assembly_duration = 0.0
        for item in source["assembly_order"]:
            key = (item["shot_id"], item["segment_id"])
            self.assertNotIn(item["segment_id"], technical_smoke_ids)
            assembly_duration += durations[key]
        self.assertAlmostEqual(source["target_seconds"], assembly_duration, delta=1 / fps)

    def test_example_manifest_stops_only_at_the_documented_missing_anchor(self) -> None:
        project = load_project(PROJECT_DIR / "projects" / "example" / "project.yaml")

        with self.assertRaisesRegex(RenderError, "anchor image does not exist"):
            validate_project(project)

    def _source(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "project_id": "continuity_fixture",
            "title": "Continuity fixture",
            "mode": "continuous",
            "target_seconds": 4,
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
            "model_files": [
                "configured-high.safetensors",
                "configured-low.safetensors",
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            ],
            "workflow_api": str(self.segment_graph),
            "bridge_workflow_api": str(self.bridge_graph),
            "workflow_hashes": {
                "segment_api": hashlib.sha256(self.segment_graph.read_bytes()).hexdigest(),
                "bridge_api": hashlib.sha256(self.bridge_graph.read_bytes()).hexdigest(),
            },
            "environment_snapshot": {
                "captured_at": "2026-07-19T12:00:00Z",
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
                "seed_base": 400,
                "seed_increment": 17,
            },
            "inputs": {"opening_frame": str(self.fallback)},
            "attempts_dir": str(self.root / "attempts"),
            "candidate_count": 2,
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
            "request": {
                "positive": "neutral fully clothed adult",
                "negative": "low quality",
                "seed": 400,
                "width": 640,
                "height": 640,
                "frames": 17,
            },
            "prompt_blocks": {
                "identity_lock": "same adult identity",
                "continuity_lock": "same lighting and wardrobe",
                "negative": "low quality",
            },
            "shots": [
                {
                    "id": "S010",
                    "target_seconds": 4,
                    "anchor_image": str(self.anchor),
                    "camera": "eye level",
                    "environment": "fixture room",
                    "segments": [
                        {
                            "id": "S010_C001",
                            "action": "The adult takes one natural step.",
                            "expected_seconds": 2,
                            "seed_offset": 0,
                        },
                        {
                            "id": "S010_C002",
                            "action": "The adult takes one more natural step.",
                            "expected_seconds": 2,
                            "seed_offset": 17,
                            "continue_from": "S010_C001",
                        },
                    ],
                },
                {
                    "id": "S020",
                    "target_seconds": 2,
                    "anchor_image": str(self.second_anchor),
                    "camera": "eye level",
                    "environment": "fixture room",
                    "segments": [
                        {
                            "id": "S020_C001",
                            "action": "The adult pauses naturally.",
                            "expected_seconds": 2,
                            "seed_offset": 34,
                        }
                    ],
                },
            ],
            "bridges": [],
            "assembly_order": [
                {"shot_id": "S010", "segment_id": "S010_C001"},
                {"shot_id": "S010", "segment_id": "S010_C002"},
                {"shot_id": "S020", "segment_id": "S020_C001"},
            ],
        }

    def _project(self, source: dict[str, object]) -> ProjectConfig:
        self._write_manifest(source)
        return ProjectConfig(path=self.manifest, source=source)

    def _write_manifest(self, source: dict[str, object]) -> None:
        self.manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")

    def _artifact(self, name: str, contents: bytes) -> Path:
        path = self.root / name
        path.write_bytes(contents)
        return path

    def _review_attempt(
        self, shot_id: str, segment_id: str, project: ProjectConfig | None = None
    ) -> tuple[Attempt, Path, Path]:
        attempt = create_attempt(project or self.project, shot_id, segment_id)
        for state in (AttemptState.RENDERING, AttemptState.RENDERED, AttemptState.NEEDS_REVIEW):
            attempt = transition_attempt(attempt, state, "fixture")
        head = self._artifact(f"{attempt.attempt_id}-head.png", b"head")
        tail = self._artifact(f"{attempt.attempt_id}-tail.png", b"tail")
        sheet = self._artifact(f"{attempt.attempt_id}-sheet.png", b"sheet")
        initialize_qc(
            attempt,
            video=self.video,
            head_frames=[head],
            tail_frames=[tail],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        return attempt, head, tail

    def _accepted_attempt(
        self, shot_id: str, segment_id: str, project: ProjectConfig | None = None
    ) -> tuple[Attempt, Path, Path]:
        review, head, tail = self._review_attempt(shot_id, segment_id, project)
        accepted = transition_attempt(
            review,
            AttemptState.ACCEPTED,
            "accepted fixture tail",
            selected_continuation_frame=tail,
        )
        return accepted, head, tail

    def _missing_upstream(self, _project: ProjectConfig) -> None:
        return None

    def _unaccepted_upstream(self, project: ProjectConfig) -> None:
        self._review_attempt("S010", "S010_C001", project)

    def _changed_tail(self, project: ProjectConfig) -> None:
        _accepted, _head, tail = self._accepted_attempt("S010", "S010_C001", project)
        tail.write_bytes(b"changed-tail")

    def _head_selected(self, project: ProjectConfig) -> None:
        review, head, _tail = self._review_attempt("S010", "S010_C001", project)
        transition_attempt(
            review,
            AttemptState.ACCEPTED,
            "incorrect fixture head",
            selected_continuation_frame=head,
        )

    def _ambiguous_upstream(self, project: ProjectConfig) -> None:
        self._accepted_attempt("S010", "S010_C001", project)
        self._accepted_attempt("S010", "S010_C001", project)

    def _cross_shot_reference(self, project: ProjectConfig) -> None:
        project.source["shots"][0]["segments"][1]["continue_from"] = "S020_C001"

    def _forward_reference(self, project: ProjectConfig) -> None:
        project.source["shots"][0]["segments"][1]["continue_from"] = "S010_C002"

    def _extract(self, _video: Path, count: int, where: str, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(count):
            frame = destination / f"{where}-{index:06d}.png"
            frame.write_bytes(f"{where}-{index}".encode("utf-8"))
            frames.append(frame)
        return frames

    @staticmethod
    def _sheet(_frames: list[Path], destination: Path) -> Path:
        destination.write_bytes(b"sheet")
        return destination

    def _render_artifact_patches(self):
        return _RenderArtifacts(self._extract, self._sheet)


class _RenderArtifacts:
    def __init__(self, extract, sheet) -> None:
        self.extract = extract
        self.sheet = sheet
        self.extract_patch = patch("wan22_longform.render.extract_candidate_frames", side_effect=extract)
        self.sheet_patch = patch("wan22_longform.render.create_contact_sheet", side_effect=sheet)

    def __enter__(self):
        self.extract_patch.__enter__()
        self.sheet_patch.__enter__()
        return self

    def __exit__(self, *args):
        self.sheet_patch.__exit__(*args)
        return self.extract_patch.__exit__(*args)


if __name__ == "__main__":
    unittest.main()
