from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform import cli  # noqa: E402
from wan22_longform.assembly import (  # noqa: E402
    AssemblyExecution,
    BoundaryDecision,
)
from wan22_longform.ffmpeg import MediaSpec  # noqa: E402
from wan22_longform.metadata import RenderMetadata, write_metadata  # noqa: E402
from wan22_longform.project import (  # noqa: E402
    AssemblyState,
    AttemptState,
    assembly_records,
    create_assembly_record,
    load_assembly_record,
    create_attempt,
    load_attempt,
    transition_attempt,
)
from wan22_longform.config import ConfigError, ProjectConfig, load_project  # noqa: E402


class AssemblyRecordCliTests(unittest.TestCase):
    def test_assemble_project_creates_a_final_record_without_mutating_accepted_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001", "S010_C002"))
            project = load_project(manifest)
            first = self._accepted_attempt(project, "S010_C001")
            second = self._accepted_attempt(project, "S010_C002")
            stdout = io.StringIO()

            with self._assembly_execution_patch(), contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    cli.main(
                        [
                            "assemble-project",
                            str(manifest),
                            "--output-dir",
                            str(root / "deliverables"),
                        ]
                    ),
                    0,
                )

            records = assembly_records(project)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].state, AssemblyState.FINAL)
            self.assertEqual(load_attempt(first.path).state, AttemptState.ACCEPTED)
            self.assertEqual(load_attempt(second.path).state, AttemptState.ACCEPTED)
            record = json.loads((records[0].path / "assembly.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [item["segment_id"] for item in record["inputs"]],
                ["S010_C001", "S010_C002"],
            )
            self.assertTrue((records[0].path / "assembly-plan.json").is_file())
            self.assertTrue((records[0].path / "outputs.json").is_file())

    def test_failed_assembly_is_preserved_and_a_new_request_gets_a_new_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            project = load_project(manifest)
            source = self._accepted_attempt(project, "S010_C001")

            def fail_after_partial_output(plan, *, decision_log):
                decision_log.parent.mkdir(parents=True, exist_ok=True)
                decision_log.write_text('{"decisions": []}\n', encoding="utf-8")
                review = next(
                    operation.output for operation in plan.operations if operation.kind == "review_mp4"
                )
                review.parent.mkdir(parents=True, exist_ok=True)
                review.write_bytes(b"partial review")
                raise RuntimeError("fixture FFmpeg failure")

            with self._patched_assembly_execution(fail_after_partial_output):
                with self.assertRaisesRegex(RuntimeError, "fixture FFmpeg failure"):
                    cli.main(
                        [
                            "assemble-project",
                            str(manifest),
                            "--output-dir",
                            str(root / "failed-deliverables"),
                        ]
                    )

            failed_records = assembly_records(project)
            self.assertEqual(len(failed_records), 1)
            self.assertEqual(failed_records[0].state, AssemblyState.FAILED)
            self.assertTrue((root / "failed-deliverables" / "project-review.mp4").is_file())
            self.assertEqual(load_attempt(source.path).state, AttemptState.ACCEPTED)

            with self._assembly_execution_patch(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "assemble-project",
                            str(manifest),
                            "--output-dir",
                            str(root / "recovery-deliverables"),
                        ]
                    ),
                    0,
                )

            records = assembly_records(project)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].state, AssemblyState.FAILED)
            self.assertEqual(records[1].state, AssemblyState.FINAL)
            self.assertNotEqual(records[0].path, records[1].path)

    def test_status_and_resume_report_incomplete_assembly_records_without_rerunning_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            project = load_project(manifest)
            source = self._accepted_attempt(project, "S010_C001")
            record = create_assembly_record(
                project,
                "project",
                inputs=(cli._accepted_assembly_input(source)[1],),
                requested={"scope": "project", "targets": {}},
            )

            status_stdout = io.StringIO()
            with contextlib.redirect_stdout(status_stdout):
                self.assertEqual(cli.main(["status", str(manifest)]), 0)
            status = json.loads(status_stdout.getvalue())
            self.assertEqual(status["assembly_records"][0]["assembly_id"], record.assembly_id)
            self.assertEqual(status["assembly_records"][0]["state"], "planned")

            resume_stdout = io.StringIO()
            with contextlib.redirect_stdout(resume_stdout):
                self.assertEqual(cli.main(["resume", str(manifest)]), 0)
            resumed = json.loads(resume_stdout.getvalue())
            self.assertEqual(resumed["resumed"], [])
            self.assertEqual(resumed["incomplete_assembly_records"][0]["state"], "planned")
            self.assertEqual(load_assembly_record(record.path).state, AssemblyState.PLANNED)
            self.assertEqual(load_attempt(source.path).state, AttemptState.ACCEPTED)

    def test_direct_assemble_requires_an_explicit_diagnostic_acknowledgement(self) -> None:
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            cli.build_parser().parse_args(
                [
                    "assemble",
                    "--input",
                    "fixture.mp4",
                    "--review-mp4",
                    "review.mp4",
                    "--edit-master-ffv1",
                    "master.mkv",
                    "--edit-master-prores",
                    "master.mov",
                    "--decision-log",
                    "boundaries.json",
                ]
            )

        self.assertEqual(raised.exception.code, 2)

    def test_assemble_handlers_reject_an_invalid_production_order_before_source_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            source["bridges"] = [
                {
                    "id": "B010",
                    "shot_id": "S010",
                    "strategy": "flf2v",
                    "purpose": "technical_smoke",
                    "base_source_image": source["inputs"]["opening_frame"],
                    "frames": 33,
                }
            ]
            source["assembly_order"].append({"shot_id": "S010", "segment_id": "B010"})
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")

            with patch("wan22_longform.cli._accepted_project_attempts") as project_select:
                with self.assertRaisesRegex(ConfigError, "technical_smoke"):
                    cli.main(["assemble-project", str(manifest)])
            project_select.assert_not_called()

            with patch("wan22_longform.cli._accepted_shot_attempts") as shot_select:
                with self.assertRaisesRegex(ConfigError, "technical_smoke"):
                    cli.main(["assemble-shot", str(manifest), "S010"])
            shot_select.assert_not_called()

    def test_assemble_project_rejects_a_changed_accepted_output_before_creating_a_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            project = load_project(manifest)
            source = self._accepted_attempt(project, "S010_C001")
            video = source.path / "outputs" / "segment.mp4"
            video.write_bytes(b"changed after acceptance")

            with patch("wan22_longform.cli.execute_assembly_plan") as execute:
                with self.assertRaisesRegex(ValueError, "video changed or is missing"):
                    cli.main(["assemble-project", str(manifest)])

            execute.assert_not_called()
            self.assertEqual(assembly_records(project), ())

    def test_outcome_handlers_record_each_review_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            project = load_project(manifest)
            expected = {
                "accept": AttemptState.ACCEPTED,
                "reject": AttemptState.REJECTED,
                "retry": AttemptState.RETRY_REQUESTED,
            }
            for command, state in expected.items():
                with self.subTest(command=command):
                    review = create_attempt(project, "S010", f"S010_C001_{command}")
                    for transition in (
                        AttemptState.RENDERING,
                        AttemptState.RENDERED,
                        AttemptState.NEEDS_REVIEW,
                    ):
                        review = transition_attempt(review, transition, "fixture")
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        self.assertEqual(
                            cli.main(
                                [command, str(review.path), "--note", f"{command} fixture"]
                            ),
                            0,
                        )
                    self.assertEqual(load_attempt(review.path).state, state)
                    self.assertEqual(json.loads(stdout.getvalue())["state"], state)

    def test_resume_handler_selects_existing_planned_segment_and_bridge_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, ("S010_C001",))
            source = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            source["bridges"] = [
                {
                    "id": "B010",
                    "shot_id": "S010",
                    "strategy": "flf2v",
                    "purpose": "technical_smoke",
                    "base_source_image": source["inputs"]["opening_frame"],
                    "frames": 33,
                }
            ]
            manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
            project = load_project(manifest)
            segment = create_attempt(project, "S010", "S010_C001")
            bridge_source = dict(project.source)
            bridge_source["workflow_api"] = project.source["bridge_workflow_api"]
            bridge = create_attempt(
                ProjectConfig(path=project.path, source=bridge_source),
                "S010",
                "B010",
            )

            with patch("wan22_longform.cli._client", return_value=object()), patch(
                "wan22_longform.cli.resume_attempt",
                side_effect=lambda _project, attempt, _client: attempt,
            ) as resume_attempt:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(["resume", str(manifest)]), 0)

            selected_ids = {call.args[1].segment_id for call in resume_attempt.call_args_list}
            self.assertEqual(selected_ids, {segment.segment_id, bridge.segment_id})
            self.assertEqual(
                {item["segment_id"] for item in json.loads(stdout.getvalue())["resumed"]},
                selected_ids,
            )

    def _manifest(self, root: Path, segment_ids: tuple[str, ...]) -> Path:
        opening = root / "opening.png"
        opening.write_bytes(b"opening")
        segment_graph = PROJECT_DIR / "tests" / "fixtures" / "native_segment_api.json"
        bridge_graph = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"
        manifest = root / "project.yaml"
        source = {
            "schema_version": 1,
            "project_id": "assembly_record_fixture",
            "title": "Assembly record fixture",
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
            "workflow_api": str(segment_graph),
            "bridge_workflow_api": str(bridge_graph),
            "workflow_hashes": {
                "segment_api": hashlib.sha256(segment_graph.read_bytes()).hexdigest(),
                "bridge_api": hashlib.sha256(bridge_graph.read_bytes()).hexdigest(),
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
            "request": {"positive": "neutral adult", "negative": "low quality"},
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
                            "id": segment_id,
                            "action": "A neutral adult takes one measured step.",
                            "expected_seconds": 1,
                            "seed_offset": index,
                        }
                        for index, segment_id in enumerate(segment_ids)
                    ],
                }
            ],
            "bridges": [],
            "assembly_order": [
                {"shot_id": "S010", "segment_id": segment_id}
                for segment_id in segment_ids
            ],
        }
        manifest.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
        return manifest

    @staticmethod
    def _accepted_attempt(project, segment_id: str):
        attempt = create_attempt(project, "S010", segment_id)
        for state in (AttemptState.RENDERING, AttemptState.RENDERED, AttemptState.NEEDS_REVIEW):
            attempt = transition_attempt(attempt, state, "fixture")
        attempt = transition_attempt(attempt, AttemptState.ACCEPTED, "approved fixture")
        output = attempt.path / "outputs" / "segment.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(segment_id.encode("utf-8"))
        write_metadata(attempt, RenderMetadata(outputs={"segment": output}))
        return attempt

    def _assembly_execution_patch(self):
        def fake_execute(plan, *, decision_log):
            decision_log.parent.mkdir(parents=True, exist_ok=True)
            decision_log.write_text('{"decisions": []}\n', encoding="utf-8")
            for operation in plan.operations:
                if operation.kind in {"review_mp4", "edit_master_ffv1", "edit_master_prores"}:
                    operation.output.parent.mkdir(parents=True, exist_ok=True)
                    operation.output.write_bytes(operation.kind.encode("utf-8"))
            return AssemblyExecution(
                expected_frame_count=34,
                expected_duration=Fraction(17, 8),
                output_fps=Fraction(16, 1),
                validated_outputs=(),
                rife_ready=None,
            )

        return self._patched_assembly_execution(fake_execute)

    @staticmethod
    def _patched_assembly_execution(fake_execute):
        return _AssemblyExecutionPatch(fake_execute)


class _AssemblyExecutionPatch:
    def __init__(self, fake_execute):
        self.probe = patch("wan22_longform.cli.probe_media", side_effect=self._media)
        self.boundary = patch(
            "wan22_longform.cli.compare_boundary",
            return_value=BoundaryDecision("left", "right", 9, 0, False, "distinct"),
        )
        self.execute = patch("wan22_longform.cli.execute_assembly_plan", side_effect=fake_execute)

    def __enter__(self):
        self.probe.__enter__()
        self.boundary.__enter__()
        self.execute.__enter__()
        return self

    def __exit__(self, *args):
        self.execute.__exit__(*args)
        self.boundary.__exit__(*args)
        return self.probe.__exit__(*args)

    @staticmethod
    def _media(path: Path) -> MediaSpec:
        return MediaSpec(
            path=path,
            width=16,
            height=8,
            fps=Fraction(16, 1),
            time_base=Fraction(1, 16),
            pixel_format="yuv420p",
            codec="ffv1",
            profile=None,
            color_space=None,
            color_transfer=None,
            color_primaries=None,
            audio=None,
            frame_count=17,
        )


if __name__ == "__main__":
    unittest.main()
