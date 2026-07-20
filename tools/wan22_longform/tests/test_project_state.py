from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.config import ProjectConfig, load_project  # noqa: E402
from wan22_longform.metadata import RenderMetadata, write_metadata  # noqa: E402
import wan22_longform.project as project_module  # noqa: E402
from wan22_longform.project import (  # noqa: E402
    AssemblyState,
    AttemptState,
    ProjectStateError,
    create_assembly_record,
    create_attempt,
    load_assembly_record,
    load_attempt,
    needs_render,
    transition_assembly_record,
    transition_attempt,
    verify_assembly_record_inputs,
)


class ProjectStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = self.root / "project.yaml"
        self.manifest.write_text("preset: P0_IDENTITY_BASELINE\n", encoding="utf-8")
        self.workflow = self.root / "workflow.json"
        self.workflow.write_text('{"1":{"class_type":"KSampler"}}', encoding="utf-8")
        self.input_frame = self.root / "opening.png"
        self.input_frame.write_bytes(b"opening-frame")
        self.project = ProjectConfig(
            path=self.manifest,
            source={
                "workflow_api": str(self.workflow),
                "request": {"prompt": "A fixture shot"},
                "inputs": {"opening_frame": str(self.input_frame)},
                "attempts_dir": str(self.root / "attempts"),
            },
        )
        self.now = lambda: datetime(2026, 7, 19, 12, 30, 45, tzinfo=UTC)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_retry_creates_a_distinct_immutable_attempt(self) -> None:
        first = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        retry = create_attempt(self.project, "S010", "S010_C001", now=self.now)

        self.assertNotEqual(first.path, retry.path)
        self.assertTrue((first.path / "source-manifest.yaml").is_file())
        self.assertTrue((first.path / "workflow-api.json").is_file())
        self.assertTrue((first.path / "request.json").is_file())
        self.assertTrue((first.path / "provenance.json").is_file())
        self.assertEqual(first.state, AttemptState.PLANNED)
        self.assertEqual(retry.parent_attempt, first.path)

        provenance = json.loads((first.path / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(
            provenance["inputs"]["opening_frame"],
            hashlib.sha256(self.input_frame.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["workflow"],
            hashlib.sha256(self.workflow.read_bytes()).hexdigest(),
        )

    def test_transition_history_is_append_only_and_rejects_invalid_jumps(self) -> None:
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)

        with self.assertRaisesRegex(ProjectStateError, "planned.*rendered"):
            transition_attempt(attempt, AttemptState.RENDERED, None, now=self.now)

        rendering = transition_attempt(attempt, AttemptState.RENDERING, "queued", now=self.now)
        rendered = transition_attempt(rendering, AttemptState.RENDERED, "complete", now=self.now)
        review = transition_attempt(rendered, AttemptState.NEEDS_REVIEW, None, now=self.now)
        frame = self.root / "last-frame.png"
        frame.write_bytes(b"last-frame")
        accepted = transition_attempt(
            review,
            AttemptState.ACCEPTED,
            "approved",
            selected_continuation_frame=frame,
            now=self.now,
        )

        decision_paths = sorted((attempt.path / "decisions").glob("*.json"))
        self.assertEqual(len(decision_paths), 4)
        accepted_record = json.loads(decision_paths[-1].read_text(encoding="utf-8"))
        self.assertEqual(accepted_record["note"], "approved")
        self.assertEqual(accepted_record["selected_continuation_frame"]["path"], str(frame))
        self.assertEqual(
            accepted_record["selected_continuation_frame"]["sha256"],
            hashlib.sha256(frame.read_bytes()).hexdigest(),
        )
        self.assertTrue(accepted_record["timestamp"].endswith("Z"))
        self.assertEqual(accepted.state, AttemptState.ACCEPTED)

    def test_resume_skips_accepted_attempt(self) -> None:
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        rendering = transition_attempt(attempt, AttemptState.RENDERING, None, now=self.now)
        rendered = transition_attempt(rendering, AttemptState.RENDERED, None, now=self.now)
        review = transition_attempt(rendered, AttemptState.NEEDS_REVIEW, None, now=self.now)
        accepted = transition_attempt(review, AttemptState.ACCEPTED, "approved", now=self.now)

        self.assertFalse(needs_render(accepted))

    def test_disk_round_trip_reconstructs_accepted_state_for_resume(self) -> None:
        attempt = self._review_attempt()
        accepted = transition_attempt(attempt, AttemptState.ACCEPTED, "approved", now=self.now)

        resumed = load_attempt(accepted.path)

        self.assertEqual(resumed.state, AttemptState.ACCEPTED)
        self.assertFalse(needs_render(resumed))
        self.assertFalse(needs_render(attempt))

    def test_stale_attempt_cannot_append_a_second_decision(self) -> None:
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        transition_attempt(attempt, AttemptState.RENDERING, "queued", now=self.now)
        decision_dir = attempt.path / "decisions"
        before = sorted(path.name for path in decision_dir.glob("*.json"))

        with self.assertRaisesRegex(ProjectStateError, "stale.*persisted"):
            transition_attempt(attempt, AttemptState.RENDERING, "again", now=self.now)

        self.assertEqual(sorted(path.name for path in decision_dir.glob("*.json")), before)

    def test_concurrent_divergent_outcomes_claim_one_numeric_decision(self) -> None:
        review = self._review_attempt()
        barrier = threading.Barrier(2)
        results: list[tuple[str, object]] = []
        original_write_json = project_module._write_json

        def synchronized_write(path: Path, payload: object) -> None:
            if path.parent.name == "decisions":
                barrier.wait(timeout=5)
            original_write_json(path, payload)

        def contend(target: AttemptState, note: str) -> None:
            try:
                results.append(("success", transition_attempt(review, target, note, now=self.now)))
            except ProjectStateError as error:
                results.append(("error", error))

        with patch("wan22_longform.project._write_json", synchronized_write):
            contenders = [
                threading.Thread(target=contend, args=(AttemptState.ACCEPTED, "approved")),
                threading.Thread(target=contend, args=(AttemptState.REJECTED, "rejected")),
            ]
            for contender in contenders:
                contender.start()
            for contender in contenders:
                contender.join(timeout=5)

        self.assertFalse(any(contender.is_alive() for contender in contenders))
        self.assertEqual([kind for kind, _ in results].count("success"), 1)
        self.assertEqual([kind for kind, _ in results].count("error"), 1)
        conflict = next(value for kind, value in results if kind == "error")
        self.assertIn("persisted state", str(conflict))
        decision_paths = sorted((review.path / "decisions").glob("*.json"))
        self.assertEqual(len(decision_paths), 4)
        self.assertEqual(decision_paths[-1].name, "0004.json")
        successful_attempt = next(value for kind, value in results if kind == "success")
        self.assertEqual(load_attempt(review.path).state, successful_attempt.state)

    def test_operator_outcomes_require_a_non_empty_note(self) -> None:
        for state, note in (
            (AttemptState.ACCEPTED, None),
            (AttemptState.REJECTED, ""),
            (AttemptState.RETRY_REQUESTED, "  "),
        ):
            with self.subTest(state=state):
                review = self._review_attempt()
                decision_dir = review.path / "decisions"
                before = sorted(path.name for path in decision_dir.glob("*.json"))

                with self.assertRaisesRegex(ProjectStateError, "non-empty operator note"):
                    transition_attempt(review, state, note, now=self.now)

                self.assertEqual(sorted(path.name for path in decision_dir.glob("*.json")), before)

    def test_assembly_record_leaves_accepted_source_attempt_reusable(self) -> None:
        source = self._review_attempt()
        source = transition_attempt(source, AttemptState.ACCEPTED, "approved", now=self.now)
        rendered = self.root / "accepted-segment.mp4"
        rendered.write_bytes(b"accepted-segment")

        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                {
                    "attempt_id": source.attempt_id,
                    "attempt_path": source.path,
                    "output": {
                        "path": rendered,
                        "sha256": hashlib.sha256(rendered.read_bytes()).hexdigest(),
                    },
                    "segment_id": source.segment_id,
                    "shot_id": source.shot_id,
                },
            ),
            requested={"scope": "project", "targets": {"review_mp4": self.root / "review.mp4"}},
            now=self.now,
        )

        assembling = transition_assembly_record(
            record, AssemblyState.ASSEMBLING, "local FFmpeg assembly started", now=self.now
        )
        assembled = transition_assembly_record(
            assembling, AssemblyState.ASSEMBLED, "all requested outputs validated", now=self.now
        )
        final = transition_assembly_record(
            assembled, AssemblyState.FINAL, "assembly evidence finalized", now=self.now
        )

        self.assertEqual(load_attempt(source.path).state, AttemptState.ACCEPTED)
        self.assertEqual(load_assembly_record(record.path).state, AssemblyState.FINAL)
        self.assertEqual(final.state, AssemblyState.FINAL)
        payload = json.loads((record.path / "assembly.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["inputs"][0]["attempt_id"], source.attempt_id)
        self.assertEqual(payload["inputs"][0]["output"]["sha256"], hashlib.sha256(rendered.read_bytes()).hexdigest())

    def test_invalid_assembly_source_provenance_leaves_no_partial_record_directory(self) -> None:
        source = self._review_attempt()
        source = transition_attempt(source, AttemptState.ACCEPTED, "approved", now=self.now)
        rendered = self.root / "accepted-segment.mp4"
        rendered.write_bytes(b"changed source")

        with self.assertRaisesRegex(ProjectStateError, "output hash changed"):
            create_assembly_record(
                self.project,
                "project",
                inputs=(
                    {
                        "attempt_id": source.attempt_id,
                        "attempt_path": source.path,
                        "output": {
                            "path": rendered,
                            "sha256": hashlib.sha256(b"original source").hexdigest(),
                        },
                        "segment_id": source.segment_id,
                        "shot_id": source.shot_id,
                    },
                ),
                requested={"scope": "project"},
                now=self.now,
            )

        self.assertFalse((self.root / "assembly-records" / "project").exists())

    def test_assembly_record_rechecks_source_hashes_before_execution(self) -> None:
        source = self._review_attempt()
        source = transition_attempt(source, AttemptState.ACCEPTED, "approved", now=self.now)
        rendered = self.root / "accepted-segment.mp4"
        rendered.write_bytes(b"original source")
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                {
                    "attempt_id": source.attempt_id,
                    "attempt_path": source.path,
                    "output": {
                        "path": rendered,
                        "sha256": hashlib.sha256(rendered.read_bytes()).hexdigest(),
                    },
                    "segment_id": source.segment_id,
                    "shot_id": source.shot_id,
                },
            ),
            requested={"scope": "project"},
            now=self.now,
        )
        rendered.write_bytes(b"changed after assembly selection")

        with self.assertRaisesRegex(ProjectStateError, "output hash changed"):
            verify_assembly_record_inputs(record)

    def test_metadata_records_output_provenance_without_overwriting(self) -> None:
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        output = self.root / "segment.mp4"
        output.write_bytes(b"rendered-output")

        metadata_path = write_metadata(
            attempt,
            RenderMetadata(outputs={"segment": output}, rendered_at=self.now()),
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["outputs"]["segment"]["path"], str(output))
        self.assertEqual(
            metadata["outputs"]["segment"]["sha256"],
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        with self.assertRaises(FileExistsError):
            write_metadata(attempt, RenderMetadata(outputs={"segment": output}))

    def test_loaded_immutable_config_snapshots_structured_request_and_workflow(self) -> None:
        self.manifest.write_text(
            """preset: P0_IDENTITY_BASELINE
models:
  high: high.safetensors
  low: low.safetensors
workflow_api:
  node: {}
request:
  prompt: fixture
inputs:
  opening_frame: opening.png
""",
            encoding="utf-8",
        )
        project = load_project(self.manifest)

        attempt = create_attempt(project, "S010", "S010_C001", now=self.now)

        self.assertEqual(
            json.loads((attempt.path / "request.json").read_text(encoding="utf-8")),
            {"prompt": "fixture"},
        )
        self.assertEqual(
            json.loads((attempt.path / "workflow-api.json").read_text(encoding="utf-8")),
            {"node": {}},
        )

    def _review_attempt(self):
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        for state in (
            AttemptState.RENDERING,
            AttemptState.RENDERED,
            AttemptState.NEEDS_REVIEW,
        ):
            attempt = transition_attempt(attempt, state, None, now=self.now)
        return attempt


if __name__ == "__main__":
    unittest.main()
