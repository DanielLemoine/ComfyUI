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
    accepted_attempt_evidence,
    create_assembly_record,
    create_attempt,
    inspect_assembly_records,
    inspect_attempt_integrity,
    load_assembly_record,
    load_attempt,
    needs_render,
    transition_assembly_record,
    transition_attempt,
    recorded_boundary_approvals,
    verify_assembly_record_integrity,
    verify_assembly_record_inputs,
)
from wan22_longform.qc import initialize_qc  # noqa: E402


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
                "project_id": "project-state-fixture",
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

        attempt_record = json.loads(
            (first.path / "attempt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            attempt_record.get("lineage"),
            {
                "project_id": "project-state-fixture",
                "source_manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            },
        )

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
        source, rendered, evidence = self._accepted_source()

        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                evidence,
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
        self.assertEqual(
            payload["lineage"],
            {
                "project_id": "project-state-fixture",
                "source_manifest_sha256": hashlib.sha256(
                    self.manifest.read_bytes()
                ).hexdigest(),
            },
        )
        self.assertEqual(
            set(payload["inputs"][0]),
            {
                "acceptance_decision",
                "attempt_id",
                "attempt_path",
                "lineage",
                "output",
                "planned",
                "provenance",
                "qc",
                "render_history",
                "render_metadata",
                "segment_id",
                "shot_id",
                "source_manifest",
                "submission",
            },
        )
        self.assertRegex(payload["root_digest"], r"^[0-9a-f]{64}$")

    def test_invalid_assembly_source_provenance_leaves_no_partial_record_directory(self) -> None:
        _source, _rendered, evidence = self._accepted_source()
        evidence["output"]["sha256"] = hashlib.sha256(b"original source").hexdigest()

        with self.assertRaisesRegex(ProjectStateError, "immutable evidence"):
            create_assembly_record(
                self.project,
                "project",
                inputs=(
                    evidence,
                ),
                requested={"scope": "project"},
                now=self.now,
            )

        self.assertFalse((self.root / "assembly-records" / "project").exists())

    def test_assembly_record_rechecks_source_hashes_before_execution(self) -> None:
        _source, rendered, evidence = self._accepted_source()
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                evidence,
            ),
            requested={"scope": "project"},
            now=self.now,
        )
        rendered.write_bytes(b"changed after assembly selection")

        with self.assertRaisesRegex(ProjectStateError, "output hash changed"):
            verify_assembly_record_inputs(record)

    def test_assembly_record_treats_missing_boundary_approvals_as_empty(self) -> None:
        _source, _rendered, evidence = self._accepted_source()
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                evidence,
            ),
            requested={"scope": "project"},
            now=self.now,
        )

        self.assertEqual(recorded_boundary_approvals(record), ())

    def test_failed_assembly_root_detects_requested_approval_tampering(self) -> None:
        _source, _rendered, evidence = self._accepted_source()
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                evidence,
            ),
            requested={
                "boundary_approvals": [
                    {"boundary_index": 1, "note": "Reviewed at 200%."}
                ],
                "scope": "project",
            },
            now=self.now,
        )
        failed = transition_assembly_record(
            record,
            AssemblyState.FAILED,
            "planning failed",
            now=self.now,
        )
        root_payload = json.loads(
            (record.path / "assembly.json").read_text(encoding="utf-8")
        )
        root_payload["requested"]["boundary_approvals"][0]["note"] = "forged approval"
        (record.path / "assembly.json").write_text(
            json.dumps(root_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        integrity = verify_assembly_record_integrity(failed)

        self.assertFalse(integrity.intact)

    def test_assembly_decision_chain_detects_intermediate_tampering(self) -> None:
        _source, _rendered, evidence = self._accepted_source()
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(
                evidence,
            ),
            requested={"scope": "project"},
            now=self.now,
        )
        assembling = transition_assembly_record(
            record, AssemblyState.ASSEMBLING, "started", now=self.now
        )
        failed = transition_assembly_record(
            assembling, AssemblyState.FAILED, "failed", now=self.now
        )
        first_decision_path = record.path / "decisions" / "0001.json"
        first_decision = json.loads(first_decision_path.read_text(encoding="utf-8"))
        first_decision["note"] = "forged start"
        first_decision_path.write_text(
            json.dumps(first_decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        integrity = verify_assembly_record_integrity(failed)

        self.assertFalse(integrity.intact)

    def test_accepted_attempt_rejects_changed_immutable_artifact_hashes(self) -> None:
        cases = (
            "source-manifest.yaml",
            "provenance.json",
            "render-metadata.json",
            "qc.yaml",
            "decisions/0004.json",
        )
        for relative_path in cases:
            with self.subTest(relative_path=relative_path):
                accepted, _output, _evidence = self._accepted_source()
                path = accepted.path / relative_path
                path.write_bytes(path.read_bytes() + b"\n")

                with self.assertRaises(ProjectStateError):
                    accepted_attempt_evidence(self.project, accepted)

    def test_planned_attempt_tamper_is_integrity_failed_not_verified(self) -> None:
        attempt = create_attempt(self.project, "S010", "S010_C001", now=self.now)
        provenance = attempt.path / "provenance.json"
        provenance.write_bytes(provenance.read_bytes() + b"\n")

        status, failures = inspect_attempt_integrity(self.project, attempt)

        self.assertEqual(status, "failed")
        self.assertTrue(any("provenance" in failure for failure in failures))

    def test_legacy_attempt_is_readable_but_explicitly_unverified(self) -> None:
        accepted, _output, _evidence = self._accepted_source()
        attempt_path = accepted.path / "attempt.json"
        attempt_payload = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt_payload.pop("lineage")
        attempt_path.write_text(
            json.dumps(attempt_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        loaded = load_attempt(accepted.path)
        status, failures = inspect_attempt_integrity(self.project, loaded)

        self.assertEqual(loaded.state, AttemptState.ACCEPTED)
        self.assertEqual(status, "legacy_unverified")
        self.assertTrue(any("legacy" in failure for failure in failures))
        with self.assertRaisesRegex(ProjectStateError, "legacy"):
            accepted_attempt_evidence(self.project, loaded)

    def test_legacy_assembly_record_is_readable_but_explicitly_unverified(self) -> None:
        _source, _output, evidence = self._accepted_source()
        record = create_assembly_record(
            self.project,
            "project",
            inputs=(evidence,),
            requested={"scope": "project"},
            now=self.now,
        )
        root_path = record.path / "assembly.json"
        root_payload = json.loads(root_path.read_text(encoding="utf-8"))
        root_payload.pop("root_digest")
        root_payload.pop("lineage")
        root_path.write_text(
            json.dumps(root_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        loaded = load_assembly_record(record.path)
        integrity = verify_assembly_record_integrity(loaded, self.project)

        self.assertEqual(loaded.state, AssemblyState.PLANNED)
        self.assertEqual(integrity.status, "legacy_unverified")
        self.assertFalse(integrity.intact)

    def test_project_inspection_rejects_another_projects_assembly_record(self) -> None:
        _source, _output, evidence = self._accepted_source()
        create_assembly_record(
            self.project,
            "project",
            inputs=(evidence,),
            requested={"scope": "project"},
            now=self.now,
        )
        other_manifest = self.root / "other-project.yaml"
        other_manifest.write_text("project_id: other-project\n", encoding="utf-8")
        other_source = dict(self.project.source)
        other_source["project_id"] = "other-project"
        other_source["assembly_records_dir"] = str(self.root / "assembly-records")
        other_project = ProjectConfig(path=other_manifest, source=other_source)

        inspected = inspect_assembly_records(other_project)

        self.assertEqual(len(inspected), 1)
        self.assertEqual(inspected[0].status, "failed")
        self.assertTrue(
            any("project lineage" in failure for failure in inspected[0].failures)
        )

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
            """project_id: structured-snapshot-fixture
preset: P0_IDENTITY_BASELINE
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

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _submission_fixture(self, attempt) -> dict[str, str]:
        input_upload = attempt.path / "input-upload.json"
        configured = attempt.path / "configured-workflow-api.json"
        request = attempt.path / "submission-request.json"
        provenance = attempt.path / "submission-provenance.json"
        self._write_json(input_upload, {})
        configured_payload = json.loads(
            (attempt.path / "workflow-api.json").read_text(encoding="utf-8")
        )
        self._write_json(configured, configured_payload)
        self._write_json(request, {"prompt": configured_payload})
        source_manifest = next(attempt.path.glob("source-manifest.*"))
        self._write_json(
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
        self._write_json(
            attempt.path / "submission-intent.json",
            {
                "submission_request": request_evidence,
                "submission_provenance": provenance_evidence,
            },
        )
        self._write_json(
            attempt.path / "queue.json",
            {
                "kind": "segment",
                "prompt_id": "fixture-prompt",
                "submission_request": request_evidence,
                "submission_provenance": provenance_evidence,
            },
        )
        self._write_json(
            attempt.path / "history.json",
            {"prompt_id": "fixture-prompt", "history": {}},
        )
        return request_evidence

    def _accepted_source(self):
        attempt = self._review_attempt()
        output = attempt.path / "outputs" / "segment.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"accepted-segment")
        request_evidence = self._submission_fixture(attempt)
        write_metadata(
            attempt,
            RenderMetadata(
                outputs={"segment": output},
                details={
                    "prompt_id": "fixture-prompt",
                    "submission_request": request_evidence,
                },
            ),
        )
        head = attempt.path / "head.png"
        tail = attempt.path / "tail.png"
        sheet = attempt.path / "contact-sheet.png"
        head.write_bytes(b"head")
        tail.write_bytes(b"tail")
        sheet.write_bytes(b"sheet")
        initialize_qc(
            attempt,
            video=output,
            head_frames=[head],
            tail_frames=[tail],
            contact_sheet=sheet,
            automatic_continuation_authorized=False,
        )
        accepted = transition_attempt(
            attempt, AttemptState.ACCEPTED, "approved", now=self.now
        )
        return accepted, output, accepted_attempt_evidence(self.project, accepted)


if __name__ == "__main__":
    unittest.main()
