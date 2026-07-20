from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Mapping, Sequence

import yaml

from .config import ProjectConfig
from .hashing import sha256_file


class AttemptState(StrEnum):
    PLANNED = "planned"
    RENDERING = "rendering"
    RENDERED = "rendered"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRY_REQUESTED = "retry_requested"


class AssemblyState(StrEnum):
    """Lifecycle for one immutable output assembly record."""

    PLANNED = "planned"
    ASSEMBLING = "assembling"
    ASSEMBLED = "assembled"
    FINAL = "final"
    FAILED = "failed"


class ProjectStateError(ValueError):
    """Raised when an attempt is moved outside the long-form review flow."""


@dataclass(frozen=True)
class Attempt:
    path: Path
    attempt_id: str
    shot_id: str
    segment_id: str
    state: AttemptState
    created_at: datetime
    parent_attempt: Path | None = None


@dataclass(frozen=True)
class AssemblyRecord:
    path: Path
    assembly_id: str
    scope: str
    state: AssemblyState
    created_at: datetime


@dataclass(frozen=True)
class AssemblyRecordIntegrity:
    """Read-only verification result for one persisted assembly record."""

    record: AssemblyRecord | None
    path: Path
    failures: tuple[str, ...]
    status: str = "verified"

    @property
    def intact(self) -> bool:
        return self.record is not None and self.status == "verified" and not self.failures


@dataclass(frozen=True)
class QcCandidate:
    path: Path
    sha256: str
    kind: str


@dataclass(frozen=True)
class SelectedInput:
    path: Path
    sha256: str
    details: Mapping[str, Any]


_TRANSITIONS: Mapping[AttemptState, frozenset[AttemptState]] = {
    AttemptState.PLANNED: frozenset({AttemptState.RENDERING}),
    AttemptState.RENDERING: frozenset({AttemptState.RENDERED}),
    AttemptState.RENDERED: frozenset({AttemptState.NEEDS_REVIEW}),
    AttemptState.NEEDS_REVIEW: frozenset(
        {AttemptState.ACCEPTED, AttemptState.REJECTED, AttemptState.RETRY_REQUESTED}
    ),
    AttemptState.ACCEPTED: frozenset(),
    AttemptState.REJECTED: frozenset(),
    AttemptState.RETRY_REQUESTED: frozenset(),
}

_ASSEMBLY_TRANSITIONS: Mapping[AssemblyState, frozenset[AssemblyState]] = {
    AssemblyState.PLANNED: frozenset({AssemblyState.ASSEMBLING, AssemblyState.FAILED}),
    AssemblyState.ASSEMBLING: frozenset({AssemblyState.ASSEMBLED, AssemblyState.FAILED}),
    AssemblyState.ASSEMBLED: frozenset({AssemblyState.FINAL}),
    AssemblyState.FINAL: frozenset(),
    AssemblyState.FAILED: frozenset(),
}

_OPERATOR_OUTCOMES = frozenset(
    {AttemptState.ACCEPTED, AttemptState.REJECTED, AttemptState.RETRY_REQUESTED}
)


def project_lineage(project: ProjectConfig) -> dict[str, str]:
    """Return the current manifest identity used by all privileged project paths."""
    project_id = project.source.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ProjectStateError("project lineage requires a non-empty project_id")
    if not project.path.is_file():
        raise ProjectStateError(f"project lineage source manifest does not exist: {project.path}")
    return {
        "project_id": project_id,
        "source_manifest_sha256": sha256_file(project.path),
    }


def create_attempt(
    project: ProjectConfig,
    shot_id: str,
    segment_id: str,
    *,
    selected_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Attempt:
    """Create a new append-only attempt directory with local provenance snapshots."""
    _validate_identifier(shot_id, "shot_id")
    _validate_identifier(segment_id, "segment_id")
    created_at = _as_utc(now())
    lineage = project_lineage(project)
    segment_root = _attempts_root(project) / shot_id / segment_id
    segment_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in segment_root.iterdir() if path.is_dir())
    attempt_number = len(existing) + 1
    timestamp = _utc_timestamp(created_at).replace(":", "").replace("-", "")
    while True:
        attempt_id = f"attempt-{attempt_number:04d}-{timestamp}"
        attempt_path = segment_root / attempt_id
        try:
            attempt_path.mkdir()
            break
        except FileExistsError:
            attempt_number += 1

    parent_attempt = existing[-1] if existing else None
    manifest_snapshot = attempt_path / f"source-manifest{project.path.suffix or '.yaml'}"
    copyfile(project.path, manifest_snapshot)
    workflow_snapshot = attempt_path / "workflow-api.json"
    _write_workflow_snapshot(workflow_snapshot, project)
    request_snapshot = attempt_path / "request.json"
    _write_json(request_snapshot, _request_payload(project))

    attempt = Attempt(
        path=attempt_path,
        attempt_id=attempt_id,
        shot_id=shot_id,
        segment_id=segment_id,
        state=AttemptState.PLANNED,
        created_at=created_at,
        parent_attempt=parent_attempt,
    )
    _write_json(
        attempt_path / "attempt.json",
        {
            "attempt_id": attempt.attempt_id,
            "created_at": _utc_timestamp(created_at),
            "parent_attempt": str(parent_attempt) if parent_attempt else None,
            "lineage": lineage,
            "retry_of": str(parent_attempt) if parent_attempt else None,
            "segment_id": segment_id,
            "shot_id": shot_id,
            "state": attempt.state,
        },
    )
    provenance = {
        "inputs": {name: sha256_file(path) for name, path in _input_paths(project).items()},
        "request": sha256_file(request_snapshot),
        "source_manifest": sha256_file(manifest_snapshot),
        "workflow": sha256_file(workflow_snapshot),
        "lineage": lineage,
    }
    if selected_inputs is not None:
        provenance["selected_inputs"] = _selected_inputs_payload(selected_inputs)
    _write_json(attempt_path / "provenance.json", provenance)
    return attempt


def transition_attempt(
    attempt: Attempt,
    target: AttemptState,
    note: str | None,
    *,
    selected_continuation_frame: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Attempt:
    """Append one validated state decision and return the new immutable value."""
    persisted, decision_paths = _load_attempt(attempt.path)
    if attempt.attempt_id != persisted.attempt_id:
        raise ProjectStateError("attempt identity does not match its persisted record")
    if attempt.state != persisted.state:
        raise ProjectStateError(
            f"stale attempt state: supplied {attempt.state}, persisted {persisted.state}"
        )
    if target not in _TRANSITIONS[persisted.state]:
        raise ProjectStateError(f"invalid attempt transition: {persisted.state} -> {target}")
    if target in _OPERATOR_OUTCOMES and not _has_operator_note(note):
        raise ProjectStateError(f"{target} requires a non-empty operator note")
    if _decision_paths(attempt.path) != decision_paths:
        raise ProjectStateError("attempt decision chain changed before transition")

    decision_dir = attempt.path / "decisions"
    decision_dir.mkdir(exist_ok=True)
    decision_number = len(decision_paths) + 1
    timestamp = _as_utc(now())
    continuation = None
    if selected_continuation_frame is not None:
        continuation = {
            "path": str(selected_continuation_frame),
            "sha256": sha256_file(selected_continuation_frame),
        }
    payload = {
        "attempt_id": persisted.attempt_id,
        "from": persisted.state,
        "note": note,
        "parent_attempt": str(persisted.parent_attempt) if persisted.parent_attempt else None,
        "retry_of": str(persisted.parent_attempt) if persisted.parent_attempt else None,
        "selected_continuation_frame": continuation,
        "timestamp": _utc_timestamp(timestamp),
        "to": target,
    }
    if target is AttemptState.ACCEPTED:
        payload["accepted_evidence"] = _acceptance_evidence(persisted)
    decision_path = decision_dir / f"{decision_number:04d}.json"
    try:
        _write_json(decision_path, payload)
    except FileExistsError as error:
        current = load_attempt(attempt.path)
        raise ProjectStateError(
            f"attempt decision claim lost; persisted state is {current.state}"
        ) from error
    if target is AttemptState.ACCEPTED:
        _write_json(
            attempt.path / "acceptance.json",
            {
                "accepted_evidence": payload["accepted_evidence"],
                "attempt_id": persisted.attempt_id,
                "decision": _attempt_artifact(decision_path, "acceptance decision"),
            },
        )
    return replace(persisted, state=target)


def load_attempt(path: Path) -> Attempt:
    """Reconstruct the current attempt state from its immutable decision log."""
    attempt, _ = _load_attempt(path)
    return attempt


def needs_render(attempt: Attempt) -> bool:
    """Only a planned attempt needs a render request; accepted attempts never do."""
    return load_attempt(attempt.path).state is AttemptState.PLANNED


def selected_tail_frame(attempt: Attempt) -> QcCandidate:
    """Read the immutable tail selection recorded with one accepted attempt."""
    persisted, decision_paths = _load_attempt(attempt.path)
    if persisted.state is not AttemptState.ACCEPTED:
        raise ProjectStateError("continuation source attempt is not accepted")
    if persisted.attempt_id != attempt.attempt_id:
        raise ProjectStateError("continuation source attempt identity does not match")
    acceptance = []
    for path in decision_paths:
        decision = _read_json(path)
        if decision.get("to") == AttemptState.ACCEPTED:
            acceptance.append(decision)
    if len(acceptance) != 1:
        raise ProjectStateError("accepted attempt has no unambiguous acceptance decision")
    selected = acceptance[0].get("selected_continuation_frame")
    if not isinstance(selected, Mapping):
        raise ProjectStateError("accepted attempt has no selected tail continuation frame")
    raw_path = selected.get("path")
    expected_hash = selected.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ProjectStateError("accepted tail continuation selection is invalid")
    candidate = accepted_qc_candidate(persisted, Path(raw_path), "tail")
    if candidate.sha256 != expected_hash:
        raise ProjectStateError("selected tail continuation frame hash changed")
    return candidate


def accepted_qc_candidate(attempt: Attempt, path: Path, kind: str) -> QcCandidate:
    """Return one hash-verified head or tail candidate from an accepted attempt."""
    if kind not in {"head", "tail"}:
        raise ProjectStateError("QC candidate kind must be head or tail")
    persisted = load_attempt(attempt.path)
    if persisted.state is not AttemptState.ACCEPTED:
        raise ProjectStateError("QC candidate source attempt is not accepted")
    qc = _read_qc(attempt.path / "qc.yaml")
    candidates = qc.get("candidate_frames")
    if not isinstance(candidates, Mapping):
        raise ProjectStateError("QC record has no candidate_frames mapping")
    entries = candidates.get(kind)
    if not isinstance(entries, list):
        raise ProjectStateError(f"QC record has no {kind} candidate list")
    wanted = path.resolve()
    matches: list[QcCandidate] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ProjectStateError("QC candidate entry is invalid")
        raw_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ProjectStateError("QC candidate entry has invalid provenance")
        candidate_path = Path(raw_path).resolve()
        if candidate_path != wanted:
            continue
        if not candidate_path.is_file():
            raise ProjectStateError(f"QC candidate frame does not exist: {candidate_path}")
        actual_hash = sha256_file(candidate_path)
        if actual_hash != expected_hash:
            raise ProjectStateError(f"QC candidate frame hash changed: {candidate_path}")
        matches.append(QcCandidate(candidate_path, actual_hash, kind))
    if len(matches) != 1:
        raise ProjectStateError(f"accepted attempt has no unambiguous {kind} QC candidate")
    return matches[0]


def selected_input(attempt: Attempt, name: str) -> SelectedInput:
    """Read one hash-verified input selected before an immutable attempt was created."""
    record = _read_json(attempt.path / "provenance.json")
    selected = record.get("selected_inputs")
    if not isinstance(selected, Mapping):
        raise ProjectStateError("attempt has no selected input provenance")
    entry = selected.get(name)
    if not isinstance(entry, Mapping):
        raise ProjectStateError(f"attempt has no selected provenance for {name}")
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ProjectStateError(f"attempt selected provenance is invalid for {name}")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ProjectStateError(f"attempt selected input does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ProjectStateError(f"attempt selected input hash changed: {path}")
    return SelectedInput(path, actual_hash, dict(entry))


def verify_attempt_lineage(project: ProjectConfig, attempt: Attempt) -> dict[str, str]:
    """Require a new-format attempt to match the current project manifest exactly."""
    expected = project_lineage(project)
    actual = _attempt_lineage(attempt)
    if actual != expected:
        raise ProjectStateError(
            "attempt project lineage does not match the current project manifest"
        )
    return actual


def accepted_attempt_evidence(
    project: ProjectConfig, attempt: Attempt
) -> dict[str, Any]:
    """Return canonical immutable evidence for a current-project accepted attempt."""
    return _accepted_attempt_evidence(attempt, project_lineage(project))


def inspect_attempt_integrity(
    project: ProjectConfig, attempt: Attempt
) -> tuple[str, tuple[str, ...]]:
    """Classify readable attempts without promoting legacy data to verified."""
    try:
        verify_attempt_lineage(project, attempt)
        if attempt.state is AttemptState.ACCEPTED:
            accepted_attempt_evidence(project, attempt)
    except ProjectStateError as error:
        detail = str(error)
        status = "legacy_unverified" if "legacy" in detail else "failed"
        return status, (detail,)
    return "verified", ()


def _attempt_lineage(attempt: Attempt) -> dict[str, str]:
    record = _read_json(attempt.path / "attempt.json")
    lineage = record.get("lineage")
    if lineage is None:
        raise ProjectStateError("legacy attempt has unverified project lineage")
    return _lineage_payload(lineage, "attempt")


def _acceptance_evidence(attempt: Attempt) -> dict[str, Any] | None:
    metadata_path = attempt.path / "render-metadata.json"
    qc_path = attempt.path / "qc.yaml"
    if not metadata_path.exists() and not qc_path.exists():
        return None
    if not metadata_path.is_file() or not qc_path.is_file():
        raise ProjectStateError(
            "accepted attempt requires render metadata and QC evidence together"
        )
    lineage = _attempt_lineage(attempt)
    source_manifest = _source_manifest_evidence(attempt, lineage)
    provenance = _attempt_artifact(attempt.path / "provenance.json", "provenance")
    render_metadata = _attempt_artifact(metadata_path, "render metadata")
    qc = _attempt_artifact(qc_path, "QC")
    output = _selected_video_evidence(attempt, metadata_path, qc_path)
    return {
        "lineage": lineage,
        "output": output,
        "provenance": provenance,
        "qc": qc,
        "render_metadata": render_metadata,
        "source_manifest": source_manifest,
    }


def _accepted_attempt_evidence(
    attempt: Attempt, expected_lineage: Mapping[str, Any]
) -> dict[str, Any]:
    persisted, decision_paths = _load_attempt(attempt.path)
    if persisted.state is not AttemptState.ACCEPTED:
        raise ProjectStateError("attempt is not accepted")
    lineage = _attempt_lineage(persisted)
    if lineage != _lineage_payload(expected_lineage, "expected project"):
        raise ProjectStateError(
            "attempt project lineage does not match the current project manifest"
        )
    source_manifest = _source_manifest_evidence(persisted, lineage)
    provenance_path = persisted.path / "provenance.json"
    provenance_payload = _read_json(provenance_path)
    if provenance_payload.get("source_manifest") != lineage["source_manifest_sha256"]:
        raise ProjectStateError("attempt provenance source-manifest hash changed")
    if _lineage_payload(provenance_payload.get("lineage"), "attempt provenance") != lineage:
        raise ProjectStateError("attempt provenance project lineage changed")
    provenance = _attempt_artifact(provenance_path, "provenance")
    metadata_path = persisted.path / "render-metadata.json"
    qc_path = persisted.path / "qc.yaml"
    render_metadata = _attempt_artifact(metadata_path, "render metadata")
    qc = _attempt_artifact(qc_path, "QC")
    output = _selected_video_evidence(persisted, metadata_path, qc_path)
    acceptances = [
        (path, _read_json(path))
        for path in decision_paths
        if _read_json(path).get("to") == AttemptState.ACCEPTED
    ]
    if len(acceptances) != 1:
        raise ProjectStateError("accepted attempt has no unambiguous terminal decision")
    acceptance_path, acceptance = acceptances[0]
    sealed = acceptance.get("accepted_evidence")
    if not isinstance(sealed, Mapping):
        raise ProjectStateError("legacy accepted attempt has unverified artifact evidence")
    current = {
        "lineage": lineage,
        "output": output,
        "provenance": provenance,
        "qc": qc,
        "render_metadata": render_metadata,
        "source_manifest": source_manifest,
    }
    if _json_ready(sealed) != current:
        raise ProjectStateError("accepted attempt immutable artifact evidence changed")
    seal = _read_json(persisted.path / "acceptance.json")
    if seal.get("attempt_id") != persisted.attempt_id:
        raise ProjectStateError("acceptance seal belongs to a different attempt")
    if _json_ready(seal.get("accepted_evidence")) != current:
        raise ProjectStateError("acceptance seal immutable artifact evidence changed")
    decision_evidence = _hashed_entry(
        seal.get("decision") if isinstance(seal.get("decision"), Mapping) else {},
        "terminal acceptance decision",
    )
    if Path(decision_evidence["path"]) != acceptance_path.resolve():
        raise ProjectStateError("acceptance seal references a different terminal decision")
    return {
        "acceptance_decision": decision_evidence,
        "attempt_id": persisted.attempt_id,
        "attempt_path": str(persisted.path.resolve()),
        **current,
        "segment_id": persisted.segment_id,
        "shot_id": persisted.shot_id,
    }


def _source_manifest_evidence(
    attempt: Attempt, lineage: Mapping[str, str]
) -> dict[str, str]:
    snapshots = tuple(attempt.path.glob("source-manifest.*"))
    if len(snapshots) != 1:
        raise ProjectStateError("attempt has no unambiguous source manifest snapshot")
    evidence = _attempt_artifact(snapshots[0], "source manifest")
    if evidence["sha256"] != lineage["source_manifest_sha256"]:
        raise ProjectStateError("attempt source-manifest project lineage changed")
    return evidence


def _selected_video_evidence(
    attempt: Attempt, metadata_path: Path, qc_path: Path
) -> dict[str, str]:
    metadata = _read_json(metadata_path)
    if metadata.get("attempt_id") != attempt.attempt_id:
        raise ProjectStateError("render metadata belongs to a different attempt")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ProjectStateError("accepted attempt has no render outputs")
    candidates = [outputs[name] for name in ("segment", "bridge") if name in outputs]
    if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ProjectStateError("accepted attempt has no unambiguous video output")
    output = _hashed_entry(candidates[0], "accepted attempt output")
    qc = _read_qc(qc_path)
    if qc.get("attempt_id") != attempt.attempt_id:
        raise ProjectStateError("QC record belongs to a different attempt")
    if _json_ready(qc.get("video")) != output:
        raise ProjectStateError("QC video evidence does not match render metadata")
    return output


def _attempt_artifact(path: Path, label: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ProjectStateError(f"attempt {label} does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _hashed_entry(entry: Mapping[str, Any], label: str) -> dict[str, str]:
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ProjectStateError(f"{label} provenance is invalid")
    path = Path(raw_path).resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ProjectStateError(f"{label} hash changed or is missing: {path}")
    return {"path": str(path), "sha256": expected_hash}


def _lineage_payload(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProjectStateError(f"{context} project lineage is invalid")
    project_id = value.get("project_id")
    manifest_hash = value.get("source_manifest_sha256")
    if not isinstance(project_id, str) or not project_id:
        raise ProjectStateError(f"{context} project lineage project_id is invalid")
    if not _is_sha256(manifest_hash):
        raise ProjectStateError(f"{context} source-manifest lineage hash is invalid")
    return {
        "project_id": project_id,
        "source_manifest_sha256": manifest_hash.casefold(),
    }


def create_assembly_record(
    project: ProjectConfig,
    scope: str,
    *,
    inputs: Sequence[Mapping[str, Any]],
    requested: Mapping[str, Any],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AssemblyRecord:
    """Create one append-only record without changing accepted render attempts."""
    _validate_identifier(scope, "assembly scope")
    if not inputs:
        raise ProjectStateError("assembly record requires at least one accepted input")
    if not isinstance(requested, Mapping):
        raise ProjectStateError("assembly record requested details must be a mapping")
    lineage = project_lineage(project)
    input_payload = _assembly_inputs_payload(inputs, lineage)
    _requested_boundary_approvals(requested)
    requested_payload = _json_ready(requested)
    created_at = _as_utc(now())
    root = _assembly_records_root(project) / scope
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in root.iterdir() if path.is_dir())
    record_number = len(existing) + 1
    timestamp = _utc_timestamp(created_at).replace(":", "").replace("-", "")
    while True:
        assembly_id = f"assembly-{record_number:04d}-{timestamp}"
        record_path = root / assembly_id
        try:
            record_path.mkdir()
            break
        except FileExistsError:
            record_number += 1

    record = AssemblyRecord(
        path=record_path,
        assembly_id=assembly_id,
        scope=scope,
        state=AssemblyState.PLANNED,
        created_at=created_at,
    )
    root_payload = {
        "assembly_id": record.assembly_id,
        "created_at": _utc_timestamp(created_at),
        "inputs": input_payload,
        "lineage": lineage,
        "requested": requested_payload,
        "scope": scope,
        "state": record.state,
    }
    root_payload["root_digest"] = _payload_sha256(root_payload)
    _write_json(record.path / "assembly.json", root_payload)
    return record


def transition_assembly_record(
    record: AssemblyRecord,
    target: AssemblyState,
    note: str | None,
    *,
    details: Mapping[str, Any] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AssemblyRecord:
    """Append a lifecycle decision to one immutable assembly record."""
    persisted, decision_paths = _load_assembly_record(record.path)
    if record.assembly_id != persisted.assembly_id:
        raise ProjectStateError("assembly record identity does not match its persisted record")
    if record.state is not persisted.state:
        raise ProjectStateError(
            f"stale assembly record state: supplied {record.state}, persisted {persisted.state}"
        )
    if target not in _ASSEMBLY_TRANSITIONS[persisted.state]:
        raise ProjectStateError(
            f"invalid assembly transition: {persisted.state} -> {target}"
        )
    if target is AssemblyState.FAILED and not _has_operator_note(note):
        raise ProjectStateError("failed assembly record requires a non-empty failure note")
    if _assembly_decision_paths(record.path) != decision_paths:
        raise ProjectStateError("assembly record decision chain changed before transition")
    if details is not None and not isinstance(details, Mapping):
        raise ProjectStateError("assembly record decision details must be a mapping")

    decision_dir = record.path / "decisions"
    decision_dir.mkdir(exist_ok=True)
    decision_number = len(decision_paths) + 1
    decision_path = decision_dir / f"{decision_number:04d}.json"
    root_payload = _read_json(record.path / "assembly.json")
    root_digest = _verified_assembly_root_digest(root_payload)
    if root_digest is None:
        raise ProjectStateError("legacy assembly record cannot append lifecycle decisions")
    payload = {
        "assembly_id": persisted.assembly_id,
        "details": _json_ready(details) if details is not None else None,
        "from": persisted.state,
        "note": note,
        "previous_decision_sha256": (
            sha256_file(decision_paths[-1]) if decision_paths else None
        ),
        "root_digest": root_digest,
        "timestamp": _utc_timestamp(_as_utc(now())),
        "to": target,
    }
    payload["decision_digest"] = _payload_sha256(payload)
    try:
        _write_json(decision_path, payload)
    except FileExistsError as error:
        current = load_assembly_record(record.path)
        raise ProjectStateError(
            f"assembly record decision claim lost; persisted state is {current.state}"
        ) from error
    return replace(persisted, state=target)


def load_assembly_record(path: Path) -> AssemblyRecord:
    """Reconstruct one immutable assembly lifecycle from its decision log."""
    record, _ = _load_assembly_record(path)
    return record


def assembly_records(project: ProjectConfig) -> tuple[AssemblyRecord, ...]:
    """Return every immutable assembly record associated with a project."""
    root = _assembly_records_root(project)
    if not root.is_dir():
        return ()
    records = [load_assembly_record(path.parent) for path in root.rglob("assembly.json")]
    return tuple(sorted(records, key=lambda record: str(record.path)))


def inspect_assembly_records(project: ProjectConfig) -> tuple[AssemblyRecordIntegrity, ...]:
    """Return record-level integrity results without hiding malformed sibling roots."""
    root = _assembly_records_root(project)
    if not root.is_dir():
        return ()
    results: list[AssemblyRecordIntegrity] = []
    for assembly_json in sorted(root.rglob("assembly.json"), key=lambda path: str(path)):
        path = assembly_json.parent
        try:
            record = load_assembly_record(path)
            results.append(verify_assembly_record_integrity(record, project))
        except (OSError, ProjectStateError) as error:
            results.append(
                AssemblyRecordIntegrity(
                    record=None,
                    path=path,
                    failures=(f"assembly root: {error}",),
                    status="failed",
                )
            )
    return tuple(results)


def write_assembly_record_json(
    record: AssemblyRecord,
    name: str,
    payload: Mapping[str, Any],
) -> Path:
    """Add one immutable JSON evidence artifact to an assembly record."""
    if not name or Path(name).name != name or not name.endswith(".json"):
        raise ProjectStateError("assembly record artifact name must be one JSON filename")
    if not isinstance(payload, Mapping):
        raise ProjectStateError("assembly record artifact payload must be a mapping")
    persisted = load_assembly_record(record.path)
    if persisted.assembly_id != record.assembly_id:
        raise ProjectStateError("assembly record artifact identity does not match")
    destination = record.path / name
    _write_json(destination, payload)
    return destination


def verify_assembly_record_inputs(
    record: AssemblyRecord, project: ProjectConfig | None = None
) -> tuple[dict[str, Any], ...]:
    """Recheck accepted source identity and hashes immediately before assembly work."""
    persisted = load_assembly_record(record.path)
    if persisted.assembly_id != record.assembly_id:
        raise ProjectStateError("assembly record input verification identity does not match")
    payload = _read_json(record.path / "assembly.json")
    root_digest = _verified_assembly_root_digest(payload)
    if root_digest is None:
        raise ProjectStateError("legacy assembly record has unverified root integrity")
    lineage = _lineage_payload(payload.get("lineage"), "assembly record")
    if project is not None and lineage != project_lineage(project):
        raise ProjectStateError(
            "assembly record project lineage does not match the current project manifest"
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ProjectStateError("assembly record requires an immutable input list")
    return tuple(_assembly_inputs_payload(inputs, lineage))


def recorded_boundary_approvals(record: AssemblyRecord) -> tuple[dict[str, Any], ...]:
    """Read canonical reviewed-boundary approvals from an immutable assembly record."""
    persisted = load_assembly_record(record.path)
    if persisted.assembly_id != record.assembly_id:
        raise ProjectStateError("assembly record approval verification identity does not match")
    payload = _read_json(record.path / "assembly.json")
    requested = payload.get("requested")
    if not isinstance(requested, Mapping):
        raise ProjectStateError("assembly record requested details must be a mapping")
    return _requested_boundary_approvals(requested)


def verify_assembly_record_integrity(
    record: AssemblyRecord, project: ProjectConfig | None = None
) -> AssemblyRecordIntegrity:
    """Rehash persisted assembly evidence without changing a lifecycle record."""
    try:
        persisted, decision_paths = _load_assembly_record(record.path)
    except (OSError, ProjectStateError) as error:
        return AssemblyRecordIntegrity(
            record,
            record.path,
            (f"assembly root or decision chain: {error}",),
            "failed",
        )
    if persisted.assembly_id != record.assembly_id:
        raise ProjectStateError("assembly record integrity identity does not match")
    failures: list[str] = []
    payload = _read_json(record.path / "assembly.json")
    root_digest = payload.get("root_digest")
    decisions = tuple(_read_json(path) for path in decision_paths)
    if root_digest is None:
        if any(
            any(
                key in decision
                for key in (
                    "decision_digest",
                    "previous_decision_sha256",
                    "root_digest",
                )
            )
            for decision in decisions
        ):
            return AssemblyRecordIntegrity(
                persisted,
                persisted.path,
                ("legacy assembly root has partial lifecycle integrity fields",),
                "failed",
            )
        return AssemblyRecordIntegrity(
            persisted,
            persisted.path,
            ("legacy assembly record has no root digest or decision chain",),
            "legacy_unverified",
        )
    _capture_integrity_failure(
        failures,
        "assembly root",
        lambda: _verified_assembly_root_digest(payload),
    )
    if project is not None:
        _capture_integrity_failure(
            failures,
            "project lineage",
            lambda: _verify_assembly_project_lineage(payload, project),
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        failures.append("source inputs: assembly record requires an immutable input list")
    else:
        _capture_integrity_failure(
            failures,
            "source inputs",
            lambda: _assembly_inputs_payload(
                inputs, _lineage_payload(payload.get("lineage"), "assembly record")
            ),
        )

    assembling = _assembly_transition(decisions, AssemblyState.ASSEMBLING)
    if assembling is not None:
        _capture_integrity_failure(
            failures,
            "assembly evidence",
            lambda: _verify_assembly_start_evidence(record, assembling),
        )

    assembled = _assembly_transition(decisions, AssemblyState.ASSEMBLED)
    if assembled is not None:
        _capture_integrity_failure(
            failures,
            "final outputs",
            lambda: _verify_assembly_output_evidence(record, assembled),
        )

    final = _assembly_transition(decisions, AssemblyState.FINAL)
    if final is not None:
        _capture_integrity_failure(
            failures,
            "finalized evidence",
            lambda: _verify_assembly_output_evidence(record, final),
        )
    return AssemblyRecordIntegrity(
        persisted,
        persisted.path,
        tuple(failures),
        "failed" if failures else "verified",
    )


def _attempts_root(project: ProjectConfig) -> Path:
    configured = project.source.get("attempts_dir", project.path.parent / "attempts")
    root = Path(configured)
    if not root.is_absolute():
        root = project.path.parent / root
    return root


def _assembly_records_root(project: ProjectConfig) -> Path:
    configured = project.source.get("assembly_records_dir", project.path.parent / "assembly-records")
    if not isinstance(configured, (str, Path)):
        raise ProjectStateError("assembly_records_dir must be a local path")
    root = Path(configured)
    if not root.is_absolute():
        root = project.path.parent / root
    return root


def _assembly_inputs_payload(
    inputs: Sequence[Mapping[str, Any]],
    expected_lineage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lineage = _lineage_payload(expected_lineage, "assembly input")
    payload: list[dict[str, Any]] = []
    for entry in inputs:
        if not isinstance(entry, Mapping):
            raise ProjectStateError("assembly record input must be a mapping")
        attempt_path_raw = entry.get("attempt_path")
        if not isinstance(attempt_path_raw, (str, Path)):
            raise ProjectStateError("assembly record input requires attempt_path")
        attempt = load_attempt(Path(attempt_path_raw))
        if attempt.state is not AttemptState.ACCEPTED:
            raise ProjectStateError("assembly record input attempt is not accepted")
        if entry.get("attempt_id") != attempt.attempt_id:
            raise ProjectStateError("assembly record input attempt_id does not match")
        if entry.get("shot_id") != attempt.shot_id or entry.get("segment_id") != attempt.segment_id:
            raise ProjectStateError("assembly record input identity does not match")
        canonical = _accepted_attempt_evidence(attempt, lineage)
        if _json_ready(entry) != canonical:
            raise ProjectStateError(
                "assembly record input immutable evidence does not match the accepted attempt"
            )
        payload.append(canonical)
    return payload


def _verify_assembly_project_lineage(
    payload: Mapping[str, Any], project: ProjectConfig
) -> None:
    if _lineage_payload(payload.get("lineage"), "assembly record") != project_lineage(
        project
    ):
        raise ProjectStateError(
            "assembly record project lineage does not match the current project manifest"
        )


def _requested_boundary_approvals(requested: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_approvals = requested.get("boundary_approvals", [])
    if not isinstance(raw_approvals, list):
        raise ProjectStateError("assembly requested boundary approvals must be a list")
    approvals: list[dict[str, Any]] = []
    indexes: set[int] = set()
    for raw_approval in raw_approvals:
        if not isinstance(raw_approval, Mapping):
            raise ProjectStateError("assembly requested boundary approval must be a mapping")
        boundary_index = raw_approval.get("boundary_index")
        note = raw_approval.get("note")
        if not isinstance(boundary_index, int) or isinstance(boundary_index, bool) or boundary_index < 1:
            raise ProjectStateError("assembly requested boundary approval index must be positive")
        if not isinstance(note, str) or not note or note != note.strip():
            raise ProjectStateError(
                "assembly requested boundary approval note must be stripped and non-empty"
            )
        if boundary_index in indexes:
            raise ProjectStateError("assembly requested boundary approvals must be unique")
        indexes.add(boundary_index)
        approvals.append({"boundary_index": boundary_index, "note": note})
    if [approval["boundary_index"] for approval in approvals] != sorted(indexes):
        raise ProjectStateError("assembly requested boundary approvals must be sorted")
    return tuple(approvals)


def _capture_integrity_failure(
    failures: list[str],
    label: str,
    check: Callable[[], Any],
) -> None:
    try:
        check()
    except (OSError, ProjectStateError) as error:
        failures.append(f"{label}: {error}")


def _assembly_transition(
    decisions: Sequence[Mapping[str, Any]], target: AssemblyState
) -> Mapping[str, Any] | None:
    matches = [decision for decision in decisions if decision.get("to") == target]
    if len(matches) > 1:
        raise ProjectStateError(f"assembly record has multiple {target} transitions")
    return matches[0] if matches else None


def _verify_assembly_start_evidence(
    record: AssemblyRecord, decision: Mapping[str, Any]
) -> None:
    evidence = _assembly_evidence(decision, AssemblyState.ASSEMBLING)
    _verify_record_artifact_evidence(record, evidence, "assembly_json", "assembly.json")
    _verify_record_artifact_evidence(record, evidence, "assembly_plan", "assembly-plan.json")
    plan = _read_json(record.path / "assembly-plan.json")
    if not isinstance(plan.get("targets"), Mapping):
        raise ProjectStateError("assembly plan has no target mapping")
    record_payload = _read_json(record.path / "assembly.json")
    _verify_boundary_approval_evidence(record_payload, plan)


def _verify_assembly_output_evidence(
    record: AssemblyRecord, decision: Mapping[str, Any]
) -> None:
    evidence = _assembly_evidence(decision, _assembly_state(decision.get("to"), "assembly decision target state"))
    _verify_record_artifact_evidence(record, evidence, "assembly_plan", "assembly-plan.json")
    _verify_record_artifact_evidence(record, evidence, "outputs_json", "outputs.json")
    plan = _read_json(record.path / "assembly-plan.json")
    targets = plan.get("targets")
    if not isinstance(targets, Mapping):
        raise ProjectStateError("assembly plan has no target mapping")
    outputs_payload = _read_json(record.path / "outputs.json")
    final_outputs = outputs_payload.get("outputs")
    if not isinstance(final_outputs, Mapping) or not final_outputs:
        raise ProjectStateError("assembly output evidence has no output mapping")
    if evidence.get("final_outputs") != final_outputs:
        raise ProjectStateError("assembly final output hashes do not match transition evidence")
    boundary = outputs_payload.get("boundary_decisions")
    if evidence.get("boundary_decisions") != boundary:
        raise ProjectStateError("assembly boundary evidence does not match transition evidence")
    _verify_hashed_path(boundary, "assembly boundary decisions")
    decision_log = plan.get("decision_log")
    if not isinstance(decision_log, str) or Path(decision_log).resolve() != Path(
        boundary["path"]
    ).resolve():
        raise ProjectStateError("assembly boundary decisions do not match the serialized plan")
    decision_payload = _read_json(Path(boundary["path"]))
    if decision_payload.get("boundary_approvals", []) != plan.get("boundary_approvals", []):
        raise ProjectStateError("assembly boundary approvals do not match the serialized plan")
    _verify_boundary_decision_log(plan, decision_payload)
    for name in ("review_mp4", "edit_master_ffv1", "edit_master_prores"):
        target = targets.get(name)
        output = final_outputs.get(name)
        if not isinstance(target, str):
            raise ProjectStateError(f"assembly plan has no {name} target")
        _verify_hashed_path(output, f"assembly {name}")
        if Path(target).resolve() != Path(output["path"]).resolve():
            raise ProjectStateError(f"assembly {name} does not match the serialized plan")
    rife_target = targets.get("rife_review_mp4")
    rife_output = final_outputs.get("rife_review_mp4")
    if rife_output is not None:
        if not isinstance(rife_target, str):
            raise ProjectStateError("assembly RIFE output has no serialized target")
        _verify_hashed_path(rife_output, "assembly rife_review_mp4")
        if Path(rife_target).resolve() != Path(rife_output["path"]).resolve():
            raise ProjectStateError("assembly RIFE output does not match the serialized plan")


def _assembly_evidence(
    decision: Mapping[str, Any], target: AssemblyState
) -> Mapping[str, Any]:
    details = decision.get("details")
    if not isinstance(details, Mapping):
        raise ProjectStateError(f"assembly {target} transition has no evidence details")
    evidence = details.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ProjectStateError(f"assembly {target} transition has no evidence mapping")
    return evidence


def _verify_record_artifact_evidence(
    record: AssemblyRecord,
    evidence: Mapping[str, Any],
    key: str,
    filename: str,
) -> None:
    entry = evidence.get(key)
    _verify_hashed_path(entry, f"assembly {key}")
    if Path(entry["path"]).resolve() != (record.path / filename).resolve():
        raise ProjectStateError(f"assembly {key} must reference {filename}")


def _verify_boundary_approval_evidence(
    record_payload: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    requested = record_payload.get("requested")
    if not isinstance(requested, Mapping):
        raise ProjectStateError("assembly record requested details must be a mapping")
    approvals = _requested_boundary_approvals(requested)
    plan_approvals = plan.get("boundary_approvals", [])
    if not isinstance(plan_approvals, list):
        raise ProjectStateError("assembly plan boundary approvals must be a list")
    if len(plan_approvals) != len(approvals):
        raise ProjectStateError("assembly plan boundary approvals do not match the record")
    inputs = record_payload.get("inputs")
    decisions = plan.get("boundary_decisions")
    if not isinstance(inputs, list) or not isinstance(decisions, list):
        raise ProjectStateError("assembly plan approval evidence is missing source boundaries")
    for requested_approval, evidence in zip(approvals, plan_approvals):
        if not isinstance(evidence, Mapping):
            raise ProjectStateError("assembly plan boundary approval evidence must be a mapping")
        index = requested_approval["boundary_index"]
        if evidence.get("boundary_index") != index or evidence.get("note") != requested_approval["note"]:
            raise ProjectStateError("assembly plan boundary approval does not match the record")
        if index > len(decisions) or index >= len(inputs):
            raise ProjectStateError("assembly plan boundary approval index is out of range")
        decision = decisions[index - 1]
        diagnostic = evidence.get("diagnostic")
        if not isinstance(decision, Mapping) or not isinstance(diagnostic, Mapping):
            raise ProjectStateError("assembly plan boundary approval diagnostic evidence is invalid")
        if decision.get("requires_review") is not True or decision.get("trim_right_frames") != 0:
            raise ProjectStateError(
                "assembly plan reviewed boundary approval must preserve review and zero trim"
            )
        if diagnostic != {
            "left_hash": decision.get("left_hash"),
            "right_hash": decision.get("right_hash"),
        }:
            raise ProjectStateError("assembly plan boundary approval diagnostic hashes do not match")
        if evidence.get("left_source") != inputs[index - 1] or evidence.get("right_source") != inputs[index]:
            raise ProjectStateError("assembly plan boundary approval sources do not match the record")


def _verify_boundary_decision_log(
    plan: Mapping[str, Any], decision_payload: Mapping[str, Any]
) -> None:
    approvals = plan.get("boundary_approvals", [])
    decisions = decision_payload.get("decisions")
    planned_decisions = plan.get("boundary_decisions")
    if not isinstance(approvals, list) or not isinstance(decisions, list) or not isinstance(
        planned_decisions, list
    ):
        raise ProjectStateError("assembly boundary decision evidence is invalid")
    for approval in approvals:
        if not isinstance(approval, Mapping) or not isinstance(
            index := approval.get("boundary_index"), int
        ) or isinstance(index, bool):
            raise ProjectStateError("assembly boundary approval index is invalid")
        if index < 1 or index > len(decisions) or index > len(planned_decisions):
            raise ProjectStateError("assembly boundary decision approval index is out of range")
        if decisions[index - 1] != planned_decisions[index - 1]:
            raise ProjectStateError("assembly boundary decision does not match the serialized plan")


def _verify_hashed_path(entry: Any, label: str) -> None:
    if not isinstance(entry, Mapping):
        raise ProjectStateError(f"{label} hash evidence is invalid")
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ProjectStateError(f"{label} hash evidence is invalid")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ProjectStateError(f"{label} does not exist: {path}")
    if sha256_file(path) != expected_hash:
        raise ProjectStateError(f"{label} hash changed: {path}")


def _write_workflow_snapshot(destination: Path, project: ProjectConfig) -> None:
    workflow = project.source.get("workflow_api", project.source.get("workflow", {}))
    if isinstance(workflow, (str, Path)):
        source = Path(workflow)
        if not source.is_absolute():
            source = project.path.parent / source
        copyfile(source, destination)
        return
    _write_json(destination, workflow)


def _request_payload(project: ProjectConfig) -> Any:
    return project.source.get("request", project.source.get("request_payload", {}))


def _input_paths(project: ProjectConfig) -> dict[str, Path]:
    raw_inputs = project.source.get("inputs", {})
    if not isinstance(raw_inputs, Mapping):
        raise ProjectStateError("inputs must be a mapping of names to local paths")
    paths: dict[str, Path] = {}
    for name, value in raw_inputs.items():
        if not isinstance(name, str):
            raise ProjectStateError("input names must be strings")
        if isinstance(value, Mapping):
            value = value.get("path")
        if not isinstance(value, (str, Path)):
            raise ProjectStateError(f"input {name} must name a local file")
        path = Path(value)
        if not path.is_absolute():
            path = project.path.parent / path
        paths[name] = path
    return paths


def _selected_inputs_payload(
    selected_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name, entry in selected_inputs.items():
        if not isinstance(name, str) or not name:
            raise ProjectStateError("selected input names must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ProjectStateError(f"selected input {name} must be a mapping")
        raw_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(raw_path, (str, Path)) or not isinstance(expected_hash, str):
            raise ProjectStateError(f"selected input {name} has invalid provenance")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ProjectStateError(f"selected input does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ProjectStateError(f"selected input hash changed: {path}")
        copied = {str(key): _json_ready(value) for key, value in entry.items()}
        copied["path"] = str(path)
        copied["sha256"] = actual_hash
        payload[name] = copied
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(_json_ready(payload), destination, indent=2, sort_keys=True)
        destination.write("\n")


def _load_attempt(path: Path) -> tuple[Attempt, tuple[Path, ...]]:
    record = _read_json(path / "attempt.json")
    attempt_id = _required_string(record, "attempt_id", "attempt record")
    shot_id = _required_string(record, "shot_id", "attempt record")
    segment_id = _required_string(record, "segment_id", "attempt record")
    initial_state = _attempt_state(record.get("state"), "attempt record state")
    if initial_state is not AttemptState.PLANNED:
        raise ProjectStateError("attempt record must begin in planned state")
    created_at = _timestamp(record.get("created_at"), "attempt record created_at")
    parent_raw = record.get("parent_attempt")
    if parent_raw is not None and not isinstance(parent_raw, str):
        raise ProjectStateError("attempt record parent_attempt must be a string or null")
    attempt = Attempt(
        path=path,
        attempt_id=attempt_id,
        shot_id=shot_id,
        segment_id=segment_id,
        state=AttemptState.PLANNED,
        created_at=created_at,
        parent_attempt=Path(parent_raw) if parent_raw else None,
    )
    decision_paths = _decision_paths(path)
    state = attempt.state
    for expected_number, decision_path in enumerate(decision_paths, start=1):
        if _decision_number(decision_path) != expected_number:
            raise ProjectStateError("attempt decision log is not sequential")
        decision = _read_json(decision_path)
        if decision.get("attempt_id") != attempt_id:
            raise ProjectStateError("attempt decision belongs to a different attempt")
        source = _attempt_state(decision.get("from"), "decision from state")
        target = _attempt_state(decision.get("to"), "decision target state")
        if source is not state:
            raise ProjectStateError("attempt decision log has a stale source state")
        if target not in _TRANSITIONS[state]:
            raise ProjectStateError(f"invalid persisted attempt transition: {state} -> {target}")
        if target in _OPERATOR_OUTCOMES and not _has_operator_note(decision.get("note")):
            raise ProjectStateError(f"persisted {target} decision lacks a non-empty operator note")
        _timestamp(decision.get("timestamp"), "decision timestamp")
        state = target
    return replace(attempt, state=state), decision_paths


def _load_assembly_record(path: Path) -> tuple[AssemblyRecord, tuple[Path, ...]]:
    payload = _read_json(path / "assembly.json")
    assembly_id = _required_string(payload, "assembly_id", "assembly record")
    scope = _required_string(payload, "scope", "assembly record")
    _validate_identifier(scope, "assembly record scope")
    initial_state = _assembly_state(payload.get("state"), "assembly record state")
    if initial_state is not AssemblyState.PLANNED:
        raise ProjectStateError("assembly record must begin in planned state")
    created_at = _timestamp(payload.get("created_at"), "assembly record created_at")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ProjectStateError("assembly record requires an immutable input list")
    if not isinstance(payload.get("requested"), Mapping):
        raise ProjectStateError("assembly record requested details must be a mapping")
    record = AssemblyRecord(
        path=path,
        assembly_id=assembly_id,
        scope=scope,
        state=AssemblyState.PLANNED,
        created_at=created_at,
    )
    decision_paths = _assembly_decision_paths(path)
    root_digest = _verified_assembly_root_digest(payload)
    state = record.state
    previous_decision_hash: str | None = None
    for expected_number, decision_path in enumerate(decision_paths, start=1):
        if _decision_number(decision_path) != expected_number:
            raise ProjectStateError("assembly record decision log is not sequential")
        decision = _read_json(decision_path)
        _verify_assembly_decision_linkage(
            decision,
            root_digest=root_digest,
            previous_decision_hash=previous_decision_hash,
        )
        if decision.get("assembly_id") != assembly_id:
            raise ProjectStateError("assembly record decision belongs to a different record")
        source = _assembly_state(decision.get("from"), "assembly decision from state")
        target = _assembly_state(decision.get("to"), "assembly decision target state")
        if source is not state:
            raise ProjectStateError("assembly record decision log has a stale source state")
        if target not in _ASSEMBLY_TRANSITIONS[state]:
            raise ProjectStateError(
                f"invalid persisted assembly transition: {state} -> {target}"
            )
        if target is AssemblyState.FAILED and not _has_operator_note(decision.get("note")):
            raise ProjectStateError("persisted failed assembly decision lacks a non-empty failure note")
        if decision.get("details") is not None and not isinstance(decision.get("details"), Mapping):
            raise ProjectStateError("assembly record decision details must be a mapping or null")
        _timestamp(decision.get("timestamp"), "assembly decision timestamp")
        state = target
        previous_decision_hash = sha256_file(decision_path)
    return replace(record, state=state), decision_paths


def _verified_assembly_root_digest(payload: Mapping[str, Any]) -> str | None:
    root_digest = payload.get("root_digest")
    if root_digest is None:
        return None
    if not _is_sha256(root_digest):
        raise ProjectStateError("assembly root digest is invalid")
    if _payload_sha256(payload, omit="root_digest") != root_digest.casefold():
        raise ProjectStateError("assembly root digest changed")
    _lineage_payload(payload.get("lineage"), "assembly record")
    return root_digest.casefold()


def _verify_assembly_decision_linkage(
    decision: Mapping[str, Any],
    *,
    root_digest: str | None,
    previous_decision_hash: str | None,
) -> None:
    linkage_keys = {
        "decision_digest",
        "previous_decision_sha256",
        "root_digest",
    }
    if root_digest is None:
        if any(key in decision for key in linkage_keys):
            raise ProjectStateError(
                "legacy assembly record has partial lifecycle integrity fields"
            )
        return
    if decision.get("root_digest") != root_digest:
        raise ProjectStateError("assembly decision root linkage changed")
    if decision.get("previous_decision_sha256") != previous_decision_hash:
        raise ProjectStateError("assembly predecessor decision linkage changed")
    decision_digest = decision.get("decision_digest")
    if not _is_sha256(decision_digest):
        raise ProjectStateError("assembly decision digest is invalid")
    if _payload_sha256(decision, omit="decision_digest") != decision_digest.casefold():
        raise ProjectStateError("assembly decision digest changed")


def _payload_sha256(payload: Mapping[str, Any], *, omit: str | None = None) -> str:
    canonical = {
        str(key): _json_ready(value)
        for key, value in payload.items()
        if key != omit
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _decision_paths(attempt_path: Path) -> tuple[Path, ...]:
    decision_dir = attempt_path / "decisions"
    if not decision_dir.exists():
        return ()
    if not decision_dir.is_dir():
        raise ProjectStateError("attempt decisions path is not a directory")
    return tuple(sorted(decision_dir.glob("*.json"), key=lambda path: path.name))


def _assembly_decision_paths(record_path: Path) -> tuple[Path, ...]:
    decision_dir = record_path / "decisions"
    if not decision_dir.exists():
        return ()
    if not decision_dir.is_dir():
        raise ProjectStateError("assembly record decisions path is not a directory")
    return tuple(sorted(decision_dir.glob("*.json"), key=lambda path: path.name))


def _decision_number(path: Path) -> int:
    prefix = path.stem.partition("-")[0]
    if not prefix.isdigit():
        raise ProjectStateError("attempt decision filename is invalid")
    return int(prefix)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectStateError(f"invalid attempt record: {path}") from error
    if not isinstance(value, Mapping):
        raise ProjectStateError(f"attempt record must be a JSON object: {path}")
    return value


def _read_qc(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ProjectStateError(f"invalid QC record: {path}") from error
    if not isinstance(value, Mapping):
        raise ProjectStateError(f"QC record must be a mapping: {path}")
    return value


def _required_string(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ProjectStateError(f"{context} {key} must be a non-empty string")
    return value


def _attempt_state(value: Any, context: str) -> AttemptState:
    try:
        return AttemptState(value)
    except (TypeError, ValueError) as error:
        raise ProjectStateError(f"{context} is invalid") from error


def _assembly_state(value: Any, context: str) -> AssemblyState:
    try:
        return AssemblyState(value)
    except (TypeError, ValueError) as error:
        raise ProjectStateError(f"{context} is invalid") from error


def _timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProjectStateError(f"{context} must be a UTC timestamp")
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as error:
        raise ProjectStateError(f"{context} must be a UTC timestamp") from error


def _has_operator_note(note: Any) -> bool:
    return isinstance(note, str) and bool(note.strip())


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return str(value)
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return value.astimezone(UTC)


def _utc_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_identifier(value: str, name: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ProjectStateError(f"{name} must be a single path component")
