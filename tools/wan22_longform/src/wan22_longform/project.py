from __future__ import annotations

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
    decision_path = decision_dir / f"{decision_number:04d}.json"
    try:
        _write_json(decision_path, payload)
    except FileExistsError as error:
        current = load_attempt(attempt.path)
        raise ProjectStateError(
            f"attempt decision claim lost; persisted state is {current.state}"
        ) from error
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
    input_payload = _assembly_inputs_payload(inputs)
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
    _write_json(
        record.path / "assembly.json",
        {
            "assembly_id": record.assembly_id,
            "created_at": _utc_timestamp(created_at),
            "inputs": input_payload,
            "requested": requested_payload,
            "scope": scope,
            "state": record.state,
        },
    )
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
    payload = {
        "assembly_id": persisted.assembly_id,
        "details": _json_ready(details) if details is not None else None,
        "from": persisted.state,
        "note": note,
        "timestamp": _utc_timestamp(_as_utc(now())),
        "to": target,
    }
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


def verify_assembly_record_inputs(record: AssemblyRecord) -> tuple[dict[str, Any], ...]:
    """Recheck accepted source identity and hashes immediately before assembly work."""
    persisted = load_assembly_record(record.path)
    if persisted.assembly_id != record.assembly_id:
        raise ProjectStateError("assembly record input verification identity does not match")
    payload = _read_json(record.path / "assembly.json")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ProjectStateError("assembly record requires an immutable input list")
    return tuple(_assembly_inputs_payload(inputs))


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
) -> list[dict[str, Any]]:
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
        output = entry.get("output")
        if not isinstance(output, Mapping):
            raise ProjectStateError("assembly record input requires output provenance")
        raw_output_path = output.get("path")
        expected_hash = output.get("sha256")
        if not isinstance(raw_output_path, (str, Path)) or not isinstance(expected_hash, str):
            raise ProjectStateError("assembly record input output provenance is invalid")
        output_path = Path(raw_output_path).resolve()
        if not output_path.is_file():
            raise ProjectStateError(f"assembly record input output does not exist: {output_path}")
        actual_hash = sha256_file(output_path)
        if actual_hash != expected_hash:
            raise ProjectStateError(
                f"assembly record input output hash changed: {output_path}"
            )
        payload.append(
            {
                "attempt_id": attempt.attempt_id,
                "attempt_path": str(attempt.path.resolve()),
                "output": {"path": str(output_path), "sha256": actual_hash},
                "segment_id": attempt.segment_id,
                "shot_id": attempt.shot_id,
            }
        )
    return payload


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
    state = record.state
    for expected_number, decision_path in enumerate(decision_paths, start=1):
        if _decision_number(decision_path) != expected_number:
            raise ProjectStateError("assembly record decision log is not sequential")
        decision = _read_json(decision_path)
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
    return replace(record, state=state), decision_paths


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
    except (OSError, json.JSONDecodeError) as error:
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
