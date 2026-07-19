from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Mapping

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
    ASSEMBLING = "assembling"
    ASSEMBLED = "assembled"
    FINAL = "final"


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


_TRANSITIONS: Mapping[AttemptState, frozenset[AttemptState]] = {
    AttemptState.PLANNED: frozenset({AttemptState.RENDERING}),
    AttemptState.RENDERING: frozenset({AttemptState.RENDERED}),
    AttemptState.RENDERED: frozenset({AttemptState.NEEDS_REVIEW}),
    AttemptState.NEEDS_REVIEW: frozenset(
        {AttemptState.ACCEPTED, AttemptState.REJECTED, AttemptState.RETRY_REQUESTED}
    ),
    AttemptState.ACCEPTED: frozenset({AttemptState.ASSEMBLING}),
    AttemptState.REJECTED: frozenset(),
    AttemptState.RETRY_REQUESTED: frozenset(),
    AttemptState.ASSEMBLING: frozenset({AttemptState.ASSEMBLED}),
    AttemptState.ASSEMBLED: frozenset({AttemptState.FINAL}),
    AttemptState.FINAL: frozenset(),
}

_OPERATOR_OUTCOMES = frozenset(
    {AttemptState.ACCEPTED, AttemptState.REJECTED, AttemptState.RETRY_REQUESTED}
)


def create_attempt(
    project: ProjectConfig,
    shot_id: str,
    segment_id: str,
    *,
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
    _write_json(
        attempt_path / "provenance.json",
        {
            "inputs": {name: sha256_file(path) for name, path in _input_paths(project).items()},
            "request": sha256_file(request_snapshot),
            "source_manifest": sha256_file(manifest_snapshot),
            "workflow": sha256_file(workflow_snapshot),
        },
    )
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


def _attempts_root(project: ProjectConfig) -> Path:
    configured = project.source.get("attempts_dir", project.path.parent / "attempts")
    root = Path(configured)
    if not root.is_absolute():
        root = project.path.parent / root
    return root


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


def _decision_paths(attempt_path: Path) -> tuple[Path, ...]:
    decision_dir = attempt_path / "decisions"
    if not decision_dir.exists():
        return ()
    if not decision_dir.is_dir():
        raise ProjectStateError("attempt decisions path is not a directory")
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
