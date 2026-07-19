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
    if target not in _TRANSITIONS[attempt.state]:
        raise ProjectStateError(f"invalid attempt transition: {attempt.state} -> {target}")
    decision_dir = attempt.path / "decisions"
    decision_dir.mkdir(exist_ok=True)
    decision_number = len(list(decision_dir.glob("*.json"))) + 1
    timestamp = _as_utc(now())
    continuation = None
    if selected_continuation_frame is not None:
        continuation = {
            "path": str(selected_continuation_frame),
            "sha256": sha256_file(selected_continuation_frame),
        }
    payload = {
        "attempt_id": attempt.attempt_id,
        "from": attempt.state,
        "note": note,
        "parent_attempt": str(attempt.parent_attempt) if attempt.parent_attempt else None,
        "retry_of": str(attempt.parent_attempt) if attempt.parent_attempt else None,
        "selected_continuation_frame": continuation,
        "timestamp": _utc_timestamp(timestamp),
        "to": target,
    }
    while True:
        decision_path = decision_dir / f"{decision_number:04d}-{target}.json"
        try:
            _write_json(decision_path, payload)
            break
        except FileExistsError:
            decision_number += 1
    return replace(attempt, state=target)


def needs_render(attempt: Attempt) -> bool:
    """Only a planned attempt needs a render request; accepted attempts never do."""
    return attempt.state is AttemptState.PLANNED


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
