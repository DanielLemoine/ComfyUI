from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import sha256_file
from .project import Attempt


class QcError(ValueError):
    """Raised when a QC record is invalid or cannot be preserved."""


def initialize_qc(
    attempt: Attempt,
    *,
    video: Path,
    head_frames: list[Path],
    tail_frames: list[Path],
    contact_sheet: Path,
    automatic_continuation_authorized: bool,
) -> Path:
    path = attempt.path / "qc.yaml"
    payload = {
        "attempt_id": attempt.attempt_id,
        "automatic_continuation_authorized": automatic_continuation_authorized,
        "candidate_frames": {
            "head": [_artifact(frame) for frame in head_frames],
            "tail": [_artifact(frame) for frame in tail_frames],
        },
        "contact_sheet": _artifact(contact_sheet),
        "operator_note": None,
        "selected_continuation_frame": None,
        "status": "needs_review",
        "video": _artifact(video),
    }
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        yaml.safe_dump(payload, destination, sort_keys=True)
    return path


def read_qc(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise QcError(f"invalid QC record: {path}") from error
    if not isinstance(payload, Mapping):
        raise QcError(f"QC record must be a mapping: {path}")
    return dict(payload)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}
