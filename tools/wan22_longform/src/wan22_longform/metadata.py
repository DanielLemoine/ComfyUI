from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .atomic import write_text
from .hashing import sha256_file
from .project import Attempt


@dataclass(frozen=True)
class RenderMetadata:
    outputs: Mapping[str, Path]
    rendered_at: datetime | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def write_metadata(
    attempt: Attempt, metadata: RenderMetadata, *, replace_invalid: bool = False
) -> Path:
    """Persist one immutable record of the files produced by an attempt."""
    path = attempt.path / "render-metadata.json"
    payload = {
        "attempt_id": attempt.attempt_id,
        "rendered_at": _utc_timestamp(metadata.rendered_at) if metadata.rendered_at else None,
        "outputs": {
            name: {"path": str(output), "sha256": sha256_file(output)}
            for name, output in metadata.outputs.items()
        },
        "details": _json_ready(metadata.details),
    }
    return write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        replace_existing=replace_invalid,
    )


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
