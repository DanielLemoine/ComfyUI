from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path


def write_text(
    destination: Path,
    content: str,
    *,
    replace_existing: bool = False,
) -> Path:
    """Flush a complete text artifact before making its final path visible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace_existing:
        raise FileExistsError(destination)
    staging = _staging_path(destination)
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(staging, destination)
        else:
            # Linking the fully flushed staging file claims a new destination
            # without the check-to-replace overwrite race.
            os.link(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    return destination


def copy_file(
    source: Path,
    destination: Path,
    *,
    replace_existing: bool = False,
) -> Path:
    """Flush a complete copied artifact before making its final path visible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace_existing:
        raise FileExistsError(destination)
    staging = _staging_path(destination)
    try:
        with source.open("rb") as input_stream, staging.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if replace_existing:
            os.replace(staging, destination)
        else:
            os.link(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    return destination


def write_bytes(
    destination: Path,
    content: bytes,
    *,
    replace_existing: bool = False,
) -> Path:
    """Flush a complete byte payload before making its final path visible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace_existing:
        raise FileExistsError(destination)
    staging = _staging_path(destination)
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(staging, destination)
        else:
            os.link(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    return destination


def _staging_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.staging"
    )
