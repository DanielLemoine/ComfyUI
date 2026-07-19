from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Literal


class FrameError(RuntimeError):
    """Raised when deterministic candidate-frame extraction cannot complete."""


def candidate_start_index(*, frame_count: int, count: int, where: str) -> int:
    if count <= 0:
        raise FrameError("candidate count must be positive")
    if frame_count < count:
        raise FrameError("video is shorter than requested candidate count")
    if where not in {"head", "tail"}:
        raise FrameError("candidate location must be head or tail")
    return 0 if where == "head" else frame_count - count


def extract_candidate_frames(
    video: Path,
    count: int,
    where: Literal["head", "tail"],
    destination: Path,
) -> list[Path]:
    if not video.is_file():
        raise FrameError(f"rendered video does not exist: {video}")
    frame_count = probe_frame_count(video)
    start = candidate_start_index(frame_count=frame_count, count=count, where=where)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for frame_index in range(start, start + count):
        output = destination / f"frame-{frame_index:06d}.png"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{frame_index})",
                "-frames:v",
                "1",
                "-fps_mode",
                "vfr",
                str(output),
            ],
            "extract candidate frame",
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise FrameError(f"FFmpeg did not produce candidate frame {frame_index}")
        outputs.append(output)
    return outputs


def create_contact_sheet(frames: list[Path], destination: Path) -> Path:
    if not frames:
        raise FrameError("contact sheet requires at least one frame")
    missing = [path for path in frames if not path.is_file()]
    if missing:
        raise FrameError(f"contact sheet frame does not exist: {missing[0]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = min(5, len(frames))
    rows = math.ceil(len(frames) / columns)
    layout = "|".join(
        f"{_multiple(column, 'w0')}_{_multiple(row, 'h0')}"
        for row in range(rows)
        for column in range(columns)
        if row * columns + column < len(frames)
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n"]
    for frame in frames:
        command.extend(["-i", str(frame)])
    command.extend(
        [
            "-filter_complex",
            f"xstack=inputs={len(frames)}:layout={layout}:fill=black",
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    _run(command, "create contact sheet")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FrameError("FFmpeg did not produce the contact sheet")
    return destination


def probe_frame_count(video: Path) -> int:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        "count video frames",
    )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        raw_count = stream.get("nb_read_frames", stream.get("nb_frames"))
        frame_count = int(raw_count)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise FrameError(f"FFprobe returned no usable frame count for {video}") from error
    if frame_count <= 0:
        raise FrameError(f"video has no frames: {video}")
    return frame_count


def _multiple(value: int, unit: str) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return unit
    return f"{value}*{unit}"


def _run(command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise FrameError(f"{command[0]} is required to {operation}") from error
    except subprocess.TimeoutExpired as error:
        raise FrameError(f"Timed out while attempting to {operation}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or f"exit {error.returncode}"
        raise FrameError(f"Unable to {operation}: {detail}") from error
