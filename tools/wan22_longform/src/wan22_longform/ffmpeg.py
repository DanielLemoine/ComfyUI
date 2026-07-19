from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


class FfmpegError(RuntimeError):
    """Raised when a local FFmpeg tool cannot safely inspect media."""


@dataclass(frozen=True)
class AudioSpec:
    codec: str
    sample_rate: int | None
    channels: int | None


@dataclass(frozen=True)
class MediaSpec:
    path: Path
    width: int
    height: int
    fps: Fraction
    time_base: Fraction
    pixel_format: str
    codec: str
    profile: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    audio: AudioSpec | None
    frame_count: int


def run_ffmpeg(
    args: Sequence[str],
    *,
    ffmpeg: Path = Path("ffmpeg"),
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg with argument vectors and refuse to overwrite any output."""
    return _run_media_tool(ffmpeg, args, cwd=cwd, no_overwrite=True)


def probe_media(path: Path, *, ffprobe: Path = Path("ffprobe")) -> MediaSpec:
    """Return the complete first-video-stream compatibility record for local media."""
    if not path.is_file():
        raise FfmpegError(f"media does not exist: {path}")
    completed = _run_media_tool(
        ffprobe,
        [
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
    )
    try:
        streams = json.loads(completed.stdout)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise FfmpegError(f"FFprobe returned invalid stream data for {path}") from error
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise FfmpegError(f"FFprobe found no video stream in {path}")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    try:
        return MediaSpec(
            path=path,
            width=_positive_int(video.get("width"), "width", path),
            height=_positive_int(video.get("height"), "height", path),
            fps=_fraction(video.get("avg_frame_rate"), "avg_frame_rate", path),
            time_base=_fraction(video.get("time_base"), "time_base", path),
            pixel_format=_required_text(video.get("pix_fmt"), "pixel format", path),
            codec=_required_text(video.get("codec_name"), "codec", path),
            profile=_optional_text(video.get("profile")),
            color_space=_optional_text(video.get("color_space")),
            color_transfer=_optional_text(video.get("color_transfer")),
            color_primaries=_optional_text(video.get("color_primaries")),
            audio=_audio_spec(audio, path),
            frame_count=_frame_count(video, path),
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise FfmpegError(f"FFprobe returned incomplete media data for {path}") from error


def _run_media_tool(
    executable: Path,
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    no_overwrite: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(executable), "-hide_banner"]
    if no_overwrite:
        command.extend(["-nostdin", "-n"])
    command.extend(args)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise FfmpegError(f"local media tool is unavailable: {executable}") from error
    except subprocess.TimeoutExpired as error:
        raise FfmpegError(f"local media tool timed out: {executable}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or f"exit {error.returncode}"
        raise FfmpegError(f"local media tool failed: {detail}") from error


def _audio_spec(stream: Any, path: Path) -> AudioSpec | None:
    if stream is None:
        return None
    if not isinstance(stream, dict):
        raise FfmpegError(f"FFprobe returned invalid audio data for {path}")
    return AudioSpec(
        codec=_required_text(stream.get("codec_name"), "audio codec", path),
        sample_rate=_optional_int(stream.get("sample_rate"), "audio sample rate", path),
        channels=_optional_int(stream.get("channels"), "audio channels", path),
    )


def _frame_count(stream: dict[str, Any], path: Path) -> int:
    raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    return _positive_int(raw, "frame count", path)


def _positive_int(value: Any, field: str, path: Path) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive for {path}")
    return result


def _optional_int(value: Any, field: str, path: Path) -> int | None:
    if value in {None, "N/A"}:
        return None
    return _positive_int(value, field, path)


def _fraction(value: Any, field: str, path: Path) -> Fraction:
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        raise ValueError(f"{field} is missing for {path}")
    result = Fraction(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive for {path}")
    return result


def _required_text(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value or value == "N/A":
        raise ValueError(f"{field} is missing for {path}")
    return value


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value and value != "N/A" else None
