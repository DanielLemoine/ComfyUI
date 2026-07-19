from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from .ffmpeg import FfmpegError, MediaSpec, probe_media, run_ffmpeg


class AssemblyError(ValueError):
    """Raised when a local edit cannot be assembled without an explicit plan."""


@dataclass(frozen=True)
class BoundaryDecision:
    left_hash: str
    right_hash: str
    perceptual_distance: int
    trim_right_frames: int
    requires_review: bool
    reason: str


@dataclass(frozen=True)
class AssemblyTargets:
    review_mp4: Path
    edit_master_ffv1: Path
    edit_master_prores: Path
    rife_review_mp4: Path | None = None
    audio_policy: Literal["drop", "preserve"] = "drop"


@dataclass(frozen=True)
class AssemblyOperation:
    kind: str
    inputs: tuple[Path, ...]
    output: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class AssemblyPlan:
    normalize_first: bool
    audio_policy: Literal["drop", "preserve"]
    concat_manifest: Path
    operations: tuple[AssemblyOperation, ...]


def duration_seconds(frame_count: int, fps: Fraction) -> Fraction:
    if frame_count <= 0:
        raise AssemblyError("frame_count must be positive")
    if fps <= 0:
        raise AssemblyError("fps must be positive")
    return Fraction(frame_count, 1) / fps


def compare_frame_hashes(
    left_hash: str,
    right_hash: str,
    *,
    perceptual_distance: int,
) -> BoundaryDecision:
    if perceptual_distance < 0:
        raise AssemblyError("perceptual distance must not be negative")
    if left_hash == right_hash and perceptual_distance == 0:
        return BoundaryDecision(
            left_hash=left_hash,
            right_hash=right_hash,
            perceptual_distance=perceptual_distance,
            trim_right_frames=1,
            requires_review=False,
            reason="exact duplicate boundary frame",
        )
    if perceptual_distance <= 5:
        return BoundaryDecision(
            left_hash=left_hash,
            right_hash=right_hash,
            perceptual_distance=perceptual_distance,
            trim_right_frames=0,
            requires_review=True,
            reason="perceptually similar boundary frame requires review",
        )
    return BoundaryDecision(
        left_hash=left_hash,
        right_hash=right_hash,
        perceptual_distance=perceptual_distance,
        trim_right_frames=0,
        requires_review=False,
        reason="boundary frames are distinct",
    )


def compare_boundary(left: Path, right: Path) -> BoundaryDecision:
    """Diagnose the final left frame against the opening right frame locally."""
    left_spec = probe_media(left)
    right_spec = probe_media(right)
    with tempfile.TemporaryDirectory(prefix="wan22-boundary-") as temporary:
        root = Path(temporary)
        left_frame = _extract_boundary_frame(left_spec, left_spec.frame_count - 1, root / "left.gray")
        right_frame = _extract_boundary_frame(right_spec, 0, root / "right.gray")
        left_bytes = left_frame.read_bytes()
        right_bytes = right_frame.read_bytes()
    return compare_frame_hashes(
        hashlib.sha256(left_bytes).hexdigest(),
        hashlib.sha256(right_bytes).hexdigest(),
        perceptual_distance=_mean_absolute_distance(left_bytes, right_bytes),
    )


def write_boundary_decision(decision: BoundaryDecision, destination: Path) -> Path:
    """Persist one boundary decision without rewriting a prior review record."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(asdict(decision), output, indent=2, sort_keys=True)
        output.write("\n")
    return destination


def plan_assembly(
    inputs: list[MediaSpec],
    output: AssemblyTargets,
    *,
    request_rife: bool = False,
    qc_approved: bool = False,
) -> AssemblyPlan:
    if not inputs:
        raise AssemblyError("assembly requires at least one accepted input")
    if output.audio_policy not in {"drop", "preserve"}:
        raise AssemblyError("audio policy must be drop or preserve")
    _ensure_new_targets(output, request_rife)
    if request_rife:
        if not qc_approved:
            raise AssemblyError("RIFE may be scheduled only after approved QC")
        if output.rife_review_mp4 is None:
            raise AssemblyError("RIFE requires an explicit review MP4 target")
    _validate_inputs(inputs)

    concat_manifest = output.review_mp4.with_suffix(".concat.txt")
    if concat_manifest.exists():
        raise AssemblyError(f"assembly concat manifest already exists: {concat_manifest}")
    normalized_inputs, normalization = _normalization_operations(inputs, output, concat_manifest)
    native = concat_manifest.with_name(f"{concat_manifest.stem}-native.mkv")
    if native.exists():
        raise AssemblyError(f"assembly native output already exists: {native}")
    concat = AssemblyOperation(
        kind="concat",
        inputs=normalized_inputs,
        output=native,
        command=(
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            *_audio_args(output.audio_policy),
            str(native),
        ),
    )
    review = AssemblyOperation(
        kind="review_mp4",
        inputs=(native,),
        output=output.review_mp4,
        command=("-i", str(native), "-c:v", "libx264", "-pix_fmt", "yuv420p", *_audio_args(output.audio_policy), str(output.review_mp4)),
    )
    ffv1 = AssemblyOperation(
        kind="edit_master_ffv1",
        inputs=(native,),
        output=output.edit_master_ffv1,
        command=("-i", str(native), "-c:v", "ffv1", *_audio_args(output.audio_policy), str(output.edit_master_ffv1)),
    )
    prores = AssemblyOperation(
        kind="edit_master_prores",
        inputs=(native,),
        output=output.edit_master_prores,
        command=("-i", str(native), "-c:v", "prores_ks", "-profile:v", "3", *_audio_args(output.audio_policy), str(output.edit_master_prores)),
    )
    operations = [*normalization, concat, review, ffv1, prores]
    if request_rife:
        operations.append(
            AssemblyOperation(
                kind="rife",
                inputs=(output.review_mp4,),
                output=output.rife_review_mp4,
                command=(),
            )
        )
    return AssemblyPlan(
        normalize_first=bool(normalization),
        audio_policy=output.audio_policy,
        concat_manifest=concat_manifest,
        operations=tuple(operations),
    )


def _extract_boundary_frame(spec: MediaSpec, index: int, destination: Path) -> Path:
    try:
        run_ffmpeg(
            [
                "-i",
                str(spec.path),
                "-vf",
                f"select=eq(n\\,{index}),scale=32:32:flags=neighbor",
                "-frames:v",
                "1",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                str(destination),
            ]
        )
    except FfmpegError as error:
        raise AssemblyError(f"could not inspect boundary frame: {spec.path}") from error
    if not destination.is_file() or destination.stat().st_size != 32 * 32:
        raise AssemblyError(f"FFmpeg did not extract boundary frame: {spec.path}")
    return destination


def _mean_absolute_distance(left: bytes, right: bytes) -> int:
    if len(left) != len(right) or not left:
        raise AssemblyError("boundary frames could not be normalized to matching data")
    return round(sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) / len(left))


def _normalization_operations(
    inputs: list[MediaSpec],
    output: AssemblyTargets,
    concat_manifest: Path,
) -> tuple[tuple[Path, ...], list[AssemblyOperation]]:
    reference = inputs[0]
    needs_normalization = any(
        not _compatible(spec, reference, output.audio_policy) for spec in inputs
    )
    if not needs_normalization:
        return tuple(spec.path for spec in inputs), []
    normalized: list[Path] = []
    operations: list[AssemblyOperation] = []
    for index, spec in enumerate(inputs, start=1):
        target = concat_manifest.with_name(f"{concat_manifest.stem}-normalized-{index:04d}.mkv")
        if target.exists():
            raise AssemblyError(f"normalization target already exists: {target}")
        normalized.append(target)
        operations.append(
            AssemblyOperation(
                kind="normalize",
                inputs=(spec.path,),
                output=target,
                command=(
                    "-i",
                    str(spec.path),
                    "-vf",
                    f"scale={reference.width}:{reference.height}",
                    "-r",
                    str(reference.fps),
                    "-pix_fmt",
                    reference.pixel_format,
                    "-c:v",
                    "ffv1",
                    *_color_args(reference),
                    *_audio_args(output.audio_policy),
                    str(target),
                ),
            )
        )
    return tuple(normalized), operations


def _compatible(left: MediaSpec, right: MediaSpec, audio_policy: str) -> bool:
    return _video_signature(left) == _video_signature(right) and (
        audio_policy == "drop" or left.audio == right.audio
    )


def _video_signature(spec: MediaSpec) -> tuple[object, ...]:
    return (
        spec.width,
        spec.height,
        spec.fps,
        spec.time_base,
        spec.pixel_format,
        spec.codec,
        spec.profile,
        spec.color_space,
        spec.color_transfer,
        spec.color_primaries,
    )


def _audio_args(policy: str) -> tuple[str, ...]:
    return ("-an",) if policy == "drop" else ("-map", "0:a?", "-c:a", "copy")


def _color_args(spec: MediaSpec) -> tuple[str, ...]:
    arguments: list[str] = []
    for option, value in (
        ("-colorspace", spec.color_space),
        ("-color_trc", spec.color_transfer),
        ("-color_primaries", spec.color_primaries),
    ):
        if value is not None:
            arguments.extend([option, value])
    return tuple(arguments)


def _ensure_new_targets(output: AssemblyTargets, request_rife: bool) -> None:
    targets = [output.review_mp4, output.edit_master_ffv1, output.edit_master_prores]
    if request_rife and output.rife_review_mp4 is not None:
        targets.append(output.rife_review_mp4)
    duplicate = next((path for path in targets if targets.count(path) > 1), None)
    if duplicate is not None:
        raise AssemblyError(f"assembly targets must be distinct: {duplicate}")
    existing = next((path for path in targets if path.exists()), None)
    if existing is not None:
        raise AssemblyError(f"assembly target already exists: {existing}")


def _validate_inputs(inputs: list[MediaSpec]) -> None:
    for spec in inputs:
        if not spec.path.is_file():
            raise AssemblyError(f"accepted input does not exist: {spec.path}")
