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
    canonical_width: int | None = None
    canonical_height: int | None = None
    thumbnail_left_hash: str | None = None
    thumbnail_right_hash: str | None = None
    full_fidelity: bool = False


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
    expected_duration: Fraction | None = None
    output_fps: Fraction | None = None


@dataclass(frozen=True)
class AssemblyPlan:
    normalize_first: bool
    audio_policy: Literal["drop", "preserve"]
    concat_manifest: Path
    expected_frame_count: int | None
    expected_duration: Fraction | None
    output_fps: Fraction
    boundary_decisions: tuple[BoundaryDecision, ...]
    operations: tuple[AssemblyOperation, ...]


@dataclass(frozen=True)
class AssemblyExecution:
    """Post-output evidence from a successfully validated native assembly."""

    expected_frame_count: int
    expected_duration: Fraction
    output_fps: Fraction
    validated_outputs: tuple[MediaSpec, ...]
    rife_ready: Path | None


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
    """Classify unverified hashes for review; they cannot authorize a trim."""
    if perceptual_distance < 0:
        raise AssemblyError("perceptual distance must not be negative")
    if left_hash == right_hash and perceptual_distance == 0:
        return BoundaryDecision(
            left_hash=left_hash,
            right_hash=right_hash,
            perceptual_distance=perceptual_distance,
            trim_right_frames=0,
            requires_review=True,
            reason="matching hashes require full-fidelity boundary confirmation",
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
    if _media_has_alpha(left_spec) or _media_has_alpha(right_spec):
        return BoundaryDecision(
            left_hash="",
            right_hash="",
            perceptual_distance=0,
            trim_right_frames=0,
            requires_review=True,
            reason="alpha-capable boundary media requires review in V1",
        )
    with tempfile.TemporaryDirectory(prefix="wan22-boundary-") as temporary:
        root = Path(temporary)
        left_canonical = _extract_canonical_boundary_frame(
            left_spec,
            left_spec.frame_count - 1,
            root / "left.yuv444p16le",
        )
        right_canonical = _extract_canonical_boundary_frame(
            right_spec,
            0,
            root / "right.yuv444p16le",
        )
        left_thumbnail = _extract_thumbnail_boundary_frame(
            left_spec,
            left_spec.frame_count - 1,
            root / "left.gray",
        )
        right_thumbnail = _extract_thumbnail_boundary_frame(
            right_spec,
            0,
            root / "right.gray",
        )
        left_canonical_bytes = left_canonical.read_bytes()
        right_canonical_bytes = right_canonical.read_bytes()
        left_thumbnail_bytes = left_thumbnail.read_bytes()
        right_thumbnail_bytes = right_thumbnail.read_bytes()
    return _compare_full_fidelity_boundary(
        left_hash=hashlib.sha256(left_canonical_bytes).hexdigest(),
        right_hash=hashlib.sha256(right_canonical_bytes).hexdigest(),
        left_dimensions=(left_spec.width, left_spec.height),
        right_dimensions=(right_spec.width, right_spec.height),
        perceptual_distance=_mean_absolute_distance(left_thumbnail_bytes, right_thumbnail_bytes),
        thumbnail_left_hash=hashlib.sha256(left_thumbnail_bytes).hexdigest(),
        thumbnail_right_hash=hashlib.sha256(right_thumbnail_bytes).hexdigest(),
    )


def write_boundary_decision(decision: BoundaryDecision, destination: Path) -> Path:
    """Persist one boundary decision without rewriting a prior review record."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(asdict(decision), output, indent=2, sort_keys=True)
        output.write("\n")
    return destination


def write_boundary_decisions(
    decisions: tuple[BoundaryDecision, ...],
    destination: Path,
) -> Path:
    """Persist the immutable decision set required by an assembly execution."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(
            {"decisions": [asdict(decision) for decision in decisions]},
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")
    return destination


def validate_output_duration(
    output: Path,
    expected_duration: Fraction,
    *,
    output_fps: Fraction | None = None,
) -> MediaSpec:
    """Probe assembled media and reject timeline drift greater than one output frame."""
    if expected_duration <= 0:
        raise AssemblyError("expected duration must be positive")
    spec = probe_media(output)
    if output_fps is not None and spec.fps != output_fps:
        raise AssemblyError(
            f"assembled output FPS {spec.fps} differs from planned FPS {output_fps}"
        )
    tolerance_fps = output_fps or spec.fps
    if tolerance_fps <= 0:
        raise AssemblyError("output FPS must be positive")
    actual_duration = duration_seconds(spec.frame_count, spec.fps)
    if abs(actual_duration - expected_duration) > Fraction(1, 1) / tolerance_fps:
        raise AssemblyError(
            "assembled output duration differs from the expected duration by more than one output frame"
        )
    return spec


def validate_assembly_outputs(
    plan: AssemblyPlan,
    *,
    expected_duration: Fraction,
    output_fps: Fraction,
) -> tuple[MediaSpec, ...]:
    """Run final duration checks against the execution-time intermediate evidence."""
    validations = [
        operation for operation in plan.operations if operation.kind == "validate_duration"
    ]
    if not validations:
        raise AssemblyError("assembly plan has no final duration validation operations")
    results: list[MediaSpec] = []
    for operation in validations:
        results.append(
            validate_output_duration(
                operation.output,
                expected_duration,
                output_fps=output_fps,
            )
        )
    return tuple(results)


def execute_assembly_plan(
    plan: AssemblyPlan,
    *,
    decision_log: Path,
    ffmpeg: Path = Path("ffmpeg"),
) -> AssemblyExecution:
    """Execute an approved plan, then validate outputs from probed intermediates.

    The decision log is exclusively created before FFmpeg starts. RIFE is deliberately
    left commandless and becomes ready only after every native output has passed
    post-output duration validation.
    """
    _ensure_decision_log_does_not_collide(plan, decision_log)
    write_boundary_decisions(plan.boundary_decisions, decision_log)
    rife_output: Path | None = None
    for operation in plan.operations:
        if operation.kind == "validate_duration":
            continue
        if operation.kind == "rife":
            rife_output = operation.output
            continue
        if operation.kind == "write_concat_manifest":
            _write_concat_manifest_file(operation.inputs, operation.output)
            continue
        if not operation.command:
            raise AssemblyError(f"assembly operation has no executable command: {operation.kind}")
        operation.output.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(operation.command, ffmpeg=ffmpeg)

    frame_count, expected_duration, output_fps = _probed_concat_timeline(plan)
    validated_outputs = validate_assembly_outputs(
        plan,
        expected_duration=expected_duration,
        output_fps=output_fps,
    )
    return AssemblyExecution(
        expected_frame_count=frame_count,
        expected_duration=expected_duration,
        output_fps=output_fps,
        validated_outputs=validated_outputs,
        rife_ready=rife_output,
    )


def _ensure_decision_log_does_not_collide(plan: AssemblyPlan, decision_log: Path) -> None:
    decision_location = decision_log.resolve()
    operation_paths = [
        path
        for operation in plan.operations
        for path in (*operation.inputs, operation.output)
    ]
    if any(path.resolve() == decision_location for path in operation_paths):
        raise AssemblyError("assembly decision log must not collide with an input or output")


def plan_assembly(
    inputs: list[MediaSpec],
    output: AssemblyTargets,
    *,
    boundary_decisions: list[BoundaryDecision] | None = None,
    request_rife: bool = False,
    qc_approved: bool = False,
) -> AssemblyPlan:
    if not inputs:
        raise AssemblyError("assembly requires at least one accepted input")
    if output.audio_policy not in {"drop", "preserve"}:
        raise AssemblyError("audio policy must be drop or preserve")
    if output.audio_policy == "preserve":
        raise AssemblyError(
            "audio preservation is not supported without end-to-end A/V timing validation"
        )
    if request_rife:
        if not qc_approved:
            raise AssemblyError("RIFE may be scheduled only after approved QC")
        if output.rife_review_mp4 is None:
            raise AssemblyError("RIFE requires an explicit review MP4 target")
    _validate_inputs(inputs)
    trims = _trim_counts(inputs, boundary_decisions)

    concat_manifest = output.review_mp4.with_suffix(".concat.txt")
    normalized_inputs, normalization = _normalization_operations(
        inputs,
        output,
        concat_manifest,
        force_normalization=any(trims),
    )
    concat_inputs, trims_operations = _trim_operations(
        normalized_inputs,
        inputs,
        trims,
        output,
        concat_manifest,
    )
    native = concat_manifest.with_name(f"{concat_manifest.stem}-native.mkv")
    _reserve_assembly_output_paths(
        output,
        request_rife,
        concat_manifest,
        native,
        normalization,
        trims_operations,
    )
    manifest_operation = _concat_manifest_operation(concat_inputs, concat_manifest)
    concat = AssemblyOperation(
        kind="concat",
        inputs=concat_inputs,
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
    duration_checks = _duration_validation_operations((review, ffv1, prores))
    operations = [
        *normalization,
        *trims_operations,
        manifest_operation,
        concat,
        review,
        ffv1,
        prores,
        *duration_checks,
    ]
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
        expected_frame_count=None,
        expected_duration=None,
        output_fps=inputs[0].fps,
        boundary_decisions=tuple(boundary_decisions or ()),
        operations=tuple(operations),
    )


def _compare_full_fidelity_boundary(
    *,
    left_hash: str,
    right_hash: str,
    left_dimensions: tuple[int, int],
    right_dimensions: tuple[int, int],
    perceptual_distance: int,
    thumbnail_left_hash: str,
    thumbnail_right_hash: str,
) -> BoundaryDecision:
    if left_dimensions == right_dimensions and left_hash == right_hash:
        return BoundaryDecision(
            left_hash=left_hash,
            right_hash=right_hash,
            perceptual_distance=perceptual_distance,
            trim_right_frames=1,
            requires_review=False,
            reason="exact duplicate boundary frame verified from full-fidelity canonical samples",
            canonical_width=left_dimensions[0],
            canonical_height=left_dimensions[1],
            thumbnail_left_hash=thumbnail_left_hash,
            thumbnail_right_hash=thumbnail_right_hash,
            full_fidelity=True,
        )
    if perceptual_distance <= 5:
        reason = "perceptually similar boundary frame requires review"
        if left_dimensions != right_dimensions:
            reason = "boundary dimensions differ and perceptual similarity requires review"
        return BoundaryDecision(
            left_hash=left_hash,
            right_hash=right_hash,
            perceptual_distance=perceptual_distance,
            trim_right_frames=0,
            requires_review=True,
            reason=reason,
            thumbnail_left_hash=thumbnail_left_hash,
            thumbnail_right_hash=thumbnail_right_hash,
        )
    return BoundaryDecision(
        left_hash=left_hash,
        right_hash=right_hash,
        perceptual_distance=perceptual_distance,
        trim_right_frames=0,
        requires_review=False,
        reason="boundary frames are distinct",
        thumbnail_left_hash=thumbnail_left_hash,
        thumbnail_right_hash=thumbnail_right_hash,
    )


def _extract_canonical_boundary_frame(spec: MediaSpec, index: int, destination: Path) -> Path:
    try:
        run_ffmpeg(
            [
                "-i",
                str(spec.path),
                "-vf",
                f"select=eq(n\\,{index})",
                "-frames:v",
                "1",
                "-pix_fmt",
                "yuv444p16le",
                "-f",
                "rawvideo",
                str(destination),
            ]
        )
    except FfmpegError as error:
        raise AssemblyError(f"could not inspect boundary frame: {spec.path}") from error
    expected_bytes = spec.width * spec.height * 3 * 2
    if not destination.is_file() or destination.stat().st_size != expected_bytes:
        raise AssemblyError(f"FFmpeg did not extract boundary frame: {spec.path}")
    return destination


def _extract_thumbnail_boundary_frame(spec: MediaSpec, index: int, destination: Path) -> Path:
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
    *,
    force_normalization: bool = False,
) -> tuple[tuple[Path, ...], list[AssemblyOperation]]:
    reference = inputs[0]
    needs_normalization = force_normalization or any(
        not _compatible(spec, reference, output.audio_policy) for spec in inputs
    )
    if not needs_normalization:
        return tuple(spec.path for spec in inputs), []
    normalized: list[Path] = []
    operations: list[AssemblyOperation] = []
    for index, spec in enumerate(inputs, start=1):
        target = concat_manifest.with_name(f"{concat_manifest.stem}-normalized-{index:04d}.mkv")
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
                    f"scale={reference.width}:{reference.height},fps={reference.fps}:round=down:eof_action=pass",
                    "-fps_mode",
                    "passthrough",
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


def _trim_counts(
    inputs: list[MediaSpec],
    boundary_decisions: list[BoundaryDecision] | None,
) -> tuple[int, ...]:
    if boundary_decisions is None:
        return (0,) * len(inputs)
    if len(boundary_decisions) != len(inputs) - 1:
        raise AssemblyError("assembly requires one boundary decision between each input pair")
    counts = [0] * len(inputs)
    for index, decision in enumerate(boundary_decisions, start=1):
        if decision.requires_review:
            raise AssemblyError("boundary decision requires review before automatic assembly")
        if decision.trim_right_frames not in {0, 1}:
            raise AssemblyError("boundary decision requires review before automatic assembly")
        if decision.trim_right_frames == 0:
            continue
        if decision.trim_right_frames >= inputs[index].frame_count:
            raise AssemblyError("boundary trim would remove the complete right-hand segment")
        _verify_exact_duplicate_boundary(decision, inputs[index - 1], inputs[index])
        counts[index] = decision.trim_right_frames
    return tuple(counts)


def _trim_operations(
    normalized_inputs: tuple[Path, ...],
    inputs: list[MediaSpec],
    trims: tuple[int, ...],
    output: AssemblyTargets,
    concat_manifest: Path,
) -> tuple[tuple[Path, ...], list[AssemblyOperation]]:
    reference = inputs[0]
    concat_inputs = list(normalized_inputs)
    operations: list[AssemblyOperation] = []
    for index, trim_frames in enumerate(trims):
        if trim_frames == 0:
            continue
        target = concat_manifest.with_name(
            f"{concat_manifest.stem}-trimmed-{index + 1:04d}.mkv"
        )
        source = normalized_inputs[index]
        concat_inputs[index] = target
        operations.append(
            AssemblyOperation(
                kind="trim_boundary",
                inputs=(source,),
                output=target,
                command=(
                    "-i",
                    str(source),
                    "-vf",
                    f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS",
                    "-map",
                    "0:v:0",
                    "-fps_mode",
                    "passthrough",
                    "-c:v",
                    "ffv1",
                    *_color_args(reference),
                    *_audio_args(output.audio_policy),
                    str(target),
                ),
            )
        )
    return tuple(concat_inputs), operations


def _concat_manifest_operation(
    inputs: tuple[Path, ...],
    destination: Path,
) -> AssemblyOperation:
    return AssemblyOperation(
        kind="write_concat_manifest",
        inputs=inputs,
        output=destination,
        command=(),
    )


def _write_concat_manifest_file(inputs: tuple[Path, ...], destination: Path) -> Path:
    lines = tuple(_concat_manifest_line(path) for path in inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as manifest:
        manifest.writelines(lines)
    return destination


def _concat_manifest_line(path: Path) -> str:
    value = path.resolve().as_posix()
    if "\r" in value or "\n" in value:
        raise AssemblyError(f"concat path contains a line break: {path}")
    escaped = value.replace("'", r"'\''")
    return f"file '{escaped}'\n"


def _verify_exact_duplicate_boundary(
    decision: BoundaryDecision,
    left: MediaSpec,
    right: MediaSpec,
) -> None:
    if _media_has_alpha(left) or _media_has_alpha(right):
        raise AssemblyError("boundary requires review because alpha-capable media cannot be auto-trimmed in V1")
    try:
        actual = compare_boundary(left.path, right.path)
    except (AssemblyError, FfmpegError) as error:
        raise AssemblyError("boundary requires review because exact evidence could not be verified") from error
    if not _is_exact_duplicate(actual) or not _same_exact_evidence(decision, actual):
        raise AssemblyError("boundary requires review because exact duplicate evidence was not verified")


def _is_exact_duplicate(decision: BoundaryDecision) -> bool:
    return (
        decision.left_hash == decision.right_hash
        and decision.trim_right_frames == 1
        and not decision.requires_review
        and decision.full_fidelity
        and decision.canonical_width is not None
        and decision.canonical_height is not None
    )


def _same_exact_evidence(left: BoundaryDecision, right: BoundaryDecision) -> bool:
    return (
        _is_exact_duplicate(left)
        and left.left_hash == right.left_hash
        and left.right_hash == right.right_hash
        and left.perceptual_distance == right.perceptual_distance
        and left.canonical_width == right.canonical_width
        and left.canonical_height == right.canonical_height
        and left.full_fidelity == right.full_fidelity
    )


def _duration_validation_operations(
    outputs: tuple[AssemblyOperation, ...],
) -> tuple[AssemblyOperation, ...]:
    return tuple(
        AssemblyOperation(
            kind="validate_duration",
            inputs=(operation.output,),
            output=operation.output,
            command=(),
        )
        for operation in outputs
    )


def _probed_concat_timeline(plan: AssemblyPlan) -> tuple[int, Fraction, Fraction]:
    concat = next(
        (operation for operation in plan.operations if operation.kind == "concat"),
        None,
    )
    if concat is None or not concat.inputs:
        raise AssemblyError("assembly plan has no concat inputs to validate")
    inputs = tuple(probe_media(path) for path in concat.inputs)
    output_fps = inputs[0].fps
    if output_fps != plan.output_fps or any(spec.fps != output_fps for spec in inputs):
        raise AssemblyError("probed concat inputs do not share the planned output FPS")
    frame_count = sum(spec.frame_count for spec in inputs)
    if frame_count <= 0:
        raise AssemblyError("probed concat inputs contain no video frames")
    return frame_count, duration_seconds(frame_count, output_fps), output_fps


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
    if policy != "drop":
        raise AssemblyError(
            "audio preservation is not supported without end-to-end A/V timing validation"
        )
    return ("-an",)


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


def _reserve_assembly_output_paths(
    output: AssemblyTargets,
    request_rife: bool,
    concat_manifest: Path,
    native: Path,
    normalization: list[AssemblyOperation],
    trims: list[AssemblyOperation],
) -> None:
    targets = [
        output.review_mp4,
        output.edit_master_ffv1,
        output.edit_master_prores,
        concat_manifest,
        native,
        *(operation.output for operation in normalization),
        *(operation.output for operation in trims),
    ]
    if request_rife and output.rife_review_mp4 is not None:
        targets.append(output.rife_review_mp4)
    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            raise AssemblyError(f"assembly output paths must be distinct: {path}")
        seen.add(resolved)
    existing = next((path for path in targets if path.exists()), None)
    if existing is not None:
        if existing.resolve() == native.resolve():
            raise AssemblyError(f"assembly native output already exists: {existing}")
        if existing.resolve() == concat_manifest.resolve():
            raise AssemblyError(f"assembly concat manifest already exists: {existing}")
        raise AssemblyError(f"assembly output already exists: {existing}")


def _media_has_alpha(spec: MediaSpec) -> bool:
    pixel_format = spec.pixel_format.lower()
    return pixel_format == "pal8" or pixel_format.startswith(
        (
            "a2",
            "abgr",
            "argb",
            "ayuv",
            "bgra",
            "gbrap",
            "rgba",
            "rgbaf",
            "vuya",
            "ya",
            "yuva",
        )
    ) or "alpha" in pixel_format


def _validate_inputs(inputs: list[MediaSpec]) -> None:
    for spec in inputs:
        if not spec.path.is_file():
            raise AssemblyError(f"accepted input does not exist: {spec.path}")
