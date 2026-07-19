from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .comfy_client import HistoryOutput, HistoryResult
from .config import (
    ModelFiles,
    ProjectConfig,
    load_presets,
    resolve_preset,
    validate_lora_policy,
)
from .frames import create_contact_sheet, extract_candidate_frames
from .hashing import sha256_file
from .metadata import RenderMetadata, write_metadata
from .project import Attempt, AttemptState, create_attempt, transition_attempt
from .qc import initialize_qc
from .workflow import (
    ApiGraph,
    NodeRef,
    WorkflowError,
    build_api_graph,
    find_unique_node,
    validate_graph_against_object_info,
    validate_two_stage_graph,
)


class RenderError(RuntimeError):
    """Raised when a local render cannot produce reviewable artifacts."""


@dataclass(frozen=True)
class SegmentRequest:
    positive: str
    negative: str
    seed: int
    width: int
    height: int
    frames: int
    opening_frame: Path


class RenderClient(Protocol):
    def upload_image(self, path: Path) -> str: ...

    def submit(self, graph: ApiGraph) -> str: ...

    def wait(self, prompt_id: str) -> HistoryResult: ...

    def fetch_output(self, output: HistoryOutput, destination: Path) -> Path: ...


def render_segment(
    project: ProjectConfig,
    shot_id: str,
    segment_id: str,
    client: RenderClient,
) -> Attempt:
    attempt = create_attempt(project, shot_id, segment_id)
    base_graph = _read_graph(attempt.path / "workflow-api.json")
    resolved, available_files = _resolved_submission_config(project)
    graph = build_api_graph(base_graph, resolved, available_files)
    segment = _segment_request(project, shot_id, segment_id)
    _write_json_exclusive(
        attempt.path / "input-upload.json",
        {
            "opening_frame": {
                "path": str(segment.opening_frame),
                "sha256": sha256_file(segment.opening_frame),
            }
        },
    )
    uploaded_opening = _safe_uploaded_name(client.upload_image(segment.opening_frame))
    _patch_segment_graph(graph, segment, uploaded_opening)
    _validate_submission_graph(project, graph)
    _write_json_exclusive(attempt.path / "configured-workflow-api.json", graph)
    submission = {"prompt": graph}
    submission_path = attempt.path / "submission-request.json"
    _write_json_exclusive(submission_path, submission)
    _write_json_exclusive(
        attempt.path / "submission-provenance.json",
        {
            "request": sha256_file(submission_path),
            "source_manifest": sha256_file(
                next(attempt.path.glob("source-manifest.*"))
            ),
            "base_workflow": sha256_file(attempt.path / "workflow-api.json"),
            "workflow": sha256_file(attempt.path / "configured-workflow-api.json"),
        },
    )

    attempt = transition_attempt(attempt, AttemptState.RENDERING, "queued locally")
    prompt_id = client.submit(graph)
    history = client.wait(prompt_id)
    history_path = attempt.path / "history.json"
    _write_json_exclusive(
        history_path,
        {"prompt_id": history.prompt_id, "history": history.raw},
    )

    history_output = _video_output(history)
    output_suffix = Path(history_output.filename).suffix.casefold()
    video = attempt.path / "outputs" / f"segment{output_suffix}"
    if history_output.local_path is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history_output.local_path, video)
    else:
        client.fetch_output(history_output, video)
    if not video.is_file() or video.stat().st_size == 0:
        raise RenderError("Local ComfyUI history output did not produce a video")

    candidate_count = _candidate_count(project)
    candidates_root = attempt.path / "candidate-frames"
    head_frames = extract_candidate_frames(
        video, candidate_count, "head", candidates_root / "head"
    )
    tail_frames = extract_candidate_frames(
        video, candidate_count, "tail", candidates_root / "tail"
    )
    contact_sheet = create_contact_sheet(
        [*head_frames, *tail_frames], attempt.path / "contact-sheet.png"
    )

    outputs: dict[str, Path] = {
        "segment": video,
        "contact_sheet": contact_sheet,
        **{f"head_{index:02d}": path for index, path in enumerate(head_frames)},
        **{f"tail_{index:02d}": path for index, path in enumerate(tail_frames)},
    }
    write_metadata(
        attempt,
        RenderMetadata(
            outputs=outputs,
            rendered_at=datetime.now(UTC),
            details={
                "candidate_count": candidate_count,
                "history": str(history_path),
                "history_output": {
                    "filename": history_output.filename,
                    "node_id": history_output.node_id,
                    "subfolder": history_output.subfolder,
                    "type": history_output.type,
                },
                "prompt_id": prompt_id,
                "submission_request": {
                    "path": str(submission_path),
                    "sha256": sha256_file(submission_path),
                },
            },
        ),
    )
    attempt = transition_attempt(attempt, AttemptState.RENDERED, "local render complete")
    initialize_qc(
        attempt,
        video=video,
        head_frames=head_frames,
        tail_frames=tail_frames,
        contact_sheet=contact_sheet,
        automatic_continuation_authorized=_automatic_continuation(project),
    )
    return transition_attempt(attempt, AttemptState.NEEDS_REVIEW, None)


def _resolved_submission_config(project: ProjectConfig):
    presets_path = _local_path(
        project, project.source.get("presets", Path(__file__).parents[2] / "config" / "presets.yaml")
    )
    resolved = resolve_preset(project, load_presets(presets_path))
    available_files = _model_files(project)
    validate_lora_policy(resolved, available_files)
    return resolved, available_files


def _validate_submission_graph(project: ProjectConfig, graph: ApiGraph) -> None:
    validate_two_stage_graph(graph)
    object_info = project.source.get("object_info")
    if object_info is not None:
        object_info_path = _local_path(project, object_info)
        try:
            schema = json.loads(object_info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RenderError(f"invalid local object_info snapshot: {object_info_path}") from error
        if not isinstance(schema, Mapping):
            raise RenderError("local object_info snapshot must be a JSON object")
        validate_graph_against_object_info(graph, schema)


def _segment_request(
    project: ProjectConfig, shot_id: str, segment_id: str
) -> SegmentRequest:
    request = _mapping(project.source.get("request", {}), "request")
    render = _mapping(project.source.get("render", {}), "render")
    shot, segment = _find_segment(project, shot_id, segment_id)
    prompt_blocks = _mapping(project.source.get("prompt_blocks", {}), "prompt_blocks")
    positive = _prompt_value(
        request,
        segment,
        prompt_blocks,
        shot,
        ("positive", "positive_prompt", "prompt"),
        "positive prompt",
    )
    negative = _prompt_value(
        request,
        segment,
        prompt_blocks,
        shot,
        ("negative", "negative_prompt"),
        "negative prompt",
    )
    width = _positive_int(
        _first_value(segment, request, render, keys=("width",)), "render width"
    )
    height = _positive_int(
        _first_value(segment, request, render, keys=("height",)), "render height"
    )
    frames = _positive_int(
        _first_value(segment, request, render, keys=("frames", "length")),
        "render frames",
    )
    return SegmentRequest(
        positive=positive,
        negative=negative,
        seed=_segment_seed(segment, request, render),
        width=width,
        height=height,
        frames=frames,
        opening_frame=_opening_frame(project),
    )


def _find_segment(
    project: ProjectConfig, shot_id: str, segment_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    shots = project.source.get("shots")
    if shots is None:
        return {}, {}
    if not isinstance(shots, (list, tuple)):
        raise RenderError("shots must be a list")
    shot = next(
        (
            item
            for item in shots
            if isinstance(item, Mapping) and item.get("id") == shot_id
        ),
        None,
    )
    if shot is None:
        raise RenderError(f"configured shot was not found: {shot_id}")
    segments = shot.get("segments")
    if not isinstance(segments, (list, tuple)):
        raise RenderError(f"configured shot has no segments: {shot_id}")
    segment = next(
        (
            item
            for item in segments
            if isinstance(item, Mapping) and item.get("id") == segment_id
        ),
        None,
    )
    if segment is None:
        raise RenderError(f"configured segment was not found: {segment_id}")
    return shot, segment


def _prompt_value(
    request: Mapping[str, Any],
    segment: Mapping[str, Any],
    prompt_blocks: Mapping[str, Any],
    shot: Mapping[str, Any],
    keys: tuple[str, ...],
    label: str,
) -> str:
    direct = _first_value(segment, request, keys=keys)
    if isinstance(direct, str) and direct.strip():
        return direct
    if keys[0] == "positive":
        parts = (
            prompt_blocks.get("identity_lock"),
            prompt_blocks.get("continuity_lock"),
            shot.get("camera"),
            shot.get("environment"),
            segment.get("action"),
        )
        assembled = "\n".join(
            value.strip() for value in parts if isinstance(value, str) and value.strip()
        )
        if assembled:
            return assembled
    if keys[0] == "negative":
        configured = prompt_blocks.get("negative")
        if isinstance(configured, str) and configured.strip():
            return configured
    raise RenderError(f"{label} must be a non-empty configured string")


def _segment_seed(
    segment: Mapping[str, Any], request: Mapping[str, Any], render: Mapping[str, Any]
) -> int:
    explicit = _first_value(segment, request, keys=("seed",))
    if explicit is not None:
        return _non_negative_int(explicit, "segment seed")
    base = render.get("seed_base")
    if base is None:
        raise RenderError("segment seed or render.seed_base is required")
    return _non_negative_int(base, "render seed_base") + _non_negative_int(
        segment.get("seed_offset", 0), "segment seed_offset"
    )


def _opening_frame(project: ProjectConfig) -> Path:
    inputs = _mapping(project.source.get("inputs", {}), "inputs")
    value = inputs.get("opening_frame")
    if isinstance(value, Mapping):
        value = value.get("path")
    if value is None:
        raise RenderError("inputs.opening_frame is required")
    path = _local_path(project, value)
    if not path.is_file():
        raise RenderError(f"opening_frame does not exist: {path}")
    return path


def _patch_segment_graph(
    graph: ApiGraph, segment: SegmentRequest, uploaded_opening: str
) -> None:
    _unique_target(graph, ("POSITIVE_PROMPT", "PROMPT_POSITIVE"), "CLIPTextEncode").node[
        "inputs"
    ]["text"] = segment.positive
    _unique_target(graph, ("NEGATIVE_PROMPT", "PROMPT_NEGATIVE"), "CLIPTextEncode").node[
        "inputs"
    ]["text"] = segment.negative
    _unique_target(graph, ("SEGMENT_FIRST_IMAGE", "START_IMAGE"), "LoadImage").node[
        "inputs"
    ]["image"] = uploaded_opening
    conditioning = _unique_target(graph, ("I2V_CONDITIONING",), "WanImageToVideo").node[
        "inputs"
    ]
    conditioning["width"] = segment.width
    conditioning["height"] = segment.height
    conditioning["length"] = segment.frames
    for title in ("SAMPLER_HIGH", "SAMPLER_LOW"):
        _unique_target(graph, (title,), "KSamplerAdvanced").node["inputs"][
            "noise_seed"
        ] = segment.seed


def _unique_target(
    graph: ApiGraph, titles: tuple[str, ...], class_type: str
) -> NodeRef:
    matches: list[NodeRef] = []
    for title in titles:
        try:
            matches.append(find_unique_node(graph, title, class_type))
        except WorkflowError as error:
            if "found zero matches" not in str(error):
                raise RenderError(str(error)) from error
    if len(matches) != 1:
        names = " or ".join(titles)
        raise RenderError(
            f"expected exactly one {class_type} target titled {names}; found {len(matches)}"
        )
    return matches[0]


def _first_value(
    *sources: Mapping[str, Any], keys: tuple[str, ...]
) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RenderError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RenderError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RenderError(f"{label} must be a non-negative integer")
    return value


def _safe_uploaded_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise RenderError("uploaded opening frame name is unsafe")
    return value


def _model_files(project: ProjectConfig) -> ModelFiles:
    configured = project.source.get("model_files")
    if isinstance(configured, (list, tuple)) and all(
        isinstance(name, str) and name for name in configured
    ):
        return ModelFiles.from_names(set(configured))
    roots = project.source.get("model_roots")
    if isinstance(roots, (list, tuple)) and all(
        isinstance(root, (str, Path)) for root in roots
    ):
        return ModelFiles.discover(*(_local_path(project, root) for root in roots))
    raise RenderError(
        "project must provide model_files or model_roots for pre-submit validation"
    )


def _read_graph(path: Path) -> ApiGraph:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RenderError(f"invalid workflow snapshot: {path}") from error
    if not isinstance(graph, dict) or not all(
        isinstance(node_id, str) and isinstance(node, dict)
        for node_id, node in graph.items()
    ):
        raise RenderError("workflow snapshot must be an API graph")
    return graph


def _video_output(history: HistoryResult) -> HistoryOutput:
    extensions = {".gif", ".mkv", ".mov", ".mp4", ".webm"}
    videos = [
        output
        for output in history.outputs
        if Path(output.filename).suffix.casefold() in extensions
    ]
    if not videos:
        raise RenderError("Local ComfyUI history contains no video output")
    if len(videos) > 1:
        raise RenderError("Local ComfyUI history contains multiple video outputs")
    output = videos[0]
    if Path(output.filename).name != output.filename:
        raise RenderError("Local ComfyUI history output filename is unsafe")
    return output


def _candidate_count(project: ProjectConfig) -> int:
    value = project.source.get("candidate_count", 5)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RenderError("candidate_count must be a positive integer")
    return value


def _automatic_continuation(project: ProjectConfig) -> bool:
    policy = project.source.get("policy", {})
    if not isinstance(policy, Mapping):
        raise RenderError("project policy must be a mapping")
    value = policy.get("automatic_continuation", False)
    if not isinstance(value, bool):
        raise RenderError("policy.automatic_continuation must be a boolean")
    return value


def _local_path(project: ProjectConfig, value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise RenderError("project local path must be a string or Path")
    path = Path(value)
    if not path.is_absolute():
        path = project.path.parent / path
    return path


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
