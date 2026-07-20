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
    load_project,
    load_presets,
    resolve_preset,
    validate_project_contract,
    validate_lora_policy,
)
from .frames import create_contact_sheet, extract_candidate_frames
from .hashing import sha256_file
from .metadata import RenderMetadata, write_metadata
from .project import (
    Attempt,
    AttemptState,
    ProjectStateError,
    accepted_attempt_evidence,
    accepted_qc_candidate,
    create_attempt,
    load_attempt,
    selected_input,
    selected_tail_frame,
    transition_attempt,
    verify_attempt_lineage,
)
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
class InputSelection:
    path: Path
    sha256: str
    source: str
    upstream_attempt: Path | None = None
    candidate_kind: str | None = None

    def provenance(self) -> dict[str, str]:
        payload = {
            "path": str(self.path),
            "sha256": self.sha256,
            "source": self.source,
        }
        if self.upstream_attempt is not None:
            payload["upstream_attempt"] = str(self.upstream_attempt)
        if self.candidate_kind is not None:
            payload["candidate_kind"] = self.candidate_kind
        return payload


@dataclass(frozen=True)
class SegmentRequest:
    positive: str
    negative: str
    seed: int
    width: int
    height: int
    frames: int
    opening: InputSelection


@dataclass(frozen=True)
class BridgeRequest:
    bridge_id: str
    shot_id: str
    positive: str
    negative: str
    seed: int
    width: int
    height: int
    frames: int
    first_image: InputSelection
    last_image: InputSelection


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
    *,
    attempt: Attempt | None = None,
) -> Attempt:
    validate_project_contract(project)
    if attempt is None:
        segment = _segment_request(project, shot_id, segment_id)
        attempt = _new_or_planned_attempt(
            project,
            shot_id,
            segment_id,
            None,
            selected_inputs={"opening_frame": segment.opening.provenance()},
        )
    else:
        attempt = _new_or_planned_attempt(project, shot_id, segment_id, attempt)
        segment = _segment_request(
            project,
            shot_id,
            segment_id,
            opening=_stored_input_selection(attempt, "opening_frame"),
        )
    base_graph = _read_graph(attempt.path / "workflow-api.json")
    resolved, available_files = _resolved_submission_config(project)
    graph = build_api_graph(base_graph, resolved, available_files)
    _write_json_exclusive(
        attempt.path / "input-upload.json",
        {"opening_frame": segment.opening.provenance()},
    )
    uploaded_opening = _safe_uploaded_name(client.upload_image(segment.opening.path))
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
                "input_provenance": {"opening_frame": segment.opening.provenance()},
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


def render_bridge(
    project: ProjectConfig,
    bridge_id: str,
    client: RenderClient,
    *,
    attempt: Attempt | None = None,
) -> Attempt:
    """Render a native FLF bridge from explicitly selected endpoint images."""
    validate_project_contract(project)
    bridge_project = _project_with_bridge_workflow(project)
    if attempt is None:
        bridge = _bridge_request(project, bridge_id)
        attempt = _new_or_planned_attempt(
            bridge_project,
            bridge.shot_id,
            bridge.bridge_id,
            None,
            selected_inputs={
                "first_image": bridge.first_image.provenance(),
                "last_image": bridge.last_image.provenance(),
            },
        )
    else:
        bridge = _bridge_request(
            project,
            bridge_id,
            first_image=_stored_input_selection(attempt, "first_image"),
            last_image=_stored_input_selection(attempt, "last_image"),
        )
        attempt = _new_or_planned_attempt(
            bridge_project, bridge.shot_id, bridge.bridge_id, attempt
        )
    base_graph = _read_graph(attempt.path / "workflow-api.json")
    resolved, available_files = _resolved_submission_config(bridge_project)
    graph = build_api_graph(base_graph, resolved, available_files)
    _write_json_exclusive(
        attempt.path / "input-upload.json",
        {
            "first_image": bridge.first_image.provenance(),
            "last_image": bridge.last_image.provenance(),
        },
    )
    uploaded_first = _safe_uploaded_name(client.upload_image(bridge.first_image.path))
    uploaded_last = _safe_uploaded_name(client.upload_image(bridge.last_image.path))
    _patch_bridge_graph(graph, bridge, uploaded_first, uploaded_last)
    _validate_submission_graph(bridge_project, graph)
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
    video = attempt.path / "outputs" / f"bridge{output_suffix}"
    if history_output.local_path is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history_output.local_path, video)
    else:
        client.fetch_output(history_output, video)
    if not video.is_file() or video.stat().st_size == 0:
        raise RenderError("Local ComfyUI history output did not produce a video")

    candidate_count = _candidate_count(bridge_project)
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
        "bridge": video,
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
                "bridge_id": bridge.bridge_id,
                "candidate_count": candidate_count,
                "history": str(history_path),
                "history_output": {
                    "filename": history_output.filename,
                    "node_id": history_output.node_id,
                    "subfolder": history_output.subfolder,
                    "type": history_output.type,
                },
                "prompt_id": prompt_id,
                "input_provenance": {
                    "first_image": bridge.first_image.provenance(),
                    "last_image": bridge.last_image.provenance(),
                },
                "submission_request": {
                    "path": str(submission_path),
                    "sha256": sha256_file(submission_path),
                },
            },
        ),
    )
    attempt = transition_attempt(attempt, AttemptState.RENDERED, "local bridge render complete")
    initialize_qc(
        attempt,
        video=video,
        head_frames=head_frames,
        tail_frames=tail_frames,
        contact_sheet=contact_sheet,
        automatic_continuation_authorized=_automatic_continuation(bridge_project),
    )
    return transition_attempt(attempt, AttemptState.NEEDS_REVIEW, None)


def resume_attempt(
    project: ProjectConfig,
    attempt: Attempt,
    client: RenderClient,
) -> Attempt:
    """Resume exactly one immutable planned attempt from its snapshot evidence."""
    persisted = _new_or_planned_attempt(
        project, attempt.shot_id, attempt.segment_id, attempt
    )
    replay_project = _snapshot_project(project, persisted)
    validate_project_contract(replay_project)
    graph = _read_graph(persisted.path / "workflow-api.json")
    if _has_titled_node(graph, "FLF_CONDITIONING"):
        return render_bridge(
            replay_project,
            persisted.segment_id,
            client,
            attempt=persisted,
        )
    if _has_titled_node(graph, "I2V_CONDITIONING"):
        return render_segment(
            replay_project,
            persisted.shot_id,
            persisted.segment_id,
            client,
            attempt=persisted,
        )
    raise RenderError("planned attempt has no recognized native I2V or FLF workflow")


def _new_or_planned_attempt(
    project: ProjectConfig,
    shot_id: str,
    segment_id: str,
    supplied: Attempt | None,
    *,
    selected_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Attempt:
    if supplied is None:
        return create_attempt(project, shot_id, segment_id, selected_inputs=selected_inputs)
    persisted = load_attempt(supplied.path)
    try:
        verify_attempt_lineage(project, persisted)
    except ProjectStateError as error:
        raise RenderError(f"planned attempt project lineage is invalid: {error}") from error
    if persisted.shot_id != shot_id or persisted.segment_id != segment_id:
        raise RenderError("planned attempt identity does not match the requested render")
    if persisted.state is not AttemptState.PLANNED:
        raise RenderError(f"attempt is not planned and cannot be resumed: {persisted.path}")
    return persisted


def _snapshot_project(project: ProjectConfig, attempt: Attempt) -> ProjectConfig:
    snapshots = tuple(attempt.path.glob("source-manifest.*"))
    if len(snapshots) != 1:
        raise RenderError("planned attempt has no unambiguous source manifest snapshot")
    snapshot = load_project(snapshots[0])
    return ProjectConfig(path=project.path, source=snapshot.source)


def _has_titled_node(graph: ApiGraph, title: str) -> bool:
    return any(
        node.get("_meta", {}).get("title") == title for node in graph.values()
    )


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


def validate_project(project: ProjectConfig) -> dict[str, int]:
    """Validate local render inputs and both native graph types without mutation."""
    validate_project_contract(project)
    _validate_workflow_hashes(project)
    _validate_declared_input_files(project)
    resolved, available_files = _resolved_submission_config(project)
    base_graph = _project_workflow_graph(project)
    graph = build_api_graph(base_graph, resolved, available_files)
    _validate_submission_graph(project, graph)
    segment_count = 0
    shots = project.source.get("shots", ())
    if not isinstance(shots, (list, tuple)):
        raise RenderError("shots must be a list")
    for shot in shots:
        if not isinstance(shot, Mapping) or not isinstance(shot.get("id"), str):
            raise RenderError("each shot requires a string id")
        segments = shot.get("segments")
        if not isinstance(segments, (list, tuple)):
            raise RenderError(f"configured shot has no segments: {shot['id']}")
        for segment in segments:
            if not isinstance(segment, Mapping) or not isinstance(segment.get("id"), str):
                raise RenderError(f"configured shot has an invalid segment: {shot['id']}")
            if segment.get("continue_from") is None:
                _segment_request(project, shot["id"], segment["id"])
            segment_count += 1

    bridge_count = 0
    bridges = project.source.get("bridges", ())
    if not isinstance(bridges, (list, tuple)):
        raise RenderError("bridges must be a list")
    renderable_bridges = []
    for bridge in bridges:
        if not isinstance(bridge, Mapping) or not isinstance(bridge.get("id"), str):
            raise RenderError("each bridge requires a string id")
        if bridge.get("strategy") == "flf2v":
            renderable_bridges.append(bridge)
    if renderable_bridges:
        bridge_project = _project_with_bridge_workflow(project)
        bridge_graph = build_api_graph(
            _project_workflow_graph(bridge_project), resolved, available_files
        )
        _validate_submission_graph(bridge_project, bridge_graph)
        bridge_count = len(renderable_bridges)
    return {"bridges": bridge_count, "segments": segment_count}


def _validate_workflow_hashes(project: ProjectConfig) -> None:
    expected = _mapping(project.source.get("workflow_hashes"), "workflow_hashes")
    for workflow_key, hash_key in (
        ("workflow_api", "segment_api"),
        ("bridge_workflow_api", "bridge_api"),
    ):
        path = _local_path(project, project.source.get(workflow_key))
        if not path.is_file():
            raise RenderError(f"{workflow_key} does not exist: {path}")
        if sha256_file(path) != expected.get(hash_key):
            raise RenderError(f"{workflow_key} hash does not match workflow_hashes.{hash_key}")


def _validate_declared_input_files(project: ProjectConfig) -> None:
    inputs = _mapping(project.source.get("inputs", {}), "inputs")
    opening = inputs.get("opening_frame")
    if isinstance(opening, Mapping):
        opening = opening.get("path")
    if opening is not None:
        _existing_local_path(project, opening, "project opening frame")
    shots = project.source.get("shots")
    if not isinstance(shots, (list, tuple)):
        raise RenderError("shots must be a list")
    for shot in shots:
        if not isinstance(shot, Mapping):
            raise RenderError("each shot must be a mapping")
        anchor = shot.get("anchor_image")
        if anchor is not None:
            _existing_local_path(project, anchor, "shot anchor image")
        segments = shot.get("segments")
        if not isinstance(segments, (list, tuple)):
            raise RenderError("shot segments must be a list")
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise RenderError("segment must be a mapping")
            for key in ("opening_image", "opening_frame"):
                if key in segment:
                    _existing_local_path(project, segment[key], f"segment {key}")
    bridges = project.source.get("bridges", ())
    if not isinstance(bridges, (list, tuple)):
        raise RenderError("bridges must be a list")
    for bridge in bridges:
        if not isinstance(bridge, Mapping):
            raise RenderError("bridge must be a mapping")
        base = bridge.get("base_source_image")
        if base is not None:
            _existing_local_path(project, base, "bridge base_source_image")


def _project_workflow_graph(project: ProjectConfig) -> ApiGraph:
    workflow = project.source.get("workflow_api")
    if workflow is None:
        raise RenderError("workflow_api is required")
    return _read_graph(_local_path(project, workflow))


def _segment_request(
    project: ProjectConfig,
    shot_id: str,
    segment_id: str,
    *,
    opening: InputSelection | None = None,
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
        opening=opening or _opening_selection(project, shot_id, shot, segment),
    )


def _bridge_request(
    project: ProjectConfig,
    bridge_id: str,
    *,
    first_image: InputSelection | None = None,
    last_image: InputSelection | None = None,
) -> BridgeRequest:
    bridges = project.source.get("bridges")
    if not isinstance(bridges, (list, tuple)):
        raise RenderError("bridges must be a list")
    bridge = next(
        (
            item
            for item in bridges
            if isinstance(item, Mapping) and item.get("id") == bridge_id
        ),
        None,
    )
    if bridge is None:
        raise RenderError(f"configured bridge was not found: {bridge_id}")
    strategy = bridge.get("strategy", "flf2v")
    if strategy != "flf2v":
        raise RenderError("render-bridge only handles bridges with strategy flf2v")
    request = _mapping(project.source.get("request", {}), "request")
    render = _mapping(project.source.get("render", {}), "render")
    prompt_blocks = _mapping(project.source.get("prompt_blocks", {}), "prompt_blocks")
    frames = _positive_int(
        _first_value(bridge, request, render, keys=("frames", "length")),
        "bridge frames",
    )
    if frames not in {17, 33, 49, 65, 81}:
        raise RenderError("bridge frames must be one of 17, 33, 49, 65, or 81")
    shot_id = bridge.get("shot_id")
    if not isinstance(shot_id, str) or not shot_id:
        raise RenderError("bridge shot_id is required for immutable attempt paths")
    if (first_image is None) != (last_image is None):
        raise RenderError("bridge replay must preserve both endpoint selections")
    if first_image is None:
        first_image, last_image = _bridge_endpoints(project, bridge)
    return BridgeRequest(
        bridge_id=bridge_id,
        shot_id=shot_id,
        positive=_prompt_value(
            request,
            bridge,
            prompt_blocks,
            {},
            ("positive", "positive_prompt", "prompt"),
            "bridge positive prompt",
        ),
        negative=_prompt_value(
            request,
            bridge,
            prompt_blocks,
            {},
            ("negative", "negative_prompt"),
            "bridge negative prompt",
        ),
        seed=_segment_seed(bridge, request, render),
        width=_positive_int(
            _first_value(bridge, request, render, keys=("width",)), "bridge width"
        ),
        height=_positive_int(
            _first_value(bridge, request, render, keys=("height",)), "bridge height"
        ),
        frames=frames,
        first_image=first_image,
        last_image=last_image,
    )


def _bridge_endpoints(
    project: ProjectConfig, bridge: Mapping[str, Any]
) -> tuple[InputSelection, InputSelection]:
    first = bridge.get("first_image")
    last = bridge.get("last_image")
    base = bridge.get("base_source_image")
    if base is not None:
        if first is not None or last is not None:
            raise RenderError(
                "bridge base_source_image cannot be combined with first_image or last_image"
            )
        selection = _local_input_selection(
            project, base, "bridge base_source_image", "technical_smoke_base"
        )
        return selection, selection
    if first is None or last is None:
        raise RenderError(
            "bridge requires explicit first_image and last_image, or one base_source_image"
        )
    first_path = _existing_local_path(project, first, "bridge first_image")
    last_path = _existing_local_path(project, last, "bridge last_image")
    first_selection = _accepted_bridge_endpoint(project, first_path, "tail")
    last_selection = _accepted_bridge_endpoint(project, last_path, "head")
    _validate_story_bridge_endpoint_sources(bridge, first_selection, last_selection)
    return first_selection, last_selection


def _validate_story_bridge_endpoint_sources(
    bridge: Mapping[str, Any],
    first_image: InputSelection,
    last_image: InputSelection,
) -> None:
    """Bind story FLF endpoints to their declared accepted segment attempts."""
    from_segment = bridge.get("from_segment")
    to_segment = bridge.get("to_segment")
    if from_segment is None and to_segment is None:
        return
    shot_id = bridge.get("shot_id")
    if not isinstance(shot_id, str) or not isinstance(from_segment, str) or not isinstance(to_segment, str):
        raise RenderError("story bridge has invalid declared source or destination")
    if first_image.upstream_attempt is None or last_image.upstream_attempt is None:
        raise RenderError("story bridge endpoints have no accepted source provenance")
    try:
        first_attempt = load_attempt(first_image.upstream_attempt)
        last_attempt = load_attempt(last_image.upstream_attempt)
    except ProjectStateError as error:
        raise RenderError("story bridge endpoints have invalid accepted source provenance") from error
    if (
        first_attempt.state is not AttemptState.ACCEPTED
        or first_attempt.shot_id != shot_id
        or first_attempt.segment_id != from_segment
        or last_attempt.state is not AttemptState.ACCEPTED
        or last_attempt.shot_id != shot_id
        or last_attempt.segment_id != to_segment
    ):
        raise RenderError(
            "story bridge endpoints do not match the declared source and destination segments"
        )


def _project_with_bridge_workflow(project: ProjectConfig) -> ProjectConfig:
    workflow = project.source.get("bridge_workflow_api")
    if workflow is None:
        raise RenderError("bridge_workflow_api is required for native FLF rendering")
    source = dict(project.source)
    source["workflow_api"] = str(_local_path(project, workflow))
    return ProjectConfig(path=project.path, source=source)


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
    explicit = segment.get("seed")
    if explicit is not None:
        return _non_negative_int(explicit, "segment seed")
    base = render.get("seed_base")
    offset = _non_negative_int(segment.get("seed_offset", 0), "segment seed_offset")
    if base is not None:
        return _non_negative_int(base, "render seed_base") + offset
    request_seed = request.get("seed")
    if request_seed is None:
        raise RenderError("segment seed or render.seed_base is required")
    return _non_negative_int(request_seed, "request seed") + offset


def _opening_selection(
    project: ProjectConfig,
    shot_id: str,
    shot: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> InputSelection:
    continuation = segment.get("continue_from")
    if continuation is not None:
        if "opening_image" in segment or "opening_frame" in segment:
            raise RenderError("continued segment cannot combine continue_from with an opening override")
        return _continuation_selection(project, shot_id, shot, segment, continuation)
    for key in ("opening_image", "opening_frame"):
        if key in segment:
            return _local_input_selection(
                project, segment[key], f"segment {key}", "segment_override"
            )
    anchor = shot.get("anchor_image")
    if anchor is not None:
        return _local_input_selection(project, anchor, "shot anchor image", "shot_anchor")
    return _project_fallback_selection(project)


def _continuation_selection(
    project: ProjectConfig,
    shot_id: str,
    shot: Mapping[str, Any],
    segment: Mapping[str, Any],
    reference: object,
) -> InputSelection:
    if not isinstance(reference, str) or not reference:
        raise RenderError("continue_from must be a non-empty same-shot segment id")
    segments = shot.get("segments")
    if not isinstance(segments, (list, tuple)):
        raise RenderError(f"configured shot has no segments: {shot_id}")
    segment_id = segment.get("id")
    current_index = next(
        (
            index
            for index, item in enumerate(segments)
            if isinstance(item, Mapping) and item.get("id") == segment_id
        ),
        None,
    )
    reference_index = next(
        (
            index
            for index, item in enumerate(segments)
            if isinstance(item, Mapping) and item.get("id") == reference
        ),
        None,
    )
    if current_index is None or reference_index is None:
        raise RenderError("continue_from must reference a same-shot segment")
    if reference_index >= current_index:
        raise RenderError("continue_from must reference an earlier same-shot segment")
    accepted = [
        attempt
        for attempt in _project_attempts(project)
        if attempt.shot_id == shot_id
        and attempt.segment_id == reference
        and attempt.state is AttemptState.ACCEPTED
    ]
    if len(accepted) != 1:
        raise RenderError("continue_from requires exactly one accepted upstream attempt")
    try:
        accepted_attempt_evidence(project, accepted[0])
        candidate = selected_tail_frame(accepted[0])
    except ProjectStateError as error:
        raise RenderError(f"continue_from cannot use the selected upstream tail: {error}") from error
    return InputSelection(
        path=candidate.path,
        sha256=candidate.sha256,
        source="accepted_tail",
        upstream_attempt=accepted[0].path,
        candidate_kind="tail",
    )


def _accepted_bridge_endpoint(
    project: ProjectConfig, path: Path, kind: str
) -> InputSelection:
    matches = []
    for attempt in _project_attempts(project):
        if attempt.state is not AttemptState.ACCEPTED:
            continue
        try:
            accepted_attempt_evidence(project, attempt)
            candidate = accepted_qc_candidate(attempt, path, kind)
        except ProjectStateError as error:
            if "unambiguous" not in str(error):
                raise RenderError(
                    f"bridge {kind} endpoint accepted evidence is invalid: {error}"
                ) from error
            continue
        matches.append((attempt, candidate))
    if len(matches) != 1:
        raise RenderError(
            "bridge endpoints require accepted hash-verified tail and head QC candidates"
        )
    attempt, candidate = matches[0]
    return InputSelection(
        path=candidate.path,
        sha256=candidate.sha256,
        source="accepted_qc_candidate",
        upstream_attempt=attempt.path,
        candidate_kind=kind,
    )


def _project_attempts(project: ProjectConfig) -> tuple[Attempt, ...]:
    root = project.source.get("attempts_dir", project.path.parent / "attempts")
    if not isinstance(root, (str, Path)):
        raise RenderError("attempts_dir must be a local path")
    attempts_root = Path(root)
    if not attempts_root.is_absolute():
        attempts_root = project.path.parent / attempts_root
    if not attempts_root.is_dir():
        return ()
    try:
        attempts = [load_attempt(path.parent) for path in attempts_root.rglob("attempt.json")]
    except ProjectStateError as error:
        raise RenderError(f"invalid immutable attempt record: {error}") from error
    return tuple(sorted(attempts, key=lambda attempt: str(attempt.path)))


def _stored_input_selection(attempt: Attempt, name: str) -> InputSelection:
    try:
        stored = selected_input(attempt, name)
    except ProjectStateError as error:
        raise RenderError(f"planned attempt cannot resume without {name} provenance: {error}") from error
    source = stored.details.get("source")
    if not isinstance(source, str) or not source:
        raise RenderError(f"planned attempt {name} provenance has no source")
    upstream_raw = stored.details.get("upstream_attempt")
    upstream = Path(upstream_raw) if isinstance(upstream_raw, str) else None
    candidate_kind = stored.details.get("candidate_kind")
    if candidate_kind is not None and not isinstance(candidate_kind, str):
        raise RenderError(f"planned attempt {name} provenance has invalid candidate kind")
    return InputSelection(stored.path, stored.sha256, source, upstream, candidate_kind)


def _local_input_selection(
    project: ProjectConfig, value: object, label: str, source: str
) -> InputSelection:
    path = _existing_local_path(project, value, label).resolve()
    return InputSelection(path, sha256_file(path), source)


def _project_fallback_selection(project: ProjectConfig) -> InputSelection:
    inputs = _mapping(project.source.get("inputs", {}), "inputs")
    value = inputs.get("opening_frame")
    if isinstance(value, Mapping):
        value = value.get("path")
    if value is None:
        raise RenderError("inputs.opening_frame is required")
    return _local_input_selection(project, value, "project opening frame", "project_fallback")


def _existing_local_path(project: ProjectConfig, value: object, label: str) -> Path:
    path = _local_path(project, value)
    if not path.is_file():
        raise RenderError(f"{label} does not exist: {path}")
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


def _patch_bridge_graph(
    graph: ApiGraph,
    bridge: BridgeRequest,
    uploaded_first: str,
    uploaded_last: str,
) -> None:
    _unique_target(graph, ("POSITIVE_PROMPT", "PROMPT_POSITIVE"), "CLIPTextEncode").node[
        "inputs"
    ]["text"] = bridge.positive
    _unique_target(graph, ("NEGATIVE_PROMPT", "PROMPT_NEGATIVE"), "CLIPTextEncode").node[
        "inputs"
    ]["text"] = bridge.negative
    _unique_target(graph, ("BRIDGE_FIRST_IMAGE",), "LoadImage").node["inputs"][
        "image"
    ] = uploaded_first
    _unique_target(graph, ("BRIDGE_LAST_IMAGE",), "LoadImage").node["inputs"][
        "image"
    ] = uploaded_last
    conditioning = _unique_target(
        graph, ("FLF_CONDITIONING",), "WanFirstLastFrameToVideo"
    ).node["inputs"]
    conditioning["width"] = bridge.width
    conditioning["height"] = bridge.height
    conditioning["length"] = bridge.frames
    for title in ("BRIDGE_SAMPLER_HIGH", "BRIDGE_SAMPLER_LOW"):
        _unique_target(graph, (title,), "KSamplerAdvanced").node["inputs"][
            "noise_seed"
        ] = bridge.seed


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
    qc = project.source.get("qc")
    if not isinstance(qc, Mapping):
        raise RenderError("qc must be a mapping")
    value = qc.get("candidate_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RenderError("qc.candidate_count must be a positive integer")
    legacy_candidate_count = project.source.get("candidate_count")
    if legacy_candidate_count is not None:
        if (
            isinstance(legacy_candidate_count, bool)
            or not isinstance(legacy_candidate_count, int)
            or legacy_candidate_count <= 0
        ):
            raise RenderError("candidate_count must be a positive integer")
        if legacy_candidate_count != value:
            raise RenderError("candidate_count conflicts with qc.candidate_count")
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
