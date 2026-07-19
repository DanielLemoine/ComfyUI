from __future__ import annotations

import json
import shutil
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
    validate_graph_against_object_info,
    validate_two_stage_graph,
)


class RenderError(RuntimeError):
    """Raised when a local render cannot produce reviewable artifacts."""


class RenderClient(Protocol):
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
    graph = _read_graph(attempt.path / "workflow-api.json")
    submission = {"prompt": graph}
    submission_path = attempt.path / "submission-request.json"
    _write_json_exclusive(submission_path, submission)
    _validate_submission(project, graph)
    _write_json_exclusive(
        attempt.path / "submission-provenance.json",
        {
            "request": sha256_file(submission_path),
            "source_manifest": sha256_file(
                next(attempt.path.glob("source-manifest.*"))
            ),
            "workflow": sha256_file(attempt.path / "workflow-api.json"),
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


def _validate_submission(project: ProjectConfig, graph: ApiGraph) -> None:
    presets_path = _local_path(
        project, project.source.get("presets", Path(__file__).parents[2] / "config" / "presets.yaml")
    )
    resolved = resolve_preset(project, load_presets(presets_path))
    validate_lora_policy(resolved, _model_files(project))
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
