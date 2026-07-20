from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .assembly import AssemblyTargets, compare_boundary, execute_assembly_plan, plan_assembly
from .comfy_client import ComfyClient
from .config import ProjectConfig, load_project, validate_project_contract
from .ffmpeg import probe_media
from .frames import create_contact_sheet
from .hashing import sha256_file
from .inventory import collect_preflight
from .project import (
    AssemblyRecord,
    AssemblyState,
    Attempt,
    AttemptState,
    assembly_records,
    create_assembly_record,
    load_assembly_record,
    load_attempt,
    transition_assembly_record,
    transition_attempt,
    verify_assembly_record_inputs,
    write_assembly_record_json,
)
from .qc import read_qc
from .render import render_bridge, render_segment, resume_attempt, validate_project


class DeploymentError(ValueError):
    """Raised when a workflow deployment could replace user-owned files."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Wan2.2 long-form operator workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="write local preflight evidence")
    preflight.add_argument("--comfy-root", required=True, type=Path)
    preflight.add_argument("--comfy-url")
    preflight.add_argument("--artifact-dir", default=Path("artifacts") / "preflight", type=Path)
    preflight.set_defaults(handler=_handle_preflight)

    validate = subparsers.add_parser("validate-project", help="validate a project without modifying it")
    validate.add_argument("project", type=Path)
    validate.set_defaults(handler=_handle_validate_project)

    render_segment_parser = subparsers.add_parser(
        "render-segment", help="submit one native I2V segment to local ComfyUI"
    )
    render_segment_parser.add_argument("project", type=Path)
    render_segment_parser.add_argument("shot_id")
    render_segment_parser.add_argument("segment_id")
    _add_client_options(render_segment_parser)
    render_segment_parser.set_defaults(handler=_handle_render_segment)

    render_bridge_parser = subparsers.add_parser(
        "render-bridge", help="submit one native FLF bridge with explicit endpoints"
    )
    render_bridge_parser.add_argument("project", type=Path)
    render_bridge_parser.add_argument("bridge_id")
    _add_client_options(render_bridge_parser)
    render_bridge_parser.set_defaults(handler=_handle_render_bridge)

    render_shot = subparsers.add_parser(
        "render-shot", help="submit each configured I2V segment in a shot"
    )
    render_shot.add_argument("project", type=Path)
    render_shot.add_argument("shot_id")
    _add_client_options(render_shot)
    render_shot.set_defaults(handler=_handle_render_shot)

    sheet = subparsers.add_parser(
        "qc-contact-sheet", help="create one new contact sheet from an attempt's QC frames"
    )
    sheet.add_argument("attempt", type=Path)
    sheet.add_argument("--output", type=Path)
    sheet.set_defaults(handler=_handle_qc_contact_sheet)

    for command, state, help_text in (
        ("accept", AttemptState.ACCEPTED, "accept a reviewed attempt"),
        ("reject", AttemptState.REJECTED, "reject a reviewed attempt"),
        ("retry", AttemptState.RETRY_REQUESTED, "record a retry request for a reviewed attempt"),
    ):
        outcome = subparsers.add_parser(command, help=help_text)
        outcome.add_argument("attempt", type=Path)
        outcome.add_argument("--note", required=True)
        if state is AttemptState.ACCEPTED:
            outcome.add_argument("--continuation-frame", type=Path)
        outcome.set_defaults(handler=_outcome_handler(state))

    assemble = subparsers.add_parser(
        "assemble",
        help="diagnostic-only arbitrary-input assembly; not the reviewed project assembly path",
    )
    assemble.add_argument("--input", action="append", required=True, type=Path)
    assemble.add_argument("--review-mp4", required=True, type=Path)
    assemble.add_argument("--edit-master-ffv1", required=True, type=Path)
    assemble.add_argument("--edit-master-prores", required=True, type=Path)
    assemble.add_argument("--decision-log", required=True, type=Path)
    assemble.add_argument(
        "--diagnostic-only",
        action="store_true",
        required=True,
        help="acknowledge that arbitrary inputs bypass accepted project provenance",
    )
    _add_assembly_options(assemble)
    assemble.set_defaults(handler=_handle_assemble)

    assemble_shot = subparsers.add_parser(
        "assemble-shot", help="assemble explicitly accepted segments in configured shot order"
    )
    assemble_shot.add_argument("project", type=Path)
    assemble_shot.add_argument("shot_id")
    assemble_shot.add_argument("--output-dir", type=Path)
    _add_assembly_options(assemble_shot)
    assemble_shot.set_defaults(handler=_handle_assemble_shot)

    assemble_project = subparsers.add_parser(
        "assemble-project", help="assemble accepted outputs in explicit manifest assembly_order"
    )
    assemble_project.add_argument("project", type=Path)
    assemble_project.add_argument("--output-dir", type=Path)
    _add_assembly_options(assemble_project)
    assemble_project.set_defaults(handler=_handle_assemble_project)

    status = subparsers.add_parser("status", help="show immutable attempt state")
    status.add_argument("target", type=Path, help="project manifest or attempt directory")
    status.set_defaults(handler=_handle_status)

    resume = subparsers.add_parser(
        "resume", help="submit only existing planned attempts from a project"
    )
    resume.add_argument("project", type=Path)
    _add_client_options(resume)
    resume.set_defaults(handler=_handle_resume)

    deploy = subparsers.add_parser(
        "deploy-workflows", help="copy the UI workflows to one explicit user workflow directory"
    )
    deploy.add_argument("--target", required=True, type=Path)
    deploy.add_argument(
        "--force",
        action="store_true",
        help="overwrite only this package's matching workflow files; never delete target files",
    )
    deploy.set_defaults(handler=_handle_deploy_workflows)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


def deploy_workflows(source_dir: Path, target: Path, *, force: bool) -> tuple[Path, ...]:
    """Copy only package workflow files, never deleting a user workflow directory."""
    source = source_dir.resolve()
    raw_target = target if target.is_absolute() else Path.cwd() / target
    _reject_symlink_components(raw_target, "workflow target")
    destination_root = raw_target.resolve()
    if not source.is_dir():
        raise DeploymentError(f"workflow source directory does not exist: {source}")
    if destination_root == source or destination_root.is_relative_to(source):
        raise DeploymentError("workflow target must not be inside the source directory")
    if source.is_relative_to(destination_root):
        raise DeploymentError("workflow target must not be the source directory or its parent")
    if destination_root.exists() and not destination_root.is_dir():
        raise DeploymentError(f"workflow target is not a directory: {destination_root}")
    if destination_root.exists() and not force:
        raise DeploymentError(f"workflow target already exists: {destination_root}; use --force")
    destination_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(raw_target, "workflow target")

    deployed: list[Path] = []
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source)
        target_path = destination_root / relative
        _ensure_deployment_parent(target_path.parent, destination_root)
        if target_path.is_symlink():
            raise DeploymentError(f"workflow target file cannot be a symbolic link: {target_path}")
        if target_path.exists() and not force:
            raise DeploymentError(f"workflow target file already exists: {target_path}; use --force")
        shutil.copy2(source_path, target_path)
        deployed.append(target_path)
    return tuple(deployed)


def _add_client_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="local prompt timeout in seconds (default: 1800 for cold offload loads)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)


def _add_assembly_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rife-review-mp4", type=Path)
    parser.add_argument("--request-rife", action="store_true")
    parser.add_argument("--qc-approved", action="store_true")


def _handle_preflight(args: argparse.Namespace) -> int:
    result = collect_preflight(
        comfy_root=args.comfy_root,
        comfy_url=args.comfy_url,
        artifact_dir=args.artifact_dir,
    )
    print(result.artifact_dir)  # noqa: T201
    return 0


def _handle_validate_project(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    _print_json({"project": str(project.path), **validate_project(project)})
    return 0


def _handle_render_segment(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    validate_project(project)
    attempt = render_segment(
        project,
        args.shot_id,
        args.segment_id,
        _client(args),
    )
    _print_json(_attempt_payload(attempt))
    return 0


def _handle_render_bridge(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    validate_project(project)
    attempt = render_bridge(project, args.bridge_id, _client(args))
    _print_json(_attempt_payload(attempt))
    return 0


def _handle_render_shot(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    validate_project(project)
    shot = _find_shot(project, args.shot_id)
    segments = shot.get("segments")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError(f"shot has no renderable segments: {args.shot_id}")
    if any(
        isinstance(segment, Mapping) and segment.get("continue_from") is not None
        for segment in segments
    ):
        raise ValueError(
            "render-shot refuses dependent continuations; accept an upstream tail and use render-segment"
        )
    client = _client(args)
    rendered: list[dict[str, str]] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("id"), str):
            raise ValueError(f"shot has an invalid segment: {args.shot_id}")
        attempt = render_segment(project, args.shot_id, segment["id"], client)
        rendered.append(_attempt_payload(attempt))
    _print_json({"attempts": rendered, "shot_id": args.shot_id})
    return 0


def _handle_qc_contact_sheet(args: argparse.Namespace) -> int:
    attempt_path = args.attempt.resolve()
    qc = read_qc(attempt_path / "qc.yaml")
    frames = _qc_candidate_paths(qc)
    output = args.output.resolve() if args.output else attempt_path / "contact-sheet-rerun.png"
    if output.exists():
        raise ValueError(f"contact sheet output already exists: {output}")
    sheet = create_contact_sheet(frames, output)
    _print_json({"contact_sheet": str(sheet)})
    return 0


def _outcome_handler(state: AttemptState) -> Callable[[argparse.Namespace], int]:
    def handle(args: argparse.Namespace) -> int:
        attempt = load_attempt(args.attempt.resolve())
        selected = None
        if state is AttemptState.ACCEPTED and args.continuation_frame is not None:
            selected = _validated_continuation_frame(
                attempt.path, args.continuation_frame.resolve()
            )
        updated = transition_attempt(attempt, state, args.note, selected_continuation_frame=selected)
        _print_json(_attempt_payload(updated))
        return 0

    return handle


def _handle_assemble(args: argparse.Namespace) -> int:
    result = _execute_assembly(
        [path.resolve() for path in args.input],
        AssemblyTargets(
            review_mp4=args.review_mp4.resolve(),
            edit_master_ffv1=args.edit_master_ffv1.resolve(),
            edit_master_prores=args.edit_master_prores.resolve(),
            rife_review_mp4=args.rife_review_mp4.resolve() if args.rife_review_mp4 else None,
        ),
        decision_log=args.decision_log.resolve(),
        request_rife=args.request_rife,
        qc_approved=args.qc_approved,
    )
    _print_json(_assembly_payload(result))
    return 0


def _handle_assemble_shot(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    _safe_path_component(args.shot_id, "shot_id")
    validate_project_contract(project)
    attempts = _accepted_shot_attempts(project, args.shot_id)
    return _assemble_attempts(project, attempts, f"shot-{args.shot_id}", args)


def _handle_assemble_project(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    validate_project_contract(project)
    attempts = _accepted_project_attempts(project)
    return _assemble_attempts(project, attempts, "project", args)


def _handle_status(args: argparse.Namespace) -> int:
    target = args.target.resolve()
    if target.is_file():
        project = load_project(target)
        _print_json(
            {
                "assembly_records": [
                    _assembly_record_payload(record) for record in assembly_records(project)
                ],
                "attempts": [_attempt_payload(attempt) for attempt in _project_attempts(project)],
                "project": str(project.path),
            }
        )
        return 0
    attempt = load_attempt(target)
    _print_json(_attempt_payload(attempt))
    return 0


def _handle_resume(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    validate_project_contract(project)
    planned = [
        attempt for attempt in _project_attempts(project) if attempt.state is AttemptState.PLANNED
    ]
    resumed: list[dict[str, str]] = []
    if planned:
        client = _client(args)
        for attempt in planned:
            resumed.append(_attempt_payload(resume_attempt(project, attempt, client)))
    records = assembly_records(project)
    _print_json(
        {
            "assembly_records": [_assembly_record_payload(record) for record in records],
            "incomplete_assembly_records": [
                _assembly_recovery_payload(record)
                for record in records
                if record.state is not AssemblyState.FINAL
            ],
            "project": str(project.path),
            "resumed": resumed,
        }
    )
    return 0


def _handle_deploy_workflows(args: argparse.Namespace) -> int:
    source = Path(__file__).resolve().parents[2] / "workflows" / "ui"
    deployed = deploy_workflows(source, args.target, force=args.force)
    _print_json({"deployed": [str(path) for path in deployed], "target": str(args.target.resolve())})
    return 0


def _client(args: argparse.Namespace) -> ComfyClient:
    return ComfyClient(
        args.comfy_url,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )


def _find_shot(project: ProjectConfig, shot_id: str) -> Mapping[str, Any]:
    shots = project.source.get("shots")
    if not isinstance(shots, (list, tuple)):
        raise ValueError("project shots must be a list")
    shot = next(
        (
            item
            for item in shots
            if isinstance(item, Mapping) and item.get("id") == shot_id
        ),
        None,
    )
    if shot is None:
        raise ValueError(f"configured shot was not found: {shot_id}")
    return shot


def _qc_candidate_paths(
    qc: Mapping[str, Any], *, locations: tuple[str, ...] = ("head", "tail")
) -> list[Path]:
    candidates = qc.get("candidate_frames")
    if not isinstance(candidates, Mapping):
        raise ValueError("QC record has no candidate_frames mapping")
    frames: list[Path] = []
    for location in locations:
        entries = candidates.get(location)
        if not isinstance(entries, list):
            raise ValueError(f"QC record has no {location} candidate list")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise ValueError("QC candidate entry has no local path")
            path = Path(entry["path"])
            if not path.is_file():
                raise ValueError(f"QC candidate frame does not exist: {path}")
            if entry.get("sha256") != sha256_file(path):
                raise ValueError(f"QC candidate frame hash changed: {path}")
            frames.append(path)
    return frames


def _validated_continuation_frame(attempt_path: Path, selected: Path) -> Path:
    if not selected.is_file():
        raise ValueError(f"selected continuation frame does not exist: {selected}")
    candidates = _qc_candidate_paths(read_qc(attempt_path / "qc.yaml"), locations=("tail",))
    if selected not in candidates:
        raise ValueError("selected continuation frame is not a hash-verified QC tail candidate")
    return selected


def _project_attempts(project: ProjectConfig) -> tuple[Attempt, ...]:
    root = _attempts_root(project)
    if not root.is_dir():
        return ()
    attempts = [load_attempt(path.parent) for path in root.rglob("attempt.json")]
    return tuple(sorted(attempts, key=lambda attempt: str(attempt.path)))


def _attempts_root(project: ProjectConfig) -> Path:
    configured = project.source.get("attempts_dir", project.path.parent / "attempts")
    if not isinstance(configured, (str, Path)):
        raise ValueError("attempts_dir must be a local path")
    root = Path(configured)
    return root if root.is_absolute() else project.path.parent / root


def _accepted_shot_attempts(project: ProjectConfig, shot_id: str) -> tuple[Attempt, ...]:
    shot = _find_shot(project, shot_id)
    segments = shot.get("segments")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError(f"shot has no configured segments: {shot_id}")
    accepted = _accepted_by_key(project)
    ordered: list[Attempt] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("id"), str):
            raise ValueError(f"shot has an invalid segment: {shot_id}")
        key = (shot_id, segment["id"])
        try:
            ordered.append(accepted[key])
        except KeyError as error:
            raise ValueError(f"shot segment is not accepted: {shot_id}/{segment['id']}") from error
    return tuple(ordered)


def _accepted_project_attempts(project: ProjectConfig) -> tuple[Attempt, ...]:
    order = project.source.get("assembly_order")
    if not isinstance(order, (list, tuple)) or not order:
        raise ValueError("assemble-project requires a non-empty explicit assembly_order")
    accepted = _accepted_by_key(project)
    selected: list[Attempt] = []
    for item in order:
        if not isinstance(item, Mapping):
            raise ValueError("assembly_order entries must be mappings")
        shot_id = item.get("shot_id")
        segment_id = item.get("segment_id")
        if not isinstance(shot_id, str) or not isinstance(segment_id, str):
            raise ValueError("assembly_order entries require shot_id and segment_id")
        key = (shot_id, segment_id)
        try:
            selected.append(accepted[key])
        except KeyError as error:
            raise ValueError(f"assembly_order item is not accepted: {shot_id}/{segment_id}") from error
    return tuple(selected)


def _accepted_by_key(project: ProjectConfig) -> dict[tuple[str, str], Attempt]:
    selected: dict[tuple[str, str], Attempt] = {}
    for attempt in _project_attempts(project):
        if attempt.state is not AttemptState.ACCEPTED:
            continue
        key = (attempt.shot_id, attempt.segment_id)
        if key in selected:
            raise ValueError(
                f"multiple accepted attempts for {attempt.shot_id}/{attempt.segment_id}; "
                "reject one before assembly"
            )
        selected[key] = attempt
    return selected


def _assemble_attempts(
    project: ProjectConfig,
    attempts: Sequence[Attempt],
    scope: str,
    args: argparse.Namespace,
) -> int:
    if not attempts:
        raise ValueError("assembly requires at least one accepted attempt")
    _safe_path_component(scope, "assembly scope")
    selected = [_accepted_assembly_input(attempt) for attempt in attempts]
    videos = [video for video, _ in selected]
    record = create_assembly_record(
        project,
        scope,
        inputs=tuple(input_record for _, input_record in selected),
        requested={
            "qc_approved": bool(args.qc_approved),
            "request_rife": bool(args.request_rife),
            "requested_output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
            "scope": scope,
        },
    )
    output_dir = args.output_dir.resolve() if args.output_dir else record.path / "outputs"
    if output_dir.is_file():
        _mark_assembly_failed(record, ValueError(f"assembly output path is a file: {output_dir}"))
        raise ValueError(f"assembly output path is a file: {output_dir}")
    targets = AssemblyTargets(
        review_mp4=output_dir / f"{scope}-review.mp4",
        edit_master_ffv1=output_dir / f"{scope}-edit-master.ffv1.mkv",
        edit_master_prores=output_dir / f"{scope}-edit-master.prores.mov",
        rife_review_mp4=(output_dir / f"{scope}-rife-review.mp4")
        if args.request_rife
        else None,
    )
    try:
        result, finalized = _execute_assembly_record(
            record,
            videos,
            targets,
            request_rife=args.request_rife,
            qc_approved=args.qc_approved,
        )
    except Exception as error:
        _mark_assembly_failed(record, error)
        raise
    _print_json(
        {
            **_assembly_payload(result),
            "assembly_record": _assembly_record_payload(finalized),
            "attempts": [_attempt_payload(attempt) for attempt in attempts],
            "scope": scope,
        }
    )
    return 0


def _accepted_video(attempt: Attempt) -> Path:
    metadata_path = attempt.path / "render-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"accepted attempt has no valid render metadata: {attempt.path}") from error
    outputs = metadata.get("outputs") if isinstance(metadata, Mapping) else None
    if not isinstance(outputs, Mapping):
        raise ValueError(f"accepted attempt has no outputs: {attempt.path}")
    candidates = [outputs[name] for name in ("segment", "bridge") if name in outputs]
    if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ValueError(f"accepted attempt has no unambiguous video output: {attempt.path}")
    raw_path = candidates[0].get("path")
    expected_hash = candidates[0].get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"accepted attempt has invalid video provenance: {attempt.path}")
    video = Path(raw_path)
    if not video.is_file() or sha256_file(video) != expected_hash:
        raise ValueError(f"accepted attempt video changed or is missing: {video}")
    return video


def _accepted_assembly_input(attempt: Attempt) -> tuple[Path, dict[str, Any]]:
    video = _accepted_video(attempt)
    return (
        video,
        {
            "attempt_id": attempt.attempt_id,
            "attempt_path": attempt.path,
            "output": {"path": video, "sha256": sha256_file(video)},
            "segment_id": attempt.segment_id,
            "shot_id": attempt.shot_id,
        },
    )


def _execute_assembly(
    input_paths: Sequence[Path],
    targets: AssemblyTargets,
    *,
    decision_log: Path,
    request_rife: bool,
    qc_approved: bool,
):
    plan = _plan_assembly(
        input_paths,
        targets,
        request_rife=request_rife,
        qc_approved=qc_approved,
    )
    return execute_assembly_plan(plan, decision_log=decision_log)


def _execute_assembly_record(
    record: AssemblyRecord,
    input_paths: Sequence[Path],
    targets: AssemblyTargets,
    *,
    request_rife: bool,
    qc_approved: bool,
) -> tuple[Any, AssemblyRecord]:
    verified = verify_assembly_record_inputs(record)
    verified_paths = tuple(Path(item["output"]["path"]).resolve() for item in verified)
    requested_paths = tuple(path.resolve() for path in input_paths)
    if verified_paths != requested_paths:
        raise ValueError("assembly record inputs no longer match the accepted source selection")
    plan = _plan_assembly(
        input_paths,
        targets,
        request_rife=request_rife,
        qc_approved=qc_approved,
    )
    decision_log = record.path / "boundary-decisions.json"
    write_assembly_record_json(
        record,
        "assembly-plan.json",
        _assembly_plan_payload(plan, targets, decision_log),
    )
    assembling = transition_assembly_record(
        record,
        AssemblyState.ASSEMBLING,
        "local FFmpeg assembly started",
    )
    result = execute_assembly_plan(plan, decision_log=decision_log)
    write_assembly_record_json(
        assembling,
        "outputs.json",
        _assembly_outputs_payload(targets, decision_log, result),
    )
    assembled = transition_assembly_record(
        assembling,
        AssemblyState.ASSEMBLED,
        "all requested outputs passed post-output validation",
    )
    final = transition_assembly_record(
        assembled,
        AssemblyState.FINAL,
        "assembly evidence finalized",
    )
    return result, final


def _plan_assembly(
    input_paths: Sequence[Path],
    targets: AssemblyTargets,
    *,
    request_rife: bool,
    qc_approved: bool,
):
    inputs = [probe_media(path) for path in input_paths]
    decisions = [compare_boundary(left.path, right.path) for left, right in zip(inputs, inputs[1:])]
    return plan_assembly(
        inputs,
        targets,
        boundary_decisions=decisions,
        request_rife=request_rife,
        qc_approved=qc_approved,
    )


def _attempt_payload(attempt: Attempt) -> dict[str, str]:
    return {
        "attempt_id": attempt.attempt_id,
        "path": str(attempt.path),
        "segment_id": attempt.segment_id,
        "shot_id": attempt.shot_id,
        "state": str(attempt.state),
    }


def _assembly_payload(result: Any) -> dict[str, Any]:
    return {
        "expected_frame_count": result.expected_frame_count,
        "expected_duration": str(result.expected_duration),
        "output_fps": str(result.output_fps),
        "rife_ready": str(result.rife_ready) if result.rife_ready else None,
    }


def _assembly_record_payload(record: AssemblyRecord) -> dict[str, str]:
    return {
        "assembly_id": record.assembly_id,
        "path": str(record.path),
        "scope": record.scope,
        "state": str(record.state),
    }


def _assembly_recovery_payload(record: AssemblyRecord) -> dict[str, str]:
    action = {
        AssemblyState.PLANNED: "plan is preserved; start a new reviewed assembly request explicitly",
        AssemblyState.ASSEMBLING: "inspect the preserved record and partial targets; do not rerun automatically",
        AssemblyState.ASSEMBLED: "inspect validated outputs before explicitly finalizing the record",
        AssemblyState.FAILED: "inspect the preserved failure and use a new output target for another request",
    }.get(record.state)
    if action is None:
        raise ValueError(f"assembly recovery requested for a final record: {record.path}")
    return {**_assembly_record_payload(record), "recovery_action": action}


def _mark_assembly_failed(record: AssemblyRecord, error: Exception) -> None:
    """Preserve a failed assembly outcome without retrying or replacing its targets."""
    current = load_assembly_record(record.path)
    if current.state not in {AssemblyState.PLANNED, AssemblyState.ASSEMBLING}:
        return
    detail = str(error).strip() or type(error).__name__
    transition_assembly_record(
        current,
        AssemblyState.FAILED,
        f"assembly failed: {detail}",
        details={"error_type": type(error).__name__, "message": detail},
    )


def _assembly_plan_payload(plan: Any, targets: AssemblyTargets, decision_log: Path) -> dict[str, Any]:
    return {
        "audio_policy": plan.audio_policy,
        "boundary_decisions": [dict(decision.__dict__) for decision in plan.boundary_decisions],
        "concat_manifest": str(plan.concat_manifest),
        "decision_log": str(decision_log),
        "expected_duration": str(plan.expected_duration) if plan.expected_duration else None,
        "expected_frame_count": plan.expected_frame_count,
        "normalize_first": plan.normalize_first,
        "operations": [
            {
                "command": list(operation.command),
                "expected_duration": (
                    str(operation.expected_duration) if operation.expected_duration else None
                ),
                "inputs": [str(path) for path in operation.inputs],
                "kind": operation.kind,
                "output": str(operation.output),
                "output_fps": str(operation.output_fps) if operation.output_fps else None,
            }
            for operation in plan.operations
        ],
        "output_fps": str(plan.output_fps),
        "targets": _assembly_targets_payload(targets),
    }


def _assembly_outputs_payload(
    targets: AssemblyTargets,
    decision_log: Path,
    result: Any,
) -> dict[str, Any]:
    outputs = {
        name: _hashed_assembly_path(path, name)
        for name, path in (
            ("review_mp4", targets.review_mp4),
            ("edit_master_ffv1", targets.edit_master_ffv1),
            ("edit_master_prores", targets.edit_master_prores),
        )
    }
    if targets.rife_review_mp4 is not None and targets.rife_review_mp4.is_file():
        outputs["rife_review_mp4"] = _hashed_assembly_path(
            targets.rife_review_mp4, "rife_review_mp4"
        )
    return {
        "boundary_decisions": _hashed_assembly_path(decision_log, "boundary decisions"),
        "execution": _assembly_payload(result),
        "outputs": outputs,
    }


def _assembly_targets_payload(targets: AssemblyTargets) -> dict[str, str | None]:
    return {
        "edit_master_ffv1": str(targets.edit_master_ffv1),
        "edit_master_prores": str(targets.edit_master_prores),
        "review_mp4": str(targets.review_mp4),
        "rife_review_mp4": str(targets.rife_review_mp4) if targets.rife_review_mp4 else None,
    }


def _hashed_assembly_path(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"assembly {label} was not produced: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))  # noqa: T201


def _safe_path_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a single path component")


def _ensure_deployment_parent(parent: Path, root: Path) -> None:
    _reject_symlink_components(parent, "workflow target directory")
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve()
    if not resolved_parent.is_relative_to(root.resolve()):
        raise DeploymentError(f"workflow target escapes deployment root: {parent}")


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise DeploymentError(f"{label} cannot contain a symbolic link: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


if __name__ == "__main__":
    raise SystemExit(main())
