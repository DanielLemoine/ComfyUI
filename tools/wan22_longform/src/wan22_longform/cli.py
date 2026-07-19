from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assembly import AssemblyTargets, compare_boundary, execute_assembly_plan, plan_assembly
from .ffmpeg import probe_media
from .inventory import collect_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local Wan2.2 preflight evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="write local preflight evidence")
    preflight.add_argument("--comfy-root", required=True, type=Path)
    preflight.add_argument("--comfy-url")
    preflight.add_argument("--artifact-dir", default=Path("artifacts") / "preflight", type=Path)
    assemble = subparsers.add_parser(
        "assemble",
        help="run a safe local native assembly and post-output duration validation",
    )
    assemble.add_argument("--input", action="append", required=True, type=Path)
    assemble.add_argument("--review-mp4", required=True, type=Path)
    assemble.add_argument("--edit-master-ffv1", required=True, type=Path)
    assemble.add_argument("--edit-master-prores", required=True, type=Path)
    assemble.add_argument("--decision-log", required=True, type=Path)
    assemble.add_argument("--rife-review-mp4", type=Path)
    assemble.add_argument("--request-rife", action="store_true")
    assemble.add_argument("--qc-approved", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "preflight":
        result = collect_preflight(
            comfy_root=arguments.comfy_root,
            comfy_url=arguments.comfy_url,
            artifact_dir=arguments.artifact_dir,
        )
        print(result.artifact_dir)  # noqa: T201
    if arguments.command == "assemble":
        inputs = [probe_media(path) for path in arguments.input]
        decisions = [
            compare_boundary(left.path, right.path)
            for left, right in zip(inputs, inputs[1:])
        ]
        plan = plan_assembly(
            inputs,
            AssemblyTargets(
                review_mp4=arguments.review_mp4,
                edit_master_ffv1=arguments.edit_master_ffv1,
                edit_master_prores=arguments.edit_master_prores,
                rife_review_mp4=arguments.rife_review_mp4,
            ),
            boundary_decisions=decisions,
            request_rife=arguments.request_rife,
            qc_approved=arguments.qc_approved,
        )
        result = execute_assembly_plan(plan, decision_log=arguments.decision_log)
        print(  # noqa: T201
            json.dumps(
                {
                    "expected_frame_count": result.expected_frame_count,
                    "expected_duration": str(result.expected_duration),
                    "output_fps": str(result.output_fps),
                    "rife_ready": str(result.rife_ready) if result.rife_ready else None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
