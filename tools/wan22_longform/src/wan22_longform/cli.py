from __future__ import annotations

import argparse
from pathlib import Path

from .inventory import collect_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local Wan2.2 preflight evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="write local preflight evidence")
    preflight.add_argument("--comfy-root", required=True, type=Path)
    preflight.add_argument("--comfy-url")
    preflight.add_argument("--artifact-dir", default=Path("artifacts") / "preflight", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "preflight":
        result = collect_preflight(
            comfy_root=arguments.comfy_root,
            comfy_url=arguments.comfy_url,
            artifact_dir=arguments.artifact_dir,
        )
        print(result.artifact_dir)


if __name__ == "__main__":
    main()
