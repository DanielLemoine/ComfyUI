from __future__ import annotations

import ipaddress
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .errors import PreflightError


@dataclass(frozen=True)
class PreflightResult:
    artifact_dir: Path
    object_info_path: Path
    active_workflow_dir: Path | None
    native_i2v_template: Path | None
    native_flf_template: Path | None
    status: str
    blockers: tuple[str, ...]


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Any,
        _fp: Any,
        _code: int,
        _message: str,
        _headers: Any,
        new_url: str,
    ) -> Any:
        raise PreflightError(
            f"Local object-info redirects are not allowed: {request.full_url} -> {new_url}"
        )


def collect_preflight(
    *,
    comfy_root: Path,
    comfy_url: str | None,
    artifact_dir: Path,
    object_info: dict[str, object] | None = None,
) -> PreflightResult:
    """Write reproducible local evidence without downloading any artifact."""
    comfy_root = comfy_root.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    object_info_data, object_info_status = _object_info(comfy_url, object_info)
    model_inventory = _model_inventory(comfy_root)
    custom_nodes = _custom_nodes(comfy_root)
    native_i2v_template, native_flf_template, templates = _official_templates(comfy_root)
    active_workflow_dir = _active_workflow_dir(comfy_root)
    blockers = _preflight_blockers(
        object_info_data, native_i2v_template, native_flf_template
    )
    status = "BLOCKED" if blockers else "READY"

    object_info_path = artifact_dir / "object_info.json"
    _write_json(artifact_dir / "environment.json", _environment(comfy_root, comfy_url))
    _write_json(artifact_dir / "model_inventory.json", model_inventory)
    _write_json(artifact_dir / "custom_nodes.json", custom_nodes)
    _write_json(object_info_path, object_info_data)
    (artifact_dir / "official_template_inventory.md").write_text(
        _template_inventory_markdown(
            templates, native_i2v_template, native_flf_template, blockers
        ),
        encoding="utf-8",
        newline="\n",
    )
    (artifact_dir / "preflight_report.md").write_text(
        _preflight_report(
            comfy_root=comfy_root,
            object_info_status=object_info_status,
            active_workflow_dir=active_workflow_dir,
            native_i2v_template=native_i2v_template,
            native_flf_template=native_flf_template,
            model_root_count=len(model_inventory["roots"]),
            custom_node_count=len(custom_nodes["nodes"]),
            status=status,
            blockers=blockers,
        ),
        encoding="utf-8",
        newline="\n",
    )

    return PreflightResult(
        artifact_dir=artifact_dir,
        object_info_path=object_info_path,
        active_workflow_dir=active_workflow_dir,
        native_i2v_template=native_i2v_template,
        native_flf_template=native_flf_template,
        status=status,
        blockers=blockers,
    )


def _object_info(
    comfy_url: str | None, object_info: dict[str, object] | None
) -> tuple[dict[str, object], str]:
    if object_info is not None:
        return object_info, "provided fixture"
    if comfy_url is None:
        return {}, "unavailable: no local ComfyUI URL was provided"

    _require_local_url(comfy_url)
    endpoint = comfy_url.rstrip("/")
    if not endpoint.endswith("/object_info"):
        endpoint = f"{endpoint}/object_info"
    try:
        with build_opener(ProxyHandler({}), _RejectRedirects()).open(
            endpoint, timeout=10
        ) as response:
            payload = json.load(response)
    except OSError as error:
        return {}, f"unavailable: unable to read local object info from {endpoint}: {error}"
    if not isinstance(payload, dict):
        return {}, f"unavailable: local object info from {endpoint} was not a JSON object"
    return payload, f"collected from {endpoint}"


def _require_local_url(comfy_url: str) -> None:
    parsed = urlsplit(comfy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PreflightError(f"ComfyUI URL must be a local HTTP(S) URL: {comfy_url}")
    if parsed.hostname.lower() == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise PreflightError(f"ComfyUI URL must resolve to a loopback address: {comfy_url}")


def _preflight_blockers(
    object_info: dict[str, object],
    native_i2v_template: Path | None,
    native_flf_template: Path | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    required_schemas = ("WanImageToVideo", "WanFirstLastFrameToVideo")
    if not object_info:
        blockers.append(
            "object_info is empty or unavailable; collect local /object_info with "
            "WanImageToVideo and WanFirstLastFrameToVideo schemas"
        )
    else:
        for schema in required_schemas:
            if not isinstance(object_info.get(schema), dict):
                blockers.append(
                    f"required object-info schema {schema} is unavailable; "
                    "install or enable the matching native ComfyUI node"
                )
    if native_i2v_template is None:
        blockers.append(
            "official native Wan I2V template could not be verified; provide a valid "
            "JSON template containing a WanImageToVideo node under an official "
            "ComfyUI template directory"
        )
    if native_flf_template is None:
        blockers.append(
            "official native Wan FLF template could not be verified; provide a valid "
            "JSON template containing a WanFirstLastFrameToVideo node under an "
            "official ComfyUI template directory"
        )
    return tuple(blockers)


def _environment(comfy_root: Path, comfy_url: str | None) -> dict[str, object]:
    return {
        "comfy_root": str(comfy_root),
        "comfy_url": comfy_url,
        "ffmpeg": _command_output(["ffmpeg", "-version"]),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _command_output(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "detail": str(error)}
    output = completed.stdout.strip() or completed.stderr.strip()
    return {"available": completed.returncode == 0, "detail": output}


def _model_inventory(comfy_root: Path) -> dict[str, object]:
    roots = _configured_model_roots(comfy_root)
    return {
        "roots": [
            {
                "configured_path": str(path),
                "files": _relative_files(path),
                "kind": kind,
                "present": path.is_dir(),
            }
            for kind, path in roots
        ]
    }


def _configured_model_roots(comfy_root: Path) -> list[tuple[str, Path]]:
    configured = _parse_extra_model_paths(comfy_root / "extra_model_paths.yaml")
    defaults = [
        (child.name, child)
        for child in (comfy_root / "models").iterdir()
        if child.is_dir()
    ] if (comfy_root / "models").is_dir() else []
    candidates = configured or defaults
    return sorted({(kind, path.resolve()) for kind, path in candidates}, key=lambda item: (item[0], str(item[1])))


def _parse_extra_model_paths(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        return []
    sections: list[tuple[str, Path, list[tuple[str, str]]]] = []
    current_name: str | None = None
    base_path: Path | None = None
    entries: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped or ":" not in stripped:
            continue
        indentation = len(stripped) - len(stripped.lstrip())
        key, value = (part.strip().strip('"\'') for part in stripped.split(":", 1))
        if indentation == 0:
            if current_name is not None and base_path is not None:
                sections.append((current_name, base_path, entries))
            current_name, base_path, entries = key, None, []
        elif current_name is not None and key == "base_path":
            base_path = Path(value)
        elif current_name is not None:
            entries.append((key, value))
    if current_name is not None and base_path is not None:
        sections.append((current_name, base_path, entries))

    roots: list[tuple[str, Path]] = []
    for _section, base_path, entries in sections:
        for kind, relative_path in entries:
            if kind == "base_path" or not relative_path:
                continue
            candidate = Path(relative_path)
            roots.append((kind, candidate if candidate.is_absolute() else base_path / candidate))
    return roots


def _relative_files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()]


def _custom_nodes(comfy_root: Path) -> dict[str, object]:
    custom_nodes_dir = comfy_root / "custom_nodes"
    nodes = []
    if custom_nodes_dir.is_dir():
        nodes = [
            {"name": path.name, "path": str(path.resolve())}
            for path in sorted(custom_nodes_dir.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir()
        ]
    return {"root": str(custom_nodes_dir.resolve()), "nodes": nodes}


def _official_templates(comfy_root: Path) -> tuple[Path | None, Path | None, list[Path]]:
    template_roots = [
        comfy_root / "workflow_templates",
        comfy_root / "web" / "assets" / "workflow_templates",
    ]
    resolved_roots = [root.resolve() for root in template_roots if root.is_dir()]
    templates = sorted(
        {
            path.resolve()
            for root in resolved_roots
            for path in root.rglob("*.json")
            if path.is_file() and path.resolve().is_relative_to(root)
        },
        key=str,
    )
    i2v = next(
        (path for path in templates if _template_has_native_node(path, "WanImageToVideo")),
        None,
    )
    flf = next(
        (
            path
            for path in templates
            if _template_has_native_node(path, "WanFirstLastFrameToVideo")
        ),
        None,
    )
    return i2v, flf, templates


def _template_has_native_node(template_path: Path, expected_node: str) -> bool:
    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return expected_node in _template_node_types(template)


def _template_node_types(value: object) -> set[str]:
    if not isinstance(value, dict):
        return set()
    workflow_nodes = value.get("nodes")
    if isinstance(workflow_nodes, list):
        return {
            node_type
            for node in workflow_nodes
            if isinstance(node, dict)
            and isinstance((node_type := node.get("type")), str)
        }
    return {
        node_type
        for node in value.values()
        if isinstance(node, dict)
        and isinstance(node.get("inputs"), dict)
        and isinstance((node_type := node.get("class_type")), str)
    }


def _active_workflow_dir(comfy_root: Path) -> Path | None:
    candidate = comfy_root / "user" / "default" / "workflows"
    return candidate.resolve() if candidate.is_dir() else None


def _template_inventory_markdown(
    templates: list[Path],
    native_i2v_template: Path | None,
    native_flf_template: Path | None,
    blockers: tuple[str, ...],
) -> str:
    lines = ["# Official Template Inventory", "", "## Official template files"]
    lines.extend(f"- `{path}`" for path in templates)
    if not templates:
        lines.append("- unavailable: no official template directory was found")
    lines.extend(
        [
            "",
            "## Required native Wan templates",
            f"- Native I2V: `{native_i2v_template}`" if native_i2v_template else "- Native I2V: unavailable",
            f"- Native FLF: `{native_flf_template}`" if native_flf_template else "- Native FLF: unavailable",
            "",
            "Community workflow graphs were not substituted for unavailable official templates.",
            "",
            "## Preflight gate",
            "",
        ]
    )
    if blockers:
        lines.extend(f"- BLOCKED: {blocker}" for blocker in blockers)
    else:
        lines.append("- READY: official native templates and object-info schemas were verified")
    lines.append("")
    return "\n".join(lines)


def _preflight_report(
    *,
    comfy_root: Path,
    object_info_status: str,
    active_workflow_dir: Path | None,
    native_i2v_template: Path | None,
    native_flf_template: Path | None,
    model_root_count: int,
    custom_node_count: int,
    status: str,
    blockers: tuple[str, ...],
) -> str:
    lines = [
        "# Wan2.2 Long-Form Preflight",
        "",
        f"- ComfyUI root: `{comfy_root}`",
        f"- Object schema: {object_info_status}",
        f"- Configured model roots: {model_root_count}",
        f"- Custom nodes: {custom_node_count}",
        f"- Active workflow directory: `{active_workflow_dir}`" if active_workflow_dir else "- Active workflow directory: unavailable",
        "",
        "## Preflight status",
        f"- Status: {status}",
        "",
    ]
    if blockers:
        lines.extend(["## BLOCKED", *(f"- BLOCKED: {blocker}" for blocker in blockers), ""])
    else:
        lines.extend(["## READY", "- READY: all required native schemas and templates were verified.", ""])
    lines.extend(
        [
        "## Template gate",
        f"- Native Wan I2V template: `{native_i2v_template}`" if native_i2v_template else "- Native Wan I2V template: unavailable",
        f"- Native Wan FLF template: `{native_flf_template}`" if native_flf_template else "- Native Wan FLF template: unavailable",
        "- No community graph was substituted for an unavailable official template.",
        "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
