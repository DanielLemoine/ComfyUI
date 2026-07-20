from __future__ import annotations

import hashlib
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

from .config import ProjectConfig
from .errors import PreflightError


_CANONICAL_I2V_TEMPLATE_ID = "video_wan2_2_14B_i2v"
_CANONICAL_I2V_TEMPLATE_FILENAME = f"{_CANONICAL_I2V_TEMPLATE_ID}.json"
_CANONICAL_I2V_TEMPLATE_SHA256 = (
    "6eea9b627b10fcfaf3e75a43aad2c58d8daabdbf72b32ede1602c668cac376bb"
)
_CANONICAL_FLF_TEMPLATE_ID = "video_wan2_2_14B_flf2v"
_CANONICAL_FLF_TEMPLATE_FILENAME = f"{_CANONICAL_FLF_TEMPLATE_ID}.json"
_CANONICAL_FLF_TEMPLATE_SHA256 = (
    "9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf"
)


@dataclass(frozen=True)
class PreflightResult:
    artifact_dir: Path
    object_info_path: Path
    active_workflow_dir: Path | None
    native_i2v_template: Path | None
    native_flf_template: Path | None
    status: str
    blockers: tuple[str, ...]
    workflow_candidates: tuple[Path, ...]


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
    project: ProjectConfig | None = None,
) -> PreflightResult:
    """Write reproducible local evidence without downloading any artifact."""
    comfy_root = comfy_root.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    object_info_data, object_info_status = _object_info(comfy_url, object_info)
    model_inventory = _model_inventory(comfy_root, project)
    custom_nodes = _custom_nodes(comfy_root)
    (
        native_i2v_template,
        native_flf_template,
        templates,
        i2v_verification,
        flf_verification,
    ) = _official_templates(comfy_root)
    active_workflow_dir, workflow_candidates = _active_workflow_dir(comfy_root)
    blockers = _preflight_blockers(
        object_info_data,
        native_i2v_template,
        native_flf_template,
        i2v_verification,
        flf_verification,
        model_inventory,
        project,
    )
    status = "BLOCKED" if blockers else "READY"

    object_info_path = artifact_dir / "object_info.json"
    _write_json(artifact_dir / "environment.json", _environment(comfy_root, comfy_url))
    _write_json(artifact_dir / "model_inventory.json", model_inventory)
    _write_json(artifact_dir / "custom_nodes.json", custom_nodes)
    _write_json(object_info_path, object_info_data)
    (artifact_dir / "official_template_inventory.md").write_text(
        _template_inventory_markdown(
            templates,
            native_i2v_template,
            native_flf_template,
            i2v_verification,
            flf_verification,
            blockers,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (artifact_dir / "preflight_report.md").write_text(
        _preflight_report(
            comfy_root=comfy_root,
            object_info_status=object_info_status,
            active_workflow_dir=active_workflow_dir,
            workflow_candidates=workflow_candidates,
            native_i2v_template=native_i2v_template,
            native_flf_template=native_flf_template,
            i2v_verification=i2v_verification,
            flf_verification=flf_verification,
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
        workflow_candidates=workflow_candidates,
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
    except json.JSONDecodeError as error:
        return {}, f"unavailable: local object info from {endpoint} was not valid JSON: {error}"
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
    i2v_verification: str,
    flf_verification: str,
    model_inventory: dict[str, object],
    project: ProjectConfig | None,
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
            if schema not in object_info:
                blockers.append(
                    f"required object-info schema {schema} is unavailable; "
                    "install or enable the matching native ComfyUI node"
                )
            elif not _has_required_input_schema(object_info[schema]):
                blockers.append(
                    f"required object-info schema {schema} lacks a non-empty "
                    "input.required mapping; collect the complete native node schema"
                )
    if native_i2v_template is None:
        blockers.append(
            "official native Wan I2V template could not be verified; "
            f"{i2v_verification}"
        )
    if native_flf_template is None:
        blockers.append(
            "official native Wan FLF template could not be verified; "
            f"{flf_verification}"
        )
    blockers.extend(_model_role_blockers(model_inventory, project))
    return tuple(blockers)


def _has_required_input_schema(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    input_spec = schema.get("input")
    if not isinstance(input_spec, dict):
        return False
    required_inputs = input_spec.get("required")
    return isinstance(required_inputs, dict) and bool(required_inputs)


def _environment(comfy_root: Path, comfy_url: str | None) -> dict[str, object]:
    return {
        "comfyui_revision": _command_output(
            ["git", "-C", str(comfy_root), "rev-parse", "HEAD"]
        ),
        "comfy_root": str(comfy_root),
        "comfy_url": comfy_url,
        "cuda_version": _command_output(
            [sys.executable, "-c", "import torch; print(torch.version.cuda or 'unavailable')"]
        ),
        "ffmpeg": _command_output(["ffmpeg", "-version"]),
        "frontend_version": _command_output(
            [
                sys.executable,
                "-c",
                "from importlib.metadata import version; print(version('comfyui-frontend-package'))",
            ]
        ),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "platform": platform.platform(),
        "pytorch_version": _command_output(
            [sys.executable, "-c", "import torch; print(torch.__version__)"]
        ),
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


def _model_inventory(
    comfy_root: Path, project: ProjectConfig | None = None
) -> dict[str, object]:
    roots = _configured_model_roots(comfy_root)
    payload: dict[str, object] = {
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
    payload["required_roles"] = _required_model_roles(project)
    return payload


def _required_model_roles(project: ProjectConfig | None) -> dict[str, str]:
    if project is None:
        return {}
    models = project.source.get("models")
    if not isinstance(models, dict):
        return {}
    return {
        role: value
        for role in ("high", "low", "vae", "text_encoder")
        if isinstance((value := models.get(role)), str) and value
    }


def _model_role_blockers(
    model_inventory: dict[str, object], project: ProjectConfig | None
) -> tuple[str, ...]:
    if project is None:
        return (
            "project model-role proof is unavailable; provide a validated project manifest "
            "for high, low, VAE, and text-encoder inventory checks",
        )
    required = _required_model_roles(project)
    missing_roles = [
        role for role in ("high", "low", "vae", "text_encoder") if role not in required
    ]
    if missing_roles:
        return tuple(f"project model role {role} is unavailable" for role in missing_roles)
    roots = model_inventory.get("roots")
    root_entries = roots if isinstance(roots, list) else []
    discovered = {
        Path(name).name.casefold()
        for root in root_entries
        if isinstance(root, dict)
        for name in root.get("files", [])
        if isinstance(root.get("files"), list) and isinstance(name, str)
    }
    return tuple(
        f"configured project model {role} is absent from local model inventory: {name}"
        for role, name in required.items()
        if name.casefold() not in discovered
    )


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
            {
                "name": path.name,
                "path": str(path.resolve()),
                "revision": _command_output(
                    ["git", "-C", str(path.resolve()), "rev-parse", "HEAD"]
                ),
            }
            for path in sorted(custom_nodes_dir.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir() and path.name.casefold() not in {"__pycache__", "test", "tests"}
        ]
    return {"root": str(custom_nodes_dir.resolve()), "nodes": nodes}


def _official_templates(
    comfy_root: Path,
) -> tuple[Path | None, Path | None, list[Path], str, str]:
    template_roots = [
        comfy_root / "blueprints",
        comfy_root / "workflow_templates",
        comfy_root / "web" / "assets" / "workflow_templates",
        comfy_root
        / "venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates",
        comfy_root
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates",
        comfy_root
        / "python_embeded"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates",
        comfy_root
        / "python_embedded"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates",
    ]
    for environment in (comfy_root / "venv", comfy_root / ".venv"):
        template_roots.extend(
            environment.glob(
                "lib/python*/site-packages/comfyui_workflow_templates_json/templates"
            )
        )
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
    i2v, i2v_verification = _find_native_i2v_template(templates)
    flf, flf_verification = _find_native_flf_template(templates)
    return i2v, flf, templates, i2v_verification, flf_verification


def _find_native_i2v_template(templates: list[Path]) -> tuple[Path | None, str]:
    return _find_registered_native_template(
        templates,
        template_id=_CANONICAL_I2V_TEMPLATE_ID,
        filename=_CANONICAL_I2V_TEMPLATE_FILENAME,
        pinned_sha256=_CANONICAL_I2V_TEMPLATE_SHA256,
        native_node="WanImageToVideo",
        label="I2V",
    )


def _find_native_flf_template(templates: list[Path]) -> tuple[Path | None, str]:
    return _find_registered_native_template(
        templates,
        template_id=_CANONICAL_FLF_TEMPLATE_ID,
        filename=_CANONICAL_FLF_TEMPLATE_FILENAME,
        pinned_sha256=_CANONICAL_FLF_TEMPLATE_SHA256,
        native_node="WanFirstLastFrameToVideo",
        label="FLF",
    )


def _find_registered_native_template(
    templates: list[Path],
    *,
    template_id: str,
    filename: str,
    pinned_sha256: str,
    native_node: str,
    label: str,
) -> tuple[Path | None, str]:
    package_candidates = [
        path
        for path in templates
        if (
            _package_template_root(path) is not None
            and path.name == filename
        )
    ]
    if not package_candidates:
        return (
            None,
            f"registered canonical Wan {label} package asset {filename} was not found; "
            f"topology-only {label} JSON in blueprints, workflow_templates, or web assets "
            "is not trusted",
        )
    package_failures: list[str] = []
    for path in package_candidates:
        verified, detail = _verify_registered_package_template(
            path,
            template_id=template_id,
            filename=filename,
            pinned_sha256=pinned_sha256,
            label=label,
        )
        if not verified:
            package_failures.append(detail)
            continue
        if _template_has_native_node(path, native_node):
            return path, detail
        package_failures.append(
            f"the registered canonical Wan {label} package asset is missing its "
            f"{native_node} node"
        )
    return None, "; ".join(package_failures)


def _package_template_root(template_path: Path) -> Path | None:
    for ancestor in template_path.parents:
        if (
            ancestor.name == "templates"
            and ancestor.parent.name == "comfyui_workflow_templates_json"
        ):
            return ancestor
    return None


def _verify_registered_package_template(
    template_path: Path,
    *,
    template_id: str,
    filename: str,
    pinned_sha256: str,
    label: str,
) -> tuple[bool, str]:
    package_root = _package_template_root(template_path)
    if package_root is None:
        return False, f"the {label} template is not inside the ComfyUI workflow-template package"
    manifest_path = (
        package_root.parent.parent
        / "comfyui_workflow_templates_core"
        / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return (
            False,
            f"the package {label} JSON has no readable registered ComfyUI manifest entry "
            f"at {manifest_path}: {error}",
        )
    manifest_hash = _registered_manifest_hash(manifest, template_id, filename)
    if manifest_hash is None:
        return (
            False,
            f"the package {label} JSON has no registered ComfyUI manifest entry for "
            f"{template_id}/{filename}",
        )
    if manifest_hash.casefold() != pinned_sha256:
        return (
            False,
            "the registered ComfyUI manifest SHA-256 for "
            f"{filename} does not match the pinned official hash",
        )
    try:
        local_hash = _sha256_file(template_path)
    except OSError as error:
        return False, f"could not calculate the package {label} SHA-256: {error}"
    if local_hash.casefold() != pinned_sha256:
        return (
            False,
            f"the package {label} JSON SHA-256 does not match its registered and pinned "
            "official hash",
        )
    return (
        True,
        "registered ComfyUI manifest entry at "
        f"{manifest_path} and pinned SHA-256 were verified locally",
    )


def _registered_manifest_hash(
    manifest: object, template_id: str, filename: str
) -> str | None:
    if not isinstance(manifest, dict):
        return None
    templates = manifest.get("templates")
    if not isinstance(templates, list):
        return None
    for entry in templates:
        if not isinstance(entry, dict) or entry.get("id") != template_id:
            continue
        assets = entry.get("assets")
        if not isinstance(assets, list):
            return None
        for asset in assets:
            if (
                isinstance(asset, dict)
                and asset.get("filename") == filename
                and isinstance(asset.get("sha256"), str)
            ):
                return asset["sha256"]
        return None
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    definitions = value.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    if isinstance(workflow_nodes, list) or isinstance(subgraphs, list):
        node_types = _workflow_node_types(workflow_nodes)
        if isinstance(subgraphs, list):
            for subgraph in subgraphs:
                if isinstance(subgraph, dict):
                    node_types.update(_workflow_node_types(subgraph.get("nodes")))
        return node_types
    return {
        node_type
        for node in value.values()
        if isinstance(node, dict)
        and isinstance(node.get("inputs"), dict)
        and isinstance((node_type := node.get("class_type")), str)
    }


def _workflow_node_types(nodes: object) -> set[str]:
    if not isinstance(nodes, list):
        return set()
    return {
        node_type
        for node in nodes
        if isinstance(node, dict) and isinstance((node_type := node.get("type")), str)
    }


def _active_workflow_dir(comfy_root: Path) -> tuple[Path | None, tuple[Path, ...]]:
    user_root = comfy_root / "user"
    candidates = tuple(
        sorted(
            (
                path.resolve()
                for path in user_root.glob("*/workflows")
                if path.is_dir()
            ),
            key=str,
        )
    )
    return None, candidates


def _template_inventory_markdown(
    templates: list[Path],
    native_i2v_template: Path | None,
    native_flf_template: Path | None,
    i2v_verification: str,
    flf_verification: str,
    blockers: tuple[str, ...],
) -> str:
    lines = ["# Template Discovery Inventory", "", "## Candidate template files"]
    lines.extend(f"- `{path}`" for path in templates)
    if not templates:
        lines.append("- unavailable: no official template directory was found")
    lines.extend(
        [
            "",
            "## Required native Wan templates",
            f"- Native I2V: `{native_i2v_template}`" if native_i2v_template else "- Native I2V: unavailable",
            f"- Native FLF: `{native_flf_template}`" if native_flf_template else "- Native FLF: unavailable",
            f"- Native I2V verification: {i2v_verification}",
            f"- Native FLF verification: {flf_verification}",
            "",
            "I2V and FLF accept only registered, hash-verified canonical package assets; "
            "community or local workflow graphs are not substitutes.",
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
    workflow_candidates: tuple[Path, ...],
    native_i2v_template: Path | None,
    native_flf_template: Path | None,
    i2v_verification: str,
    flf_verification: str,
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
        f"- Workflow profile candidates: {', '.join(f'`{path}`' for path in workflow_candidates) if workflow_candidates else 'unavailable'}",
        "",
        "## Preflight status",
        f"- Status: {status}",
        "",
    ]
    if blockers:
        lines.extend(["## BLOCKED", *(f"- BLOCKED: {blocker}" for blocker in blockers), ""])
    else:
        lines.extend(["## READY", "- READY: native schemas, registered templates, and project model roles were verified.", ""])
    lines.extend(
        [
        "## Template gate",
        f"- Native Wan I2V template: `{native_i2v_template}`" if native_i2v_template else "- Native Wan I2V template: unavailable",
        f"- Native Wan FLF template: `{native_flf_template}`" if native_flf_template else "- Native Wan FLF template: unavailable",
        f"- Native Wan I2V verification: {i2v_verification}",
        f"- Native Wan FLF verification: {flf_verification}",
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
