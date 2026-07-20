from __future__ import annotations

import ipaddress
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .atomic import copy_file, write_bytes
from .workflow import ApiGraph


class ComfyClientError(RuntimeError):
    """Raised when the local ComfyUI API violates the render contract."""


@dataclass(frozen=True)
class HistoryOutput:
    node_id: str
    filename: str
    subfolder: str = ""
    type: str = "output"
    local_path: Path | None = None


@dataclass(frozen=True)
class HistoryResult:
    prompt_id: str
    outputs: tuple[HistoryOutput, ...]
    raw: Mapping[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes: ...


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
        raise ComfyClientError(
            f"Local ComfyUI redirects are not allowed: {request.full_url} -> {new_url}"
        )


class _LocalTransport:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _RejectRedirects())

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.read()
        except ComfyClientError:
            raise
        except OSError as error:
            raise ComfyClientError(f"Local ComfyUI request failed: {url}: {error}") from error


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: Transport | None = None,
        poll_interval: float = 0.5,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = _loopback_origin(base_url)
        if poll_interval < 0 or timeout <= 0:
            raise ComfyClientError("poll_interval must be non-negative and timeout positive")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.transport = transport or _LocalTransport(timeout)

    def submit(self, graph: ApiGraph) -> str:
        response = self._request_json("POST", "/prompt", {"prompt": graph})
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyClientError("Local ComfyUI /prompt response lacks prompt_id")
        return prompt_id

    def wait(self, prompt_id: str) -> HistoryResult:
        if re.fullmatch(r"[A-Za-z0-9._-]+", prompt_id) is None:
            raise ComfyClientError("prompt_id must be a non-empty path component")
        deadline = time.monotonic() + self.timeout
        while True:
            payload = self._request_json("GET", f"/history/{prompt_id}")
            entry = payload.get(prompt_id)
            if isinstance(entry, Mapping):
                status = entry.get("status")
                if _history_failed(status):
                    raise ComfyClientError(f"Local ComfyUI prompt failed: {prompt_id}")
                if _history_complete(status) or isinstance(entry.get("outputs"), Mapping):
                    return HistoryResult(
                        prompt_id=prompt_id,
                        outputs=_history_outputs(entry.get("outputs")),
                        raw=dict(entry),
                    )
            if time.monotonic() >= deadline:
                raise ComfyClientError(f"Timed out waiting for local prompt: {prompt_id}")
            time.sleep(self.poll_interval)

    def upload_image(self, path: Path) -> str:
        if not path.is_file():
            raise ComfyClientError(f"Upload image does not exist: {path}")
        boundary = f"wan22-{secrets.token_hex(12)}"
        body = _multipart_image(path, boundary)
        response = self._request_json(
            "POST",
            "/upload/image",
            raw_body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        name = response.get("name")
        _validate_filename(name, "upload image name")
        return name

    def fetch_output(self, output: HistoryOutput, destination: Path) -> Path:
        _validate_history_output(output.filename, output.subfolder, output.type)
        if output.local_path is not None:
            return copy_file(output.local_path, destination)
        query = urlencode(
            {
                "filename": output.filename,
                "subfolder": output.subfolder,
                "type": output.type,
            }
        )
        content = self.transport.request(
            "GET", f"{self.base_url}/view?{query}", None, {}
        )
        return write_bytes(destination, content)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = dict(headers or {})
        body = raw_body
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        response = self.transport.request(
            method, f"{self.base_url}{path}", body, request_headers
        )
        try:
            decoded = json.loads(response)
        except (TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ComfyClientError(f"Local ComfyUI returned invalid JSON for {path}") from error
        if not isinstance(decoded, dict):
            raise ComfyClientError(f"Local ComfyUI returned a non-object for {path}")
        return decoded


def _loopback_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ComfyClientError(f"ComfyUI URL must be an unambiguous loopback origin: {value}")
    hostname = parsed.hostname.casefold()
    if hostname != "localhost":
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ComfyClientError(f"ComfyUI URL must use a loopback address: {value}")
    try:
        parsed.port
    except ValueError as error:
        raise ComfyClientError(f"ComfyUI URL has an invalid loopback origin: {value}") from error
    return value.rstrip("/")


def _history_complete(status: object) -> bool:
    if not isinstance(status, Mapping):
        return False
    return status.get("completed") is True or status.get("status_str") == "success"


def _history_failed(status: object) -> bool:
    if not isinstance(status, Mapping):
        return False
    return status.get("status_str") in {"error", "failed"}


def _history_outputs(value: object) -> tuple[HistoryOutput, ...]:
    if not isinstance(value, Mapping):
        return ()
    outputs: list[HistoryOutput] = []
    for node_id, node_outputs in value.items():
        if not isinstance(node_outputs, Mapping):
            continue
        for candidates in node_outputs.values():
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                filename = candidate.get("filename")
                subfolder = candidate.get("subfolder", "")
                output_type = candidate.get("type", "output")
                _validate_history_output(filename, subfolder, output_type)
                outputs.append(
                    HistoryOutput(str(node_id), filename, subfolder, output_type)
                )
    return tuple(outputs)


def _validate_history_output(filename: object, subfolder: object, output_type: object) -> None:
    _validate_filename(filename, "history output filename")
    if not isinstance(subfolder, str):
        raise ComfyClientError("Local ComfyUI history output subfolder must be a string")
    if subfolder and (
        "\\" in subfolder
        or subfolder.startswith("/")
        or subfolder.endswith("/")
        or ":" in subfolder
        or any(component in {"", ".", ".."} for component in subfolder.split("/"))
    ):
        raise ComfyClientError("Local ComfyUI history output subfolder is unsafe")
    if output_type != "output":
        raise ComfyClientError("Local ComfyUI history output type is not permitted")


def _validate_filename(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise ComfyClientError(f"Local ComfyUI {label} is unsafe")


def _multipart_image(path: Path, boundary: str) -> bytes:
    filename = path.name.replace('"', "")
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + path.read_bytes() + suffix
