from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PERMISSIVENESS_MODES = frozenset({"none", "mystic", "wan_general"})


class ConfigError(ValueError):
    """Raised when a local Wan2.2 project configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class Models:
    high: str
    low: str


@dataclass(frozen=True)
class LoraSlot:
    file: str | None = None
    branch: str = "high"
    strength: float = 1.0
    enabled: bool | None = None

    @property
    def is_enabled(self) -> bool:
        return (self.file is not None if self.enabled is None else self.enabled) and self.strength > 0

    def matches_denylist(self, patterns: tuple[str, ...]) -> bool:
        return self.file is not None and any(
            pattern.casefold() in self.file.casefold() for pattern in patterns
        )


@dataclass(frozen=True)
class IdentityLora:
    mode: str = "none"
    file: str | None = None
    high_file: str | None = None
    low_file: str | None = None
    high_strength: float = 0.0
    low_strength: float = 0.0
    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.mode == "single_both":
            if self.enabled is False:
                return
            if not self.file:
                raise ConfigError("single_both identity requires a file")
            object.__setattr__(self, "high_file", self.file)
            object.__setattr__(self, "low_file", self.file)
        elif self.mode == "split":
            if self.enabled is False:
                return
            if not self.high_file or not self.low_file:
                raise ConfigError("split identity requires high_file and low_file")
        elif self.mode != "none":
            raise ConfigError("unknown identity mode")

    def slots(self) -> tuple[LoraSlot, ...]:
        if self.mode == "none" or self.enabled is False:
            return ()
        return (
            LoraSlot(self.high_file, "high", self.high_strength),
            LoraSlot(self.low_file, "low", self.low_strength),
        )


@dataclass(frozen=True)
class Permissiveness:
    mode: str
    mystic: LoraSlot = field(default_factory=lambda: LoraSlot(branch="low", enabled=False))
    wan_general: LoraSlot = field(
        default_factory=lambda: LoraSlot(branch="low", enabled=False)
    )

    @property
    def active(self) -> LoraSlot:
        if self.mode == "mystic":
            return self.mystic
        if self.mode == "wan_general":
            return self.wan_general
        return LoraSlot(branch="low", strength=0.0, enabled=False)


@dataclass(frozen=True)
class ResolvedRenderConfig:
    models: Models
    permissiveness: Permissiveness
    identity: IdentityLora = field(default_factory=IdentityLora)
    vbvr: LoraSlot = field(default_factory=lambda: LoraSlot(branch="high", enabled=False))
    motion: LoraSlot = field(default_factory=lambda: LoraSlot(branch="high", enabled=False))
    corrective: LoraSlot = field(default_factory=lambda: LoraSlot(branch="low", enabled=False))
    dangerous_override: bool = False
    denylist: tuple[str, ...] = ()
    preset: str = ""

    @property
    def permissiveness_mode(self) -> str:
        return self.permissiveness.mode

    def enabled_loras(self) -> tuple[LoraSlot, ...]:
        slots = (self.vbvr, self.motion, self.corrective, self.permissiveness.active)
        return tuple(slot for slot in (*slots, *self.identity.slots()) if slot.is_enabled)


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    source: Mapping[str, Any]


@dataclass(frozen=True)
class Preset:
    name: str
    vbvr_high: float
    permissiveness_mode: str
    permissiveness_low: float
    motion_high: float
    identity_high: float
    identity_low: float


@dataclass(frozen=True)
class PresetCatalog:
    presets: Mapping[str, Preset]
    denylist: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFiles:
    names: frozenset[str]

    @classmethod
    def from_names(cls, names: set[str] | frozenset[str]) -> ModelFiles:
        return cls(frozenset(names))

    @classmethod
    def discover(cls, *roots: Path) -> ModelFiles:
        return cls.from_names(
            {
                path.name
                for root in roots
                if root.is_dir()
                for path in root.rglob("*")
                if path.is_file()
            }
        )

    def contains(self, name: str | None) -> bool:
        return name is not None and name in self.names


def load_project(path: Path) -> ProjectConfig:
    source = _read_mapping(path)
    _required_mapping(source, "models")
    return ProjectConfig(path=path.resolve(), source=_freeze(source))


def load_presets(path: Path) -> PresetCatalog:
    source = _read_mapping(path)
    raw_presets = _required_mapping(source, "presets")
    presets: dict[str, Preset] = {}
    required = (
        "vbvr_high",
        "permissiveness_mode",
        "permissiveness_low",
        "motion_high",
        "identity_high",
        "identity_low",
    )
    for name, raw_preset in raw_presets.items():
        if not isinstance(name, str) or not isinstance(raw_preset, dict):
            raise ConfigError("preset catalog contains an invalid preset")
        if any(key not in raw_preset for key in required):
            raise ConfigError(f"preset {name} is incomplete")
        presets[name] = Preset(
            name=name,
            vbvr_high=_number(raw_preset["vbvr_high"], name),
            permissiveness_mode=_string(raw_preset["permissiveness_mode"], name),
            permissiveness_low=_number(raw_preset["permissiveness_low"], name),
            motion_high=_number(raw_preset["motion_high"], name),
            identity_high=_number(raw_preset["identity_high"], name),
            identity_low=_number(raw_preset["identity_low"], name),
        )
    denylist_path = path.with_name("denylist.yaml")
    return PresetCatalog(MappingProxyType(presets), load_denylist(denylist_path))


def load_denylist(path: Path) -> tuple[str, ...]:
    source = _read_mapping(path)
    patterns = source.get("body_emphasis")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ConfigError("denylist body_emphasis must be a list of strings")
    return tuple(patterns)


def resolve_preset(project: ProjectConfig, presets: PresetCatalog) -> ResolvedRenderConfig:
    source = project.source
    preset_name = _string(source.get("preset"), "project preset")
    try:
        preset = presets.presets[preset_name]
    except KeyError as error:
        raise ConfigError(f"unknown preset: {preset_name}") from error
    if preset.permissiveness_mode not in PERMISSIVENESS_MODES:
        raise ConfigError("unknown permissiveness mode")

    models = _required_mapping(source, "models")
    loras = _mapping(source.get("loras"), "loras")
    permissiveness_loras = _mapping(loras.get("permissiveness"), "loras.permissiveness")
    identity_source = _mapping(loras.get("identity"), "loras.identity")
    identity = _identity(identity_source, preset)
    return ResolvedRenderConfig(
        models=Models(
            high=_string(models.get("high"), "models.high"),
            low=_string(models.get("low"), "models.low"),
        ),
        permissiveness=Permissiveness(
            mode=preset.permissiveness_mode,
            mystic=_slot(
                permissiveness_loras.get("mystic"), "low", preset.permissiveness_low
            ),
            wan_general=_slot(
                permissiveness_loras.get("wan_general"),
                "low",
                preset.permissiveness_low,
            ),
        ),
        identity=identity,
        vbvr=_slot(loras.get("vbvr"), "high", preset.vbvr_high),
        motion=_slot(loras.get("motion"), "high", preset.motion_high),
        corrective=_slot(loras.get("corrective"), "low", 1.0),
        dangerous_override=_bool(source.get("dangerous_override", False), "dangerous_override"),
        denylist=presets.denylist,
        preset=preset.name,
    )


def validate_lora_policy(config: ResolvedRenderConfig, files: ModelFiles) -> None:
    if config.models.high == config.models.low:
        raise ConfigError("high and low model files must differ")
    if config.vbvr.is_enabled and config.vbvr.branch != "high":
        raise ConfigError("VBVR must be high-noise only")
    if config.motion.is_enabled and config.motion.branch != "high":
        raise ConfigError("motion must be high-noise only")
    if config.corrective.is_enabled and config.corrective.branch != "low":
        raise ConfigError("corrective must be low-noise only")
    permissiveness = config.permissiveness
    if permissiveness.mode not in PERMISSIVENESS_MODES:
        raise ConfigError("unknown permissiveness mode")
    if permissiveness.mystic.is_enabled and permissiveness.wan_general.is_enabled:
        raise ConfigError("mystic and wan_general are mutually exclusive")
    if permissiveness.mode == "mystic" and permissiveness.wan_general.is_enabled:
        raise ConfigError("mystic and wan_general are mutually exclusive")
    if permissiveness.mode == "wan_general" and permissiveness.mystic.is_enabled:
        raise ConfigError("mystic and wan_general are mutually exclusive")
    if permissiveness.mode == "none" and (
        permissiveness.mystic.is_enabled or permissiveness.wan_general.is_enabled
    ):
        raise ConfigError("permissiveness mode none cannot enable an adapter")
    for slot in config.enabled_loras():
        if slot.matches_denylist(config.denylist) and not config.dangerous_override:
            raise ConfigError(f"body-emphasis LoRA rejected: {slot.file}")
        if not files.contains(slot.file):
            raise ConfigError(f"configured LoRA is missing: {slot.file or '<unnamed>'}")


def _identity(source: Mapping[str, Any], preset: Preset) -> IdentityLora:
    if not source:
        return IdentityLora()
    mode = _string(source.get("mode"), "loras.identity.mode")
    enabled = _bool(source.get("enabled", True), "loras.identity.enabled")
    return IdentityLora(
        mode=mode,
        file=_optional_string(source.get("file"), "loras.identity.file"),
        high_file=_optional_string(source.get("high_file"), "loras.identity.high_file"),
        low_file=_optional_string(source.get("low_file"), "loras.identity.low_file"),
        high_strength=preset.identity_high,
        low_strength=preset.identity_low,
        enabled=enabled,
    )


def _slot(value: Any, branch: str, strength: float) -> LoraSlot:
    source = _mapping(value, "LoRA")
    if not source:
        return LoraSlot(branch=branch, strength=strength, enabled=False)
    return LoraSlot(
        file=_optional_string(source.get("file"), "LoRA file"),
        branch=branch,
        strength=strength,
        enabled=_bool(source.get("enabled", True), "LoRA enabled"),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"invalid YAML configuration: {path}") from error
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _required_mapping(source: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in source:
        raise ConfigError(f"missing {name}")
    return _mapping(source[name], name)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value
