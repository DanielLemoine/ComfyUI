from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


PERMISSIVENESS_MODES = frozenset({"none", "mystic", "wan_general"})
PROJECT_SCHEMA_VERSION = 1
BRIDGE_STRATEGIES = frozenset({"direct", "flf2v", "intentional_cut", "external_control"})
ACCEPTED_SELECTED_TAIL = "accepted_selected_tail"
ACCEPTED_SELECTED_HEAD = "accepted_selected_head"


class ConfigError(ValueError):
    """Raised when a local Wan2.2 project configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class Models:
    high: str
    low: str
    vae: str
    text_encoder: str


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
class BridgeContract:
    shot_id: str
    strategy: str
    purpose: str | None
    from_segment: str | None
    to_segment: str | None


@dataclass(frozen=True)
class Preset:
    name: str
    vbvr_high: float
    permissiveness_mode: str
    permissiveness_low: float
    motion_high: float
    corrective_low: float
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
        return name is not None and any(
            discovered.casefold() == name.casefold() for discovered in self.names
        )


def load_project(path: Path) -> ProjectConfig:
    source = _read_mapping(path)
    _required_mapping(source, "models")
    return ProjectConfig(path=path.resolve(), source=_freeze(source))


def validate_project_contract(project: ProjectConfig) -> None:
    """Check the versioned operator manifest shape without changing source state."""
    source = project.source
    if source.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {PROJECT_SCHEMA_VERSION}")
    _string(source.get("project_id"), "project_id")
    _string(source.get("title"), "title")
    mode = _string(source.get("mode"), "mode")
    if mode not in {"cinematic", "continuous"}:
        raise ConfigError("mode must be cinematic or continuous")
    _positive_number(source.get("target_seconds"), "target_seconds")
    output_root = _string(source.get("output_root"), "output_root")
    _output_paths(project, output_root, _required_mapping(source, "outputs"))
    _render_contract(_required_mapping(source, "render"))
    _model_contract(_required_mapping(source, "models"))
    _model_inventory_contract(source)
    _string(source.get("workflow_api"), "workflow_api")
    _string(source.get("bridge_workflow_api"), "bridge_workflow_api")
    _workflow_hashes(_required_mapping(source, "workflow_hashes"))
    _environment_snapshot(_required_mapping(source, "environment_snapshot"))
    _lora_contract(_required_mapping(source, "loras"))
    _policy_contract(_required_mapping(source, "policy"))
    _continuation_contract(_required_mapping(source, "continuation"))
    _qc_contract(source, _required_mapping(source, "qc"))
    _manifest_inputs(source)
    shot_segments = _shots_contract(source)
    bridge_shots = _bridges_contract(source, shot_segments)
    _assembly_order(source, shot_segments, bridge_shots)


def _output_paths(project: ProjectConfig, output_root: str, outputs: Mapping[str, Any]) -> None:
    root = Path(output_root)
    if not root.is_absolute():
        root = project.path.parent / root
    root = root.resolve()
    names: set[str] = set()
    for key in ("review_mp4", "edit_master_ffv1", "edit_master_prores"):
        declared = _string(outputs.get(key), f"outputs.{key}")
        target = Path(declared)
        if not target.is_absolute():
            target = project.path.parent / target
        target = target.resolve()
        if not target.is_relative_to(root):
            raise ConfigError(f"outputs.{key} must be inside output_root")
        if not target.name:
            raise ConfigError(f"outputs.{key} must name a file")
        if target.name in names:
            raise ConfigError("output role filenames must be distinct")
        names.add(target.name)


def _render_contract(render: Mapping[str, Any]) -> None:
    _string(render.get("workflow"), "render.workflow")
    for key in ("width", "height", "frames"):
        _positive_int(render.get(key), f"render.{key}")
    _positive_number(render.get("generation_fps"), "render.generation_fps")
    _string(render.get("review_mp4_codec"), "render.review_mp4_codec")
    _string(render.get("master_codec"), "render.master_codec")
    _non_negative_int(render.get("seed_base"), "render.seed_base")
    _positive_int(render.get("seed_increment"), "render.seed_increment")


def _model_contract(models: Mapping[str, Any]) -> None:
    for key in ("high", "low", "vae", "text_encoder"):
        value = _string(models.get(key), f"models.{key}")
        if Path(value).name != value:
            raise ConfigError(f"models.{key} must be an exact filename")
    if models["high"].casefold() == models["low"].casefold():
        raise ConfigError("models.high and models.low must differ")


def _model_inventory_contract(source: Mapping[str, Any]) -> None:
    model_files = source.get("model_files")
    model_roots = source.get("model_roots")
    if model_files is None and model_roots is None:
        raise ConfigError("project requires model_files or model_roots")
    if model_files is not None:
        if not isinstance(model_files, (list, tuple)) or not all(
            isinstance(name, str) and name for name in model_files
        ):
            raise ConfigError("model_files must be a list of exact filenames")
        known = {name.casefold() for name in model_files}
        for key in ("high", "low", "vae", "text_encoder"):
            name = _string(_required_mapping(source, "models").get(key), f"models.{key}")
            if name.casefold() not in known:
                raise ConfigError(f"model_files does not contain models.{key}")
    if model_roots is not None and (
        not isinstance(model_roots, (list, tuple))
        or not all(isinstance(root, (str, Path)) and str(root) for root in model_roots)
    ):
        raise ConfigError("model_roots must be a list of local paths")


def _workflow_hashes(hashes: Mapping[str, Any]) -> None:
    for key in ("segment_api", "bridge_api"):
        value = _string(hashes.get(key), f"workflow_hashes.{key}")
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            raise ConfigError(f"workflow_hashes.{key} must be a SHA-256 hex digest")


def _environment_snapshot(snapshot: Mapping[str, Any]) -> None:
    for key in ("captured_at", "platform", "python", "gpu"):
        _string(snapshot.get(key), f"environment_snapshot.{key}")


def _lora_contract(loras: Mapping[str, Any]) -> None:
    identity = _required_mapping(loras, "identity")
    mode = _string(identity.get("mode"), "loras.identity.mode")
    if mode not in {"none", "single_both", "split"}:
        raise ConfigError("loras.identity.mode is invalid")
    for key in ("vbvr", "motion", "corrective", "permissiveness"):
        _required_mapping(loras, key)


def _policy_contract(policy: Mapping[str, Any]) -> None:
    _bool(policy.get("automatic_continuation"), "policy.automatic_continuation")


def _continuation_contract(continuation: Mapping[str, Any]) -> None:
    if _string(continuation.get("strategy"), "continuation.strategy") != "selected_tail":
        raise ConfigError("continuation.strategy must be selected_tail")
    _positive_int(continuation.get("reset_limit"), "continuation.reset_limit")


def _qc_contract(source: Mapping[str, Any], qc: Mapping[str, Any]) -> None:
    candidate_count = _positive_int(qc.get("candidate_count"), "qc.candidate_count")
    legacy_candidate_count = source.get("candidate_count")
    if legacy_candidate_count is not None:
        legacy_candidate_count = _positive_int(legacy_candidate_count, "candidate_count")
        if legacy_candidate_count != candidate_count:
            raise ConfigError("candidate_count conflicts with qc.candidate_count")
    _non_negative_int(qc.get("retry_limit"), "qc.retry_limit")


def _manifest_inputs(source: Mapping[str, Any]) -> None:
    inputs = source.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ConfigError("inputs must be a mapping")
    opening = inputs.get("opening_frame")
    if opening is not None:
        if isinstance(opening, Mapping):
            opening = opening.get("path")
        _string(opening, "inputs.opening_frame")


def _shots_contract(source: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    shots = source.get("shots")
    if not isinstance(shots, (list, tuple)) or not shots:
        raise ConfigError("shots must be a non-empty list")
    seen_shots: set[str] = set()
    shot_segments: dict[str, frozenset[str]] = {}
    for shot in shots:
        if not isinstance(shot, Mapping):
            raise ConfigError("shot must be a mapping")
        shot_id = _string(shot.get("id"), "shot.id")
        if shot_id in seen_shots:
            raise ConfigError(f"duplicate shot id: {shot_id}")
        seen_shots.add(shot_id)
        _positive_number(shot.get("target_seconds"), f"shot {shot_id} target_seconds")
        anchor = shot.get("anchor_image")
        if anchor is not None:
            _string(anchor, f"shot {shot_id} anchor_image")
        elif source.get("inputs", {}).get("opening_frame") is None:
            raise ConfigError(f"shot {shot_id} requires anchor_image or inputs.opening_frame")
        segments = shot.get("segments")
        if not isinstance(segments, (list, tuple)) or not segments:
            raise ConfigError(f"shot {shot_id} requires a non-empty segments list")
        seen_segments: set[str] = set()
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise ConfigError(f"shot {shot_id} contains an invalid segment")
            segment_id = _string(segment.get("id"), f"shot {shot_id} segment.id")
            if segment_id in seen_segments:
                raise ConfigError(f"shot {shot_id} has duplicate segment id: {segment_id}")
            seen_segments.add(segment_id)
            _string(segment.get("action"), f"segment {segment_id} action")
            _positive_number(segment.get("expected_seconds"), f"segment {segment_id} expected_seconds")
            _non_negative_int(segment.get("seed_offset"), f"segment {segment_id} seed_offset")
            continuation = segment.get("continue_from")
            if continuation is not None:
                reference = _string(continuation, f"segment {segment_id} continue_from")
                if reference not in seen_segments:
                    raise ConfigError(
                        f"segment {segment_id} continue_from must reference an earlier same-shot segment"
                    )
                if "opening_image" in segment or "opening_frame" in segment:
                    raise ConfigError(
                        f"segment {segment_id} cannot combine continue_from with an opening override"
                    )
            overrides = [key for key in ("opening_image", "opening_frame") if key in segment]
            if len(overrides) > 1:
                raise ConfigError(f"segment {segment_id} has multiple opening overrides")
            for key in overrides:
                _string(segment.get(key), f"segment {segment_id} {key}")
        shot_segments[shot_id] = frozenset(seen_segments)
    return shot_segments


def _bridges_contract(
    source: Mapping[str, Any], shot_segments: Mapping[str, frozenset[str]]
) -> dict[str, BridgeContract]:
    bridges = source.get("bridges", ())
    if not isinstance(bridges, (list, tuple)):
        raise ConfigError("bridges must be a list")
    seen: set[str] = set()
    bridge_shots: dict[str, BridgeContract] = {}
    for bridge in bridges:
        if not isinstance(bridge, Mapping):
            raise ConfigError("bridge must be a mapping")
        bridge_id = _string(bridge.get("id"), "bridge.id")
        if bridge_id in seen:
            raise ConfigError(f"duplicate bridge id: {bridge_id}")
        seen.add(bridge_id)
        shot_id = _string(bridge.get("shot_id"), f"bridge {bridge_id} shot_id")
        if shot_id not in shot_segments:
            raise ConfigError(f"bridge {bridge_id} shot_id does not reference a configured shot: {shot_id}")
        if bridge_id in shot_segments[shot_id]:
            raise ConfigError(
                f"bridge {bridge_id} collides with configured segment in shot {shot_id}"
            )
        strategy = _string(bridge.get("strategy"), f"bridge {bridge_id} strategy")
        if strategy not in BRIDGE_STRATEGIES:
            raise ConfigError(f"bridge {bridge_id} strategy is invalid")
        purpose = bridge.get("purpose")
        if purpose is not None:
            purpose = _string(purpose, f"bridge {bridge_id} purpose")
        base = bridge.get("base_source_image")
        first = bridge.get("first_image")
        last = bridge.get("last_image")
        if strategy == "flf2v":
            if base is not None:
                _string(base, f"bridge {bridge_id} base_source_image")
                if bridge.get("purpose") != "technical_smoke":
                    raise ConfigError(
                        f"bridge {bridge_id} base_source_image is only allowed for purpose technical_smoke"
                    )
                if first is not None or last is not None:
                    raise ConfigError(f"bridge {bridge_id} base source cannot have explicit endpoints")
            elif first is None or last is None:
                raise ConfigError(f"bridge {bridge_id} requires both explicit endpoints")
            else:
                has_semantic_selector = any(
                    isinstance(value, str)
                    and value in {ACCEPTED_SELECTED_TAIL, ACCEPTED_SELECTED_HEAD}
                    for value in (first, last)
                )
                if has_semantic_selector:
                    if first != ACCEPTED_SELECTED_TAIL or last != ACCEPTED_SELECTED_HEAD:
                        raise ConfigError(
                            f"bridge {bridge_id} semantic endpoint selectors must be "
                            "accepted_selected_tail then accepted_selected_head"
                        )
                    if purpose == "technical_smoke":
                        raise ConfigError(
                            f"technical_smoke bridge {bridge_id} cannot use accepted endpoint selectors"
                        )
                else:
                    _string(first, f"bridge {bridge_id} first_image")
                    _string(last, f"bridge {bridge_id} last_image")
            if purpose == "technical_smoke":
                if bridge.get("from_segment") is not None or bridge.get("to_segment") is not None:
                    raise ConfigError(
                        f"technical_smoke bridge {bridge_id} cannot declare story segments"
                    )
                from_segment = None
                to_segment = None
            else:
                from_segment = _string(
                    bridge.get("from_segment"), f"bridge {bridge_id} from_segment"
                )
                to_segment = _string(
                    bridge.get("to_segment"), f"bridge {bridge_id} to_segment"
                )
                if from_segment not in shot_segments[shot_id]:
                    raise ConfigError(
                        f"bridge {bridge_id} from_segment is not configured in shot {shot_id}"
                    )
                if to_segment not in shot_segments[shot_id]:
                    raise ConfigError(
                        f"bridge {bridge_id} to_segment is not configured in shot {shot_id}"
                    )
                if from_segment == to_segment:
                    raise ConfigError(f"bridge {bridge_id} from_segment and to_segment must differ")
        else:
            if base is not None or first is not None or last is not None:
                raise ConfigError(f"bridge {bridge_id} endpoints require strategy flf2v")
            from_segment = _string(
                bridge.get("from_segment"), f"bridge {bridge_id} from_segment"
            )
            to_segment = _string(bridge.get("to_segment"), f"bridge {bridge_id} to_segment")
            if from_segment not in shot_segments[shot_id]:
                raise ConfigError(
                    f"bridge {bridge_id} from_segment is not configured in shot {shot_id}"
                )
            if to_segment not in shot_segments[shot_id]:
                raise ConfigError(
                    f"bridge {bridge_id} to_segment is not configured in shot {shot_id}"
                )
            if from_segment == to_segment:
                raise ConfigError(f"bridge {bridge_id} from_segment and to_segment must differ")
        bridge_shots[bridge_id] = BridgeContract(
            shot_id=shot_id,
            strategy=strategy,
            purpose=purpose,
            from_segment=from_segment,
            to_segment=to_segment,
        )
    return bridge_shots


def _assembly_order(
    source: Mapping[str, Any],
    shot_segments: Mapping[str, frozenset[str]],
    bridge_shots: Mapping[str, BridgeContract],
) -> None:
    order = source.get("assembly_order")
    if not isinstance(order, (list, tuple)) or not order:
        raise ConfigError("assembly_order must be a non-empty list")
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[str, str]] = []
    for item in order:
        if not isinstance(item, Mapping):
            raise ConfigError("assembly_order entry must be a mapping")
        shot_id = _string(item.get("shot_id"), "assembly_order shot_id")
        segment_id = _string(item.get("segment_id"), "assembly_order segment_id")
        key = (shot_id, segment_id)
        if key in seen:
            raise ConfigError(f"assembly_order has a duplicate item: {shot_id}/{segment_id}")
        seen.add(key)
        if segment_id in shot_segments.get(shot_id, frozenset()):
            entries.append(key)
            continue
        bridge = bridge_shots.get(segment_id)
        if bridge is not None and bridge.shot_id == shot_id:
            if bridge.purpose == "technical_smoke":
                raise ConfigError(
                    f"assembly_order cannot include technical_smoke bridge: {shot_id}/{segment_id}"
                )
            if bridge.strategy != "flf2v":
                raise ConfigError(
                    f"assembly_order cannot include non-rendered {bridge.strategy} transition policy: "
                    f"{shot_id}/{segment_id}"
                )
            entries.append(key)
            continue
        raise ConfigError(
            f"assembly_order item does not reference a configured segment or matching bridge: "
            f"{shot_id}/{segment_id}"
        )
    for index, (shot_id, segment_id) in enumerate(entries):
        bridge = bridge_shots.get(segment_id)
        if bridge is None or bridge.strategy != "flf2v" or bridge.purpose == "technical_smoke":
            continue
        if index == 0 or index == len(entries) - 1:
            raise ConfigError(
                f"story FLF bridge {segment_id} must be directly between declared source and destination segments"
            )
        expected_left = (shot_id, bridge.from_segment)
        expected_right = (shot_id, bridge.to_segment)
        if entries[index - 1] != expected_left or entries[index + 1] != expected_right:
            raise ConfigError(
                f"story FLF bridge {segment_id} must be directly between declared source and destination segments"
            )
    for bridge_id, bridge in bridge_shots.items():
        if bridge.strategy == "flf2v":
            continue
        expected_left = (bridge.shot_id, bridge.from_segment)
        expected_right = (bridge.shot_id, bridge.to_segment)
        try:
            index = entries.index(expected_left)
        except ValueError as error:
            raise ConfigError(
                f"transition policy {bridge_id} must be directly between declared source and destination segments"
            ) from error
        if index == len(entries) - 1 or entries[index + 1] != expected_right:
            raise ConfigError(
                f"transition policy {bridge_id} must be directly between declared source and destination segments"
            )


def load_presets(path: Path) -> PresetCatalog:
    source = _read_mapping(path)
    raw_presets = _required_mapping(source, "presets")
    presets: dict[str, Preset] = {}
    required = (
        "vbvr_high",
        "permissiveness_mode",
        "permissiveness_low",
        "motion_high",
        "corrective_low",
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
            corrective_low=_number(raw_preset["corrective_low"], name),
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
            vae=_string(models.get("vae"), "models.vae"),
            text_encoder=_string(models.get("text_encoder"), "models.text_encoder"),
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
        corrective=_slot(loras.get("corrective"), "low", preset.corrective_low),
        dangerous_override=_bool(source.get("dangerous_override", False), "dangerous_override"),
        denylist=presets.denylist,
        preset=preset.name,
    )


def validate_lora_policy(config: ResolvedRenderConfig, files: ModelFiles) -> None:
    if config.models.high.casefold() == config.models.low.casefold():
        raise ConfigError("high and low model files must differ")
    for role, filename in (
        ("high", config.models.high),
        ("low", config.models.low),
        ("vae", config.models.vae),
        ("text_encoder", config.models.text_encoder),
    ):
        if not files.contains(filename):
            raise ConfigError(f"configured {role} model is missing: {filename}")
    if config.vbvr.is_enabled and config.vbvr.branch != "high":
        raise ConfigError("VBVR must be high-noise only")
    if config.motion.is_enabled and config.motion.branch != "high":
        raise ConfigError("motion must be high-noise only")
    if config.corrective.is_enabled and config.corrective.branch != "low":
        raise ConfigError("corrective must be low-noise only")
    permissiveness = config.permissiveness
    if permissiveness.mode not in PERMISSIVENESS_MODES:
        raise ConfigError("unknown permissiveness mode")
    if permissiveness.mystic.is_enabled and permissiveness.mystic.branch != "low":
        raise ConfigError("mystic must be low-noise only")
    if (
        permissiveness.wan_general.is_enabled
        and permissiveness.wan_general.branch != "low"
    ):
        raise ConfigError("wan_general must be low-noise only")
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
        strength=_number(source.get("weight", strength), "LoRA weight"),
        enabled=_bool(source.get("enabled", True), "LoRA enabled"),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
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


def _positive_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ConfigError(f"{name} must be positive")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value
