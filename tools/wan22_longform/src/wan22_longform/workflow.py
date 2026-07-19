from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

from .config import LoraSlot, ModelFiles, ResolvedRenderConfig


ApiNode: TypeAlias = dict[str, Any]
ApiGraph: TypeAlias = dict[str, ApiNode]


class WorkflowError(ValueError):
    """Raised when a workflow cannot satisfy the native Wan2.2 graph contract."""


@dataclass(frozen=True)
class NodeRef:
    node_id: str
    node: ApiNode


def find_unique_node(graph: ApiGraph, title: str, class_type: str) -> NodeRef:
    matches = [
        NodeRef(str(node_id), node)
        for node_id, node in graph.items()
        if node.get("class_type") == class_type
        and node.get("_meta", {}).get("title") == title
    ]
    if not matches:
        raise WorkflowError(
            f"expected exactly one {class_type} titled {title}; found zero matches"
        )
    if len(matches) > 1:
        raise WorkflowError(
            f"expected exactly one {class_type} titled {title}; "
            f"found multiple matches ({len(matches)})"
        )
    return matches[0]


def build_api_graph(
    base_graph: ApiGraph,
    render: ResolvedRenderConfig,
    available_files: ModelFiles | None = None,
) -> ApiGraph:
    """Build from a clean base, requiring inventory only for enabled LoRAs."""
    graph = deepcopy(base_graph)
    if any(node.get("class_type") == "LoraLoaderModelOnly" for node in graph.values()):
        raise WorkflowError("base graph must be clean and contain no optional LoRA nodes")

    validate_two_stage_graph(graph)
    if not render.models.high or not render.models.low:
        raise WorkflowError("configured high and low model filenames are required")
    if render.models.high.casefold() == render.models.low.casefold():
        raise WorkflowError("configured high and low model filenames must differ")

    find_unique_node(graph, "MODEL_HIGH", "UNETLoader").node["inputs"][
        "unet_name"
    ] = render.models.high
    find_unique_node(graph, "MODEL_LOW", "UNETLoader").node["inputs"][
        "unet_name"
    ] = render.models.low

    for sampling_title, lora_title, slot in _ordered_lora_slots(render):
        if slot.is_enabled:
            if available_files is None:
                raise WorkflowError(
                    f"enabled optional LoRA {lora_title} requires an available-file "
                    "inventory"
                )
            _insert_model_only_lora(
                graph, sampling_title, lora_title, slot, available_files
            )

    validate_two_stage_graph(graph)
    return graph


def validate_two_stage_graph(graph: ApiGraph) -> None:
    if "lightx2v" in json.dumps(graph, ensure_ascii=False).casefold():
        raise WorkflowError("clean quality graph must not contain bundled LightX2V LoRAs")

    model_high = find_unique_node(graph, "MODEL_HIGH", "UNETLoader")
    model_low = find_unique_node(graph, "MODEL_LOW", "UNETLoader")
    high_filename = _unet_filename(model_high)
    low_filename = _unet_filename(model_low)
    if high_filename.casefold() == low_filename.casefold():
        raise WorkflowError("high and low model filenames must differ")
    sampling_high = find_unique_node(
        graph, "MODEL_SAMPLING_HIGH", "ModelSamplingSD3"
    )
    sampling_low = find_unique_node(graph, "MODEL_SAMPLING_LOW", "ModelSamplingSD3")
    sampler_high = find_unique_node(graph, "SAMPLER_HIGH", "KSamplerAdvanced")
    sampler_low = find_unique_node(graph, "SAMPLER_LOW", "KSamplerAdvanced")
    i2v = find_unique_node(graph, "I2V_CONDITIONING", "WanImageToVideo")

    _require_inputs(sampling_high, {"shift": 5.0})
    _require_inputs(sampling_low, {"shift": 5.0})
    _require_inputs(
        sampler_high,
        {
            "add_noise": "enable",
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
        },
    )
    _require_inputs(
        sampler_low,
        {
            "add_noise": "disable",
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
        },
    )

    _require_link(sampler_high, "model", sampling_high, 0)
    _require_link(sampler_low, "model", sampling_low, 0)
    _require_link(sampler_high, "latent_image", i2v, 2)
    if sampler_low.node.get("inputs", {}).get("latent_image") != [
        sampler_high.node_id,
        0,
    ]:
        raise WorkflowError("low sampler must use the high sampler latent output")

    high_chain = _model_chain(graph, sampling_high, model_high)
    low_chain = _model_chain(graph, sampling_low, model_low)
    _require_chain_order(
        high_chain,
        (
            "MODEL_HIGH",
            "LORA_VBVR_HIGH",
            "LORA_MOTION_HIGH",
            "LORA_IDENTITY_HIGH",
        ),
        "high",
    )
    _require_chain_order(
        low_chain,
        (
            "MODEL_LOW",
            "LORA_PERMISSIVENESS_LOW",
            "LORA_CORRECTIVE_LOW",
            "LORA_IDENTITY_LOW",
        ),
        "low",
    )

    chained_lora_ids = {
        node.node_id for node in (*high_chain, *low_chain) if node.node["class_type"] == "LoraLoaderModelOnly"
    }
    all_loras = [
        NodeRef(str(node_id), node)
        for node_id, node in graph.items()
        if node.get("class_type") == "LoraLoaderModelOnly"
    ]
    if len(chained_lora_ids) != len(all_loras):
        raise WorkflowError("every optional LoRA node must be connected in its expert chain")
    permissiveness_count = sum(
        node.node.get("_meta", {}).get("title") == "LORA_PERMISSIVENESS_LOW"
        for node in all_loras
    )
    if permissiveness_count > 1:
        raise WorkflowError("low expert may contain exactly one permissiveness LoRA")
    for node in all_loras:
        filename = node.node.get("inputs", {}).get("lora_name")
        if not isinstance(filename, str) or not filename:
            raise WorkflowError(
                f"optional LoRA {node.node_id} requires a configured filename"
            )


def validate_graph_against_object_info(
    graph: ApiGraph, object_info: Mapping[str, object]
) -> None:
    for node_id, node in graph.items():
        class_type = node.get("class_type")
        schema = object_info.get(class_type) if isinstance(class_type, str) else None
        if not isinstance(schema, Mapping):
            raise WorkflowError(f"node {node_id} uses unavailable class_type {class_type}")
        input_schema = schema.get("input")
        if not isinstance(input_schema, Mapping):
            raise WorkflowError(f"node {node_id} class_type {class_type} has no input schema")
        required = input_schema.get("required", {})
        optional = input_schema.get("optional", {})
        if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
            raise WorkflowError(f"node {node_id} class_type {class_type} has malformed inputs")
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            raise WorkflowError(f"node {node_id} class_type {class_type} has no inputs")
        missing = sorted(set(required) - set(inputs))
        if missing:
            raise WorkflowError(
                f"node {node_id} class_type {class_type} misses required inputs: {missing}"
            )
        unknown = sorted(set(inputs) - set(required) - set(optional))
        if unknown:
            raise WorkflowError(
                f"node {node_id} class_type {class_type} has unknown inputs: {unknown}"
            )
        for input_name, value in inputs.items():
            if not _is_link(value):
                continue
            source_id = str(value[0])
            if source_id not in graph:
                raise WorkflowError(
                    f"node {node_id} input {input_name} links to missing node {value[0]}"
                )
            source_class = graph[source_id].get("class_type")
            source_schema = (
                object_info.get(source_class) if isinstance(source_class, str) else None
            )
            source_outputs = (
                source_schema.get("output")
                if isinstance(source_schema, Mapping)
                else None
            )
            output_index = value[1]
            if (
                not isinstance(source_outputs, list)
                or output_index < 0
                or output_index >= len(source_outputs)
            ):
                raise WorkflowError(
                    f"node {node_id} input {input_name} uses unavailable output "
                    f"{output_index} from node {source_id}"
                )
            input_spec = required.get(input_name, optional.get(input_name))
            expected_type = (
                input_spec[0]
                if isinstance(input_spec, list)
                and input_spec
                and isinstance(input_spec[0], str)
                else None
            )
            output_type = source_outputs[output_index]
            if expected_type is not None and output_type != expected_type:
                raise WorkflowError(
                    f"node {node_id} input {input_name} expects {expected_type}, but "
                    f"node {source_id} output {output_index} provides {output_type}"
                )


def _ordered_lora_slots(
    render: ResolvedRenderConfig,
) -> tuple[tuple[str, str, LoraSlot], ...]:
    identity_high = LoraSlot(
        render.identity.high_file,
        "high",
        render.identity.high_strength,
        render.identity.enabled,
    )
    identity_low = LoraSlot(
        render.identity.low_file,
        "low",
        render.identity.low_strength,
        render.identity.enabled,
    )
    if render.identity.mode == "none":
        identity_high = LoraSlot(branch="high", enabled=False)
        identity_low = LoraSlot(branch="low", enabled=False)
    return (
        ("MODEL_SAMPLING_HIGH", "LORA_VBVR_HIGH", render.vbvr),
        ("MODEL_SAMPLING_HIGH", "LORA_MOTION_HIGH", render.motion),
        ("MODEL_SAMPLING_HIGH", "LORA_IDENTITY_HIGH", identity_high),
        (
            "MODEL_SAMPLING_LOW",
            "LORA_PERMISSIVENESS_LOW",
            render.permissiveness.active,
        ),
        ("MODEL_SAMPLING_LOW", "LORA_CORRECTIVE_LOW", render.corrective),
        ("MODEL_SAMPLING_LOW", "LORA_IDENTITY_LOW", identity_low),
    )


def _insert_model_only_lora(
    graph: ApiGraph,
    sampling_title: str,
    lora_title: str,
    slot: LoraSlot,
    available_files: ModelFiles,
) -> None:
    if not slot.file:
        raise WorkflowError(f"enabled optional LoRA {lora_title} needs a configured filename")
    if not available_files.contains(slot.file):
        raise WorkflowError(
            f"enabled optional LoRA {lora_title} is absent from the available-file "
            f"inventory: {slot.file}"
        )
    sampling = find_unique_node(graph, sampling_title, "ModelSamplingSD3")
    upstream = sampling.node.get("inputs", {}).get("model")
    if not _is_link(upstream):
        raise WorkflowError(f"{sampling_title} model input must be a graph link")
    node_id = _next_node_id(graph)
    graph[node_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": deepcopy(upstream),
            "lora_name": slot.file,
            "strength_model": slot.strength,
        },
        "_meta": {"title": lora_title},
    }
    sampling.node["inputs"]["model"] = [node_id, 0]


def _next_node_id(graph: ApiGraph) -> str:
    numeric_ids = [int(node_id) for node_id in graph if str(node_id).isdigit()]
    return str(max(numeric_ids, default=0) + 1)


def _unet_filename(node: NodeRef) -> str:
    filename = node.node.get("inputs", {}).get("unet_name")
    if not isinstance(filename, str) or not filename:
        raise WorkflowError(f"{node.node_id} requires a configured UNET filename")
    return filename


def _model_chain(
    graph: ApiGraph, sampling: NodeRef, expected_loader: NodeRef
) -> tuple[NodeRef, ...]:
    source = sampling.node.get("inputs", {}).get("model")
    chain: list[NodeRef] = []
    visited: set[str] = set()
    while _is_link(source):
        if source[1] != 0:
            raise WorkflowError(
                f"{sampling.node_id} model chain must use model output 0"
            )
        node_id = str(source[0])
        if node_id in visited:
            raise WorkflowError(f"{sampling.node_id} model chain contains a cycle")
        visited.add(node_id)
        try:
            node = graph[node_id]
        except KeyError as error:
            raise WorkflowError(
                f"{sampling.node_id} model chain links to missing node {node_id}"
            ) from error
        node_ref = NodeRef(node_id, node)
        chain.append(node_ref)
        if node.get("class_type") == "UNETLoader":
            break
        if node.get("class_type") != "LoraLoaderModelOnly":
            raise WorkflowError(
                f"{sampling.node_id} model chain contains unsupported node {node_id}"
            )
        source = node.get("inputs", {}).get("model")
    else:
        raise WorkflowError(f"{sampling.node_id} model chain does not reach a loader")

    chain.reverse()
    if not chain or chain[0].node_id != expected_loader.node_id:
        raise WorkflowError(
            f"{sampling.node_id} model chain must begin at {expected_loader.node_id}"
        )
    return tuple(chain)


def _require_chain_order(
    chain: tuple[NodeRef, ...], allowed_titles: tuple[str, ...], branch: str
) -> None:
    observed = tuple(str(node.node.get("_meta", {}).get("title")) for node in chain)
    if len(observed) != len(set(observed)):
        raise WorkflowError(f"{branch} expert chain contains duplicate titled nodes")
    try:
        indexes = tuple(allowed_titles.index(title) for title in observed)
    except ValueError as error:
        raise WorkflowError(
            f"{branch} expert chain contains an unsupported title: {observed}"
        ) from error
    if indexes != tuple(sorted(indexes)) or observed[0] != allowed_titles[0]:
        raise WorkflowError(
            f"{branch} expert chain has invalid order: {' -> '.join(observed)}"
        )


def _require_inputs(node: NodeRef, expected: Mapping[str, object]) -> None:
    inputs = node.node.get("inputs", {})
    for name, value in expected.items():
        if inputs.get(name) != value:
            raw_label = str(node.node.get("_meta", {}).get("title"))
            if raw_label.startswith("SAMPLER_"):
                label = f"{raw_label.removeprefix('SAMPLER_').lower()} sampler"
            else:
                label = raw_label.replace("_", " ").lower()
            raise WorkflowError(f"{label} {name} must be {value!r}")


def _require_link(
    target: NodeRef, input_name: str, source: NodeRef, output_index: int
) -> None:
    if target.node.get("inputs", {}).get(input_name) != [source.node_id, output_index]:
        target_title = target.node.get("_meta", {}).get("title")
        source_title = source.node.get("_meta", {}).get("title")
        raise WorkflowError(
            f"{target_title} {input_name} must use {source_title} output {output_index}"
        )


def _is_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )
