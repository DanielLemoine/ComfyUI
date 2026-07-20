from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


Workflow = dict[str, Any]


def _input(name: str, value_type: str, *, widget: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "type": value_type, "link": None}
    if widget:
        value["widget"] = {"name": name}
    return value


def _output(name: str, value_type: str) -> dict[str, Any]:
    return {"name": name, "type": value_type, "links": None}


def _node(
    node_id: int,
    node_type: str,
    title: str,
    pos: tuple[int, int],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    widgets_values: list[Any] | None = None,
    size: tuple[int, int] = (280, 130),
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"cnr_id": "comfy-core", "ver": "0.3.45"},
        "widgets_values": widgets_values or [],
        "title": title,
    }


def _switch(node_id: int, title: str, pos: tuple[int, int], value_type: str) -> dict[str, Any]:
    return _node(
        node_id,
        "ComfySwitchNode",
        title,
        pos,
        [
            _input("on_false", value_type),
            _input("on_true", value_type),
            _input("switch", "BOOLEAN", widget=True),
        ],
        [_output("output", value_type)],
        [False],
        (280, 150),
    )


def _segment_instance(template: dict[str, Any], node_id: int, index: int, pos: tuple[int, int]) -> dict[str, Any]:
    node = deepcopy(template)
    node["id"] = node_id
    node["title"] = f"SEGMENT_{index:02}"
    node["pos"] = list(pos)
    node["size"] = [420, 500]
    node["order"] = 0
    node["inputs"][0]["link"] = None
    for output in node["outputs"]:
        output["links"] = None
    values = list(node["widgets_values"])
    values[0] = (
        "sgfw person continues the same scene with natural motion and stable identity. "
        f"Segment {index}: describe the next action here."
    )
    values[4] = 264244520398999 + index - 1
    node["widgets_values"] = values
    return node


def build_sequence_workflow(native_workflow: Workflow) -> Workflow:
    """Create the user-facing six-segment sequence workflow from the native I2V graph."""
    definitions = native_workflow.get("definitions", {})
    if not isinstance(definitions, dict) or len(definitions.get("subgraphs", [])) != 1:
        raise ValueError("native workflow must contain exactly one I2V subgraph definition")
    segment_definition = definitions["subgraphs"][0]
    segment_type = segment_definition["id"]
    templates = [node for node in native_workflow["nodes"] if node.get("type") == segment_type]
    if len(templates) != 1:
        raise ValueError("native workflow must contain exactly one outer I2V segment node")

    workflow = deepcopy(native_workflow)
    workflow["nodes"] = []
    workflow["links"] = []
    workflow["groups"] = []
    nodes: list[dict[str, Any]] = workflow["nodes"]
    links: list[dict[str, Any]] = workflow["links"]
    node_by_id: dict[int, dict[str, Any]] = {}
    next_link_id = 1

    def add(node: dict[str, Any]) -> dict[str, Any]:
        node["order"] = len(nodes)
        nodes.append(node)
        node_by_id[node["id"]] = node
        return node

    def connect(
        origin_id: int,
        origin_slot: int,
        target_id: int,
        target_slot: int,
        value_type: str,
    ) -> int:
        nonlocal next_link_id
        link_id = next_link_id
        next_link_id += 1
        origin = node_by_id[origin_id]["outputs"][origin_slot]
        origin_links = origin.get("links")
        if origin_links is None:
            origin["links"] = [link_id]
        else:
            origin_links.append(link_id)
        node_by_id[target_id]["inputs"][target_slot]["link"] = link_id
        links.append(
            [link_id, origin_id, origin_slot, target_id, target_slot, value_type]
        )
        return link_id

    seed = add(
        _node(
            1,
            "LoadImage",
            "SEED_IMAGE (Segment 1)",
            (40, 160),
            [],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["wan22_sequence_seed.png", "image"],
            (360, 300),
        )
    )
    fallback_video = add(
        _node(
            2,
            "CreateVideo",
            "DISABLED_SEGMENT_FALLBACK_VIDEO",
            (40, 520),
            [
                _input("images", "IMAGE"),
                _input("audio", "AUDIO"),
                _input("fps", "FLOAT", widget=True),
            ],
            [_output("VIDEO", "VIDEO")],
            [16.0],
        )
    )
    connect(seed["id"], 0, fallback_video["id"], 0, "IMAGE")

    active_frames_id = seed["id"]
    active_frames_slot = 0
    segment_template = templates[0]
    segment_group_x = 500
    segment_group_width = 900

    for index in range(1, 7):
        x = segment_group_x + (index - 1) * segment_group_width
        enable_id = 1000 + index
        segment_id = 1100 + index
        append_id = 1200 + index
        active_switch_id = 1300 + index
        save_switch_id = 1400 + index
        saver_id = 1500 + index
        use_existing_id = 2200 + index
        cached_video_id = 2300 + index
        cached_append_id = 2400 + index
        content_switch_id = 2500 + index
        generated_save_switch_id = 2600 + index

        enable = add(
            _node(
                enable_id,
                "PrimitiveBoolean",
                f"ENABLE_SEGMENT_{index:02}",
                (x, 0),
                [],
                [_output("BOOLEAN", "BOOLEAN")],
                [index == 1],
            )
        )
        use_existing = add(
            _node(
                use_existing_id,
                "PrimitiveBoolean",
                f"USE_EXISTING_SEGMENT_{index:02}",
                (x, 160),
                [],
                [_output("BOOLEAN", "BOOLEAN")],
                [False],
            )
        )
        cached_video = add(
            _node(
                cached_video_id,
                "VHS_LoadVideo",
                f"CACHED_SEGMENT_{index:02} (upload MP4 when reusing)",
                (x, 320),
                [],
                [
                    _output("IMAGE", "IMAGE"),
                    _output("frame_count", "INT"),
                    _output("audio", "AUDIO"),
                    _output("video_info", "VHS_VIDEOINFO"),
                ],
                ["", 16.0, 0, 0, 0, 0, 1],
                (360, 250),
            )
        )
        cached_append = add(
            _node(
                cached_append_id,
                "ImageFromBatch",
                f"CACHED_APPEND_FRAMES_{index:02} (skip repeated first frame)",
                (x, 620),
                [
                    _input("image", "IMAGE"),
                    _input("batch_index", "INT", widget=True),
                    _input("length", "INT", widget=True),
                ],
                [_output("IMAGE", "IMAGE")],
                [1, 4096],
            )
        )
        connect(cached_video["id"], 0, cached_append["id"], 0, "IMAGE")

        if index == 1:
            start_source_id, start_source_slot = seed["id"], 0
        else:
            resume_tail_id = 1600 + index
            resume_last_id = 1700 + index
            start_switch_id = 1800 + index
            use_previous_id = 2100 + index
            use_previous = add(
                _node(
                    use_previous_id,
                    "PrimitiveBoolean",
                    f"USE_PREVIOUS_TAIL_{index:02}",
                    (x + 400, 0),
                    [],
                    [_output("BOOLEAN", "BOOLEAN")],
                    [True],
                )
            )
            resume_tail = add(
                _node(
                    resume_tail_id,
                    "WanVideoTailFrames",
                    f"RESUME_VIDEO_{index:02} (paste MP4 path)",
                    (x + 400, 170),
                    [],
                    [_output("tail_frames", "IMAGE")],
                    [r"D:\\AI\\outputs\\video\\your_saved_segment.mp4", 2, 16.0],
                    (360, 150),
                )
            )
            resume_last = add(
                _node(
                    resume_last_id,
                    "ImageFromBatch",
                    f"RESUME_LAST_FRAME_{index:02}",
                    (x + 400, 350),
                    [
                        _input("image", "IMAGE"),
                        _input("batch_index", "INT", widget=True),
                        _input("length", "INT", widget=True),
                    ],
                    [_output("IMAGE", "IMAGE")],
                    [1, 1],
                )
            )
            connect(resume_tail["id"], 0, resume_last["id"], 0, "IMAGE")
            start_switch = add(
                _switch(
                    start_switch_id,
                    f"START_SOURCE_{index:02}: previous tail (true) / resume video (false)",
                    (x + 400, 520),
                    "IMAGE",
                )
            )
            connect(resume_last["id"], 0, start_switch["id"], 0, "IMAGE")
            connect(active_frames_id, active_frames_slot, start_switch["id"], 1, "IMAGE")
            connect(use_previous["id"], 0, start_switch["id"], 2, "BOOLEAN")
            start_source_id, start_source_slot = start_switch["id"], 0

        segment = add(_segment_instance(segment_template, segment_id, index, (x, 800)))
        connect(start_source_id, start_source_slot, segment["id"], 0, "IMAGE")

        append = add(
            _node(
                append_id,
                "ImageFromBatch",
                f"APPEND_FRAMES_{index:02} (skip repeated first frame)",
                (x + 460, 800),
                [
                    _input("image", "IMAGE"),
                    _input("batch_index", "INT", widget=True),
                    _input("length", "INT", widget=True),
                ],
                [_output("IMAGE", "IMAGE")],
                [1, 4096],
            )
        )
        connect(segment["id"], 1, append["id"], 0, "IMAGE")

        content_switch = add(
            _switch(
                content_switch_id,
                f"CONTENT_FOR_SEGMENT_{index:02}: generated (false) / cached (true)",
                (x + 460, 970),
                "IMAGE",
            )
        )
        connect(append["id"], 0, content_switch["id"], 0, "IMAGE")
        connect(cached_append["id"], 0, content_switch["id"], 1, "IMAGE")
        connect(use_existing["id"], 0, content_switch["id"], 2, "BOOLEAN")

        append_batch = add(
            _node(
                1900 + index,
                "ImageBatch",
                f"MERGE_ACTIVE_WITH_SEGMENT_{index:02}",
                (x + 460, 1150),
                [_input("image1", "IMAGE"), _input("image2", "IMAGE")],
                [_output("IMAGE", "IMAGE")],
            )
        )
        connect(active_frames_id, active_frames_slot, append_batch["id"], 0, "IMAGE")
        connect(content_switch["id"], 0, append_batch["id"], 1, "IMAGE")

        active_switch = add(
            _switch(
                active_switch_id,
                f"ACTIVE_FRAMES_AFTER_SEGMENT_{index:02}",
                (x + 460, 1330),
                "IMAGE",
            )
        )
        connect(active_frames_id, active_frames_slot, active_switch["id"], 0, "IMAGE")
        connect(append_batch["id"], 0, active_switch["id"], 1, "IMAGE")
        connect(enable["id"], 0, active_switch["id"], 2, "BOOLEAN")

        generated_save_switch = add(
            _switch(
                generated_save_switch_id,
                f"GENERATED_VIDEO_FOR_SEGMENT_{index:02}_SAVE",
                (x + 460, 1530),
                "VIDEO",
            )
        )
        connect(segment["id"], 0, generated_save_switch["id"], 0, "VIDEO")
        connect(fallback_video["id"], 0, generated_save_switch["id"], 1, "VIDEO")
        connect(use_existing["id"], 0, generated_save_switch["id"], 2, "BOOLEAN")
        save_switch = add(
            _switch(
                save_switch_id,
                f"VIDEO_FOR_SEGMENT_{index:02}_SAVE",
                (x + 460, 1730),
                "VIDEO",
            )
        )
        connect(fallback_video["id"], 0, save_switch["id"], 0, "VIDEO")
        connect(generated_save_switch["id"], 0, save_switch["id"], 1, "VIDEO")
        connect(enable["id"], 0, save_switch["id"], 2, "BOOLEAN")

        saver = add(
            _node(
                saver_id,
                "Wan22ConditionalSaveVideo",
                f"SAVE_SEGMENT_{index:02} (enabled only)",
                (x + 460, 1930),
                [
                    _input("enabled", "BOOLEAN"),
                    _input("video", "VIDEO"),
                    _input("filename_prefix", "STRING", widget=True),
                    _input("format", "COMBO", widget=True),
                    _input("codec", "COMBO", widget=True),
                ],
                [_output("video", "VIDEO")],
                [f"wan22_longform/segments/segment_{index:02}", "mp4", "h264"],
                (330, 190),
            )
        )
        connect(enable["id"], 0, saver["id"], 0, "BOOLEAN")
        connect(save_switch["id"], 0, saver["id"], 1, "VIDEO")

        workflow["groups"].append(
            {
                "id": index,
                "title": f"Segment {index:02}: generate, reuse, continue, and save",
                "bounding": [x - 35, -70, 800, 2250],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            }
        )
        active_frames_id, active_frames_slot = active_switch["id"], 0

    final_video = add(
        _node(
            2001,
            "CreateVideo",
            "ASSEMBLE_ENABLED_SEGMENTS",
            (segment_group_x + 6 * segment_group_width, 1080),
            [
                _input("images", "IMAGE"),
                _input("audio", "AUDIO"),
                _input("fps", "FLOAT", widget=True),
            ],
            [_output("VIDEO", "VIDEO")],
            [16.0],
        )
    )
    connect(active_frames_id, active_frames_slot, final_video["id"], 0, "IMAGE")
    final_save = add(
        _node(
            2002,
            "SaveVideo",
            "SAVE_FINAL_VIDEO",
            (segment_group_x + 6 * segment_group_width, 1280),
            [
                _input("video", "VIDEO"),
                _input("filename_prefix", "STRING", widget=True),
                _input("format", "COMBO", widget=True),
                _input("codec", "COMBO", widget=True),
            ],
            [],
            ["wan22_longform/final_sequence", "mp4", "h264"],
            (330, 180),
        )
    )
    connect(final_video["id"], 0, final_save["id"], 0, "VIDEO")

    workflow["groups"].insert(
        0,
        {
            "id": 100,
            "title": "Seed image and disabled-segment fallback",
            "bounding": [0, -70, 460, 750],
            "color": "#704c99",
            "font_size": 24,
            "flags": {},
        },
    )
    workflow["groups"].append(
        {
            "id": 101,
            "title": "Final merged video",
            "bounding": [segment_group_x + 6 * segment_group_width - 40, 1000, 440, 560],
            "color": "#5b7f3e",
            "font_size": 24,
            "flags": {},
        }
    )
    workflow["last_node_id"] = max(node["id"] for node in nodes)
    workflow["last_link_id"] = next_link_id - 1
    workflow["revision"] = 0
    workflow["extra"] = {"workflowRendererVersion": "LG", "ue_links": []}
    return workflow


def write_sequence_workflow(native_path: Path, destination: Path) -> Path:
    import json

    workflow = build_sequence_workflow(json.loads(native_path.read_text(encoding="utf-8")))
    destination.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    return destination
