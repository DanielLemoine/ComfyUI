from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from wan22_longform.sequence_workflow import build_sequence_workflow  # noqa: E402
from custom_nodes.wan22_longform_sequence import Wan22ConditionalSaveVideo  # noqa: E402
import custom_nodes.wan22_longform_sequence as sequence_nodes  # noqa: E402


NATIVE_WORKFLOW = PROJECT_DIR / "workflows" / "ui" / "wan22_segment_i2v_native.json"
SEQUENCE_WORKFLOW = PROJECT_DIR / "workflows" / "ui" / "wan22_six_segment_sequence_native.json"


class FakeVideo:
    def __init__(self) -> None:
        self.saved = False

    def get_dimensions(self) -> tuple[int, int]:
        return (640, 640)

    def save_to(self, *_: object, **__: object) -> None:
        self.saved = True


class SequenceWorkflowTests(unittest.TestCase):
    def test_conditional_saver_leaves_disabled_clip_unsaved(self) -> None:
        video = FakeVideo()

        result = Wan22ConditionalSaveVideo().save_video(
            False, video, "wan22_longform/segment_01", "mp4", "h264"
        )

        self.assertEqual(result, (video,))
        self.assertFalse(video.saved)

    def test_tail_frame_node_returns_only_requested_final_frames(self) -> None:
        self.assertTrue(hasattr(sequence_nodes, "Wan22TailFramesFromBatch"))
        tail_node = getattr(sequence_nodes, "Wan22TailFramesFromBatch", None)
        if tail_node is not None:
            result = tail_node().select_tail(("one", "two", "three", "four"), 2)
            self.assertEqual(result, (("three", "four"), 2))

    def test_drop_leading_frames_node_removes_conditioned_overlap(self) -> None:
        self.assertTrue(hasattr(sequence_nodes, "Wan22DropLeadingFrames"))
        drop_node = getattr(sequence_nodes, "Wan22DropLeadingFrames", None)
        if drop_node is not None:
            result = drop_node().drop_frames(("one", "two", "three", "four"), 2)
            self.assertEqual(result, (("three", "four"),))

    def test_sequence_has_six_lazy_segments_and_one_final_video(self) -> None:
        native = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))
        workflow = build_sequence_workflow(native)
        nodes = workflow["nodes"]
        by_title = {node["title"]: node for node in nodes}

        self.assertEqual(
            [node["title"] for node in nodes if node["title"].startswith("SEGMENT_")],
            [f"SEGMENT_{index:02}" for index in range(1, 7)],
        )
        self.assertEqual(
            [node["title"] for node in nodes if node["title"].startswith("ENABLE_SEGMENT_")],
            [f"ENABLE_SEGMENT_{index:02}" for index in range(1, 7)],
        )
        self.assertEqual(
            [node["title"] for node in nodes if node["title"].startswith("USE_PREVIOUS_TAIL_")],
            [f"USE_PREVIOUS_TAIL_{index:02}" for index in range(2, 7)],
        )
        self.assertEqual(
            [node["title"] for node in nodes if node["title"].startswith("USE_EXISTING_SEGMENT_")],
            [f"USE_EXISTING_SEGMENT_{index:02}" for index in range(1, 7)],
        )
        self.assertEqual(
            len([node for node in nodes if node["type"] == "Wan22ConditionalSaveVideo"]),
            6,
        )
        self.assertEqual(
            len([node for node in nodes if node["title"].startswith("RESUME_VIDEO_")]),
            5,
        )
        self.assertEqual(
            len([node for node in nodes if node["type"] == "VHS_LoadVideo"]),
            6,
        )
        self.assertEqual(
            len([node for node in nodes if node["type"] == "ComfySwitchNode"]),
            34,
        )
        self.assertEqual(by_title["SAVE_FINAL_VIDEO"]["type"], "SaveVideo")

        self.assertTrue(
            all(
                isinstance(link, list) and len(link) == 6
                for link in workflow["links"]
            ),
            "LiteGraph root links must use [id, origin, origin_slot, target, target_slot, type] arrays",
        )
        links = {link[0]: link for link in workflow["links"]}
        for node in nodes:
            if node["type"] != "ComfySwitchNode":
                continue
            linked_inputs = {item["name"]: item.get("link") for item in node["inputs"]}
            self.assertIsNotNone(linked_inputs["on_false"], node["title"])
            self.assertIsNotNone(linked_inputs["on_true"], node["title"])
            for input_name in ("on_false", "on_true"):
                self.assertIn(linked_inputs[input_name], links)

        by_id = {node["id"]: node for node in nodes}
        for link_id, link in links.items():
            origin = by_id[link[1]]["outputs"][link[2]]
            target = by_id[link[3]]["inputs"][link[4]]
            self.assertIsInstance(origin["links"], list)
            self.assertIn(link_id, origin["links"])
            self.assertEqual(target["link"], link_id)

        stored = json.loads(SEQUENCE_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(stored, workflow)

    def test_sequence_nodes_have_clear_non_overlapping_layout(self) -> None:
        native = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))
        nodes = build_sequence_workflow(native)["nodes"]
        overlaps: list[tuple[str, str]] = []

        for index, left in enumerate(nodes):
            left_x, left_y = left["pos"]
            left_width, left_height = left["size"]
            for right in nodes[index + 1 :]:
                right_x, right_y = right["pos"]
                right_width, right_height = right["size"]
                horizontal_gap = max(left_x, right_x) - min(left_x + left_width, right_x + right_width)
                vertical_gap = max(left_y, right_y) - min(left_y + left_height, right_y + right_height)
                if horizontal_gap < 24 and vertical_gap < 24:
                    overlaps.append((left["title"], right["title"]))

        self.assertEqual(overlaps, [])

    def test_each_continuation_uses_the_requested_final_frames_only(self) -> None:
        native = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))
        workflow = build_sequence_workflow(native)
        nodes = workflow["nodes"]
        links = {link[0]: link for link in workflow["links"]}
        by_title = {node["title"]: node for node in nodes}
        tail_nodes = [
            node for node in nodes if node["type"] == "Wan22TailFramesFromBatch"
        ]

        self.assertEqual(len(tail_nodes), 5)
        for index in range(2, 7):
            tail = by_title[f"CONTINUATION_TAIL_{index:02} (last N frames)"]
            source = by_title[
                f"START_SOURCE_{index:02}: previous tail (true) / resume video (false)"
            ]
            self.assertEqual(tail["widgets_values"], [8])
            self.assertIsNotNone(tail["inputs"][0]["link"])
            self.assertEqual(
                links[source["inputs"][1]["link"]][1],
                tail["id"],
            )
            self.assertEqual(tail["outputs"][1]["type"], "INT")
            count_switch = by_title[
                f"START_FRAME_COUNT_{index:02}: previous tail (true) / resume frame (false)"
            ]
            append = by_title[f"APPEND_FRAMES_{index:02} (skip conditioned tail)"]
            self.assertEqual(
                links[append["inputs"][1]["link"]][1],
                count_switch["id"],
            )

    def test_each_segment_has_its_own_subgraph_definition_and_prompts(self) -> None:
        native = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))
        workflow = build_sequence_workflow(native)
        nodes = workflow["nodes"]
        by_title = {node["title"]: node for node in nodes}
        definitions = workflow["definitions"]["subgraphs"]

        self.assertEqual(len(definitions), 6)
        self.assertEqual(len({definition["id"] for definition in definitions}), 6)
        self.assertEqual(
            len([node for node in nodes if node["type"] == "PrimitiveStringMultiline"]),
            0,
        )

        for index in range(1, 7):
            segment = by_title[f"SEGMENT_{index:02}"]
            definition = next(
                item for item in definitions if item["id"] == segment["type"]
            )
            self.assertIn(f"Segment {index}", segment["widgets_values"][0])
            self.assertIn(
                "PROMPT_NEGATIVE", {item["name"] for item in definition["inputs"]}
            )
            self.assertEqual(
                len(
                    [
                        node
                        for node in definition["nodes"]
                        if node["title"] == "PROMPT_NEGATIVE"
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
