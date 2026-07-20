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
            29,
        )
        self.assertEqual(by_title["SAVE_FINAL_VIDEO"]["type"], "SaveVideo")

        links = {link["id"]: link for link in workflow["links"]}
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
            origin = by_id[link["origin_id"]]["outputs"][link["origin_slot"]]
            target = by_id[link["target_id"]]["inputs"][link["target_slot"]]
            self.assertIsInstance(origin["links"], list)
            self.assertIn(link_id, origin["links"])
            self.assertEqual(target["link"], link_id)

        stored = json.loads(SEQUENCE_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(stored, workflow)


if __name__ == "__main__":
    unittest.main()
