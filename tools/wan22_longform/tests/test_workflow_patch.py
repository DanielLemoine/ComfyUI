from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.config import (  # noqa: E402
    IdentityLora,
    LoraSlot,
    ModelFiles,
    Models,
    Permissiveness,
    ResolvedRenderConfig,
)
from wan22_longform.workflow import (  # noqa: E402
    WorkflowError,
    build_api_graph,
    find_unique_node,
    validate_graph_against_object_info,
    validate_two_stage_graph,
)


def load_fixture(name: str) -> dict[str, dict[str, object]]:
    return json.loads(
        (PROJECT_DIR / "tests" / "fixtures" / name).read_text(encoding="utf-8")
    )


def base_render_config(**changes: object) -> ResolvedRenderConfig:
    values: dict[str, object] = {
        "models": Models(
            high="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            low="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ),
        "permissiveness": Permissiveness(mode="none"),
        "identity": IdentityLora(),
        "vbvr": LoraSlot(branch="high", enabled=False),
        "motion": LoraSlot(branch="high", enabled=False),
        "corrective": LoraSlot(branch="low", enabled=False),
    }
    values.update(changes)
    return ResolvedRenderConfig(**values)  # type: ignore[arg-type]


def available_files(*extra: str) -> ModelFiles:
    return ModelFiles.from_names(
        {
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "vbvr.safetensors",
            "motion.safetensors",
            "mystic.safetensors",
            "corrective.safetensors",
            "identity-high.safetensors",
            "identity-low.safetensors",
            *extra,
        }
    )


def load_native_schema() -> dict[str, object]:
    return json.loads(
        (
            PROJECT_DIR / "tests" / "fixtures" / "native_workflow_object_info.json"
        ).read_text(encoding="utf-8")
    )


def titles(graph: dict[str, dict[str, object]]) -> set[str]:
    return {
        str(node.get("_meta", {}).get("title"))
        for node in graph.values()
        if isinstance(node.get("_meta"), dict)
    }


class WorkflowPatchTests(unittest.TestCase):
    def test_disabled_loras_are_absent_from_executable_graph(self) -> None:
        base = load_fixture("native_segment_api.json")

        graph = build_api_graph(base, base_render_config())

        self.assertFalse(
            any(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values())
        )
        self.assertNotIn("LORA_VBVR_HIGH", titles(graph))
        self.assertNotIn("LORA_IDENTITY_LOW", titles(graph))
        self.assertEqual(base, load_fixture("native_segment_api.json"))

    def test_lookup_requires_exactly_one_title_and_type(self) -> None:
        graph = load_fixture("native_segment_api.json")
        duplicate = deepcopy(graph)
        duplicate["99"] = deepcopy(graph["1"])

        with self.assertRaisesRegex(WorkflowError, "multiple matches"):
            find_unique_node(duplicate, "MODEL_HIGH", "UNETLoader")
        with self.assertRaisesRegex(WorkflowError, "zero matches"):
            find_unique_node(graph, "MISSING_MODEL", "UNETLoader")
        with self.assertRaisesRegex(WorkflowError, "zero matches"):
            find_unique_node(graph, "MODEL_HIGH", "VAELoader")

    def test_low_sampler_has_no_new_noise_and_uses_high_latent(self) -> None:
        graph = load_fixture("native_segment_api.json")

        validate_two_stage_graph(graph)

        low = find_unique_node(graph, "SAMPLER_LOW", "KSamplerAdvanced").node
        high = find_unique_node(graph, "SAMPLER_HIGH", "KSamplerAdvanced")
        self.assertEqual(low["inputs"]["add_noise"], "disable")
        self.assertEqual(low["inputs"]["latent_image"], [high.node_id, 0])

    def test_invalid_low_sampler_noise_or_latent_route_is_rejected(self) -> None:
        for input_name, bad_value, expected in (
            ("add_noise", "enable", "low sampler add_noise"),
            ("latent_image", ["10", 2], "high sampler latent"),
        ):
            with self.subTest(input_name=input_name):
                graph = load_fixture("native_segment_api.json")
                low = find_unique_node(graph, "SAMPLER_LOW", "KSamplerAdvanced").node
                low["inputs"][input_name] = bad_value
                with self.assertRaisesRegex(WorkflowError, expected):
                    validate_two_stage_graph(graph)

    def test_model_chain_rejects_non_model_output_slot(self) -> None:
        graph = load_fixture("native_segment_api.json")
        sampling = find_unique_node(
            graph, "MODEL_SAMPLING_HIGH", "ModelSamplingSD3"
        ).node
        sampling["inputs"]["model"] = ["1", 1]

        with self.assertRaisesRegex(WorkflowError, "model output 0"):
            validate_two_stage_graph(graph)

    def test_enabled_loras_are_inserted_in_binding_chain_order(self) -> None:
        render = base_render_config(
            vbvr=LoraSlot("vbvr.safetensors", "high", 0.25),
            motion=LoraSlot("motion.safetensors", "high", 0.30),
            permissiveness=Permissiveness(
                mode="mystic",
                mystic=LoraSlot("mystic.safetensors", "low", 0.20),
            ),
            corrective=LoraSlot("corrective.safetensors", "low", 0.15),
            identity=IdentityLora(
                mode="split",
                high_file="identity-high.safetensors",
                low_file="identity-low.safetensors",
                high_strength=0.9,
                low_strength=0.8,
            ),
        )

        graph = build_api_graph(
            load_fixture("native_segment_api.json"), render, available_files()
        )

        self.assertEqual(
            self._model_chain_titles(graph, "MODEL_SAMPLING_HIGH"),
            ["MODEL_HIGH", "LORA_VBVR_HIGH", "LORA_MOTION_HIGH", "LORA_IDENTITY_HIGH"],
        )
        self.assertEqual(
            self._model_chain_titles(graph, "MODEL_SAMPLING_LOW"),
            [
                "MODEL_LOW",
                "LORA_PERMISSIVENESS_LOW",
                "LORA_CORRECTIVE_LOW",
                "LORA_IDENTITY_LOW",
            ],
        )
        permissiveness_nodes = [
            node
            for node in graph.values()
            if node.get("_meta", {}).get("title") == "LORA_PERMISSIVENESS_LOW"
        ]
        self.assertEqual(len(permissiveness_nodes), 1)

    def test_builder_patches_exact_high_and_low_model_names(self) -> None:
        render = base_render_config(
            models=Models("configured-high.safetensors", "configured-low.safetensors")
        )

        graph = build_api_graph(
            load_fixture("native_segment_api.json"), render, available_files()
        )

        self.assertEqual(
            find_unique_node(graph, "MODEL_HIGH", "UNETLoader").node["inputs"]["unet_name"],
            "configured-high.safetensors",
        )
        self.assertEqual(
            find_unique_node(graph, "MODEL_LOW", "UNETLoader").node["inputs"]["unet_name"],
            "configured-low.safetensors",
        )

    def test_enabled_lora_without_configured_filename_is_rejected(self) -> None:
        render = base_render_config(
            vbvr=LoraSlot(branch="high", strength=0.25, enabled=True)
        )

        with self.assertRaisesRegex(WorkflowError, "configured filename"):
            build_api_graph(
                load_fixture("native_segment_api.json"), render, available_files()
            )

    def test_enabled_lora_requires_an_available_inventory(self) -> None:
        base = load_fixture("native_segment_api.json")
        render = base_render_config(
            vbvr=LoraSlot("vbvr.safetensors", "high", 0.25)
        )

        with self.assertRaisesRegex(WorkflowError, "requires an available-file inventory"):
            build_api_graph(base, render)
        self.assertNotIn("LORA_VBVR_HIGH", titles(base))

    def test_enabled_lora_missing_from_supplied_available_inventory_is_rejected(self) -> None:
        base = load_fixture("native_segment_api.json")
        render = base_render_config(
            vbvr=LoraSlot("definitely-missing.safetensors", "high", 0.25)
        )

        with self.assertRaisesRegex(WorkflowError, "available-file inventory"):
            build_api_graph(base, render, available_files())
        self.assertNotIn("LORA_VBVR_HIGH", titles(base))

    def test_two_stage_graph_rejects_case_insensitive_duplicate_model_files(self) -> None:
        graph = load_fixture("native_segment_api.json")
        graph["2"]["inputs"]["unet_name"] = graph["1"]["inputs"][
            "unet_name"
        ].upper()

        with self.assertRaisesRegex(WorkflowError, "high and low model filenames must differ"):
            validate_two_stage_graph(graph)

    def test_clean_fixture_matches_installed_schema_and_official_topology(self) -> None:
        graph = load_fixture("native_segment_api.json")
        object_info = load_native_schema()

        validate_graph_against_object_info(graph, object_info)
        validate_two_stage_graph(graph)
        serialized = json.dumps(graph).casefold()
        self.assertNotIn("lightx2v", serialized)
        self.assertNotIn("loraloadermodelonly", serialized)

    def test_installed_schema_validation_rejects_incompatible_link_types(self) -> None:
        graph = load_fixture("native_segment_api.json")
        object_info = load_native_schema()
        low = find_unique_node(graph, "SAMPLER_LOW", "KSamplerAdvanced").node
        low["inputs"]["model"] = ["11", 0]

        with self.assertRaisesRegex(WorkflowError, "expects MODEL.*provides LATENT"):
            validate_graph_against_object_info(graph, object_info)

    def test_installed_schema_validation_rejects_negative_output_index(self) -> None:
        graph = load_fixture("native_segment_api.json")
        graph["4"]["inputs"]["model"] = ["2", -1]

        with self.assertRaisesRegex(WorkflowError, "unavailable output -1"):
            validate_graph_against_object_info(graph, load_native_schema())

    def test_persisted_i2v_graphs_are_clean_and_flf_is_documented_not_built(self) -> None:
        ui_path = PROJECT_DIR / "workflows" / "ui" / "wan22_segment_i2v_native.json"
        api_path = PROJECT_DIR / "workflows" / "api" / "wan22_segment_i2v_native_api.json"
        bridge_ui = PROJECT_DIR / "workflows" / "ui" / "wan22_bridge_flf2v_native.json"
        bridge_api = PROJECT_DIR / "workflows" / "api" / "wan22_bridge_flf2v_native_api.json"
        bridge_fixture = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"

        self.assertTrue(ui_path.is_file())
        self.assertEqual(json.loads(api_path.read_text(encoding="utf-8")), load_fixture("native_segment_api.json"))
        self.assertNotIn("lightx2v", ui_path.read_text(encoding="utf-8").casefold())
        self.assertFalse(bridge_ui.exists())
        self.assertFalse(bridge_api.exists())
        self.assertFalse(bridge_fixture.exists())
        for note_path in (
            bridge_ui.with_suffix(".NOT_BUILT.md"),
            bridge_api.with_suffix(".NOT_BUILT.md"),
            bridge_fixture.with_suffix(".NOT_BUILT.md"),
        ):
            note = note_path.read_text(encoding="utf-8")
            self.assertIn("WanFirstLastFrameToVideo", note)
            self.assertIn("BRIDGE_FIRST_IMAGE", note)
            self.assertIn("BRIDGE_LAST_IMAGE", note)
            self.assertIn("BRIDGE_START", note)
            self.assertIn("BRIDGE_END", note)

    def test_clean_ui_subgraph_external_links_match_compacted_input_slots(self) -> None:
        ui = json.loads(
            (
                PROJECT_DIR / "workflows" / "ui" / "wan22_segment_i2v_native.json"
            ).read_text(encoding="utf-8")
        )
        subgraph = ui["definitions"]["subgraphs"][0]
        links = {link["id"]: link for link in subgraph["links"]}

        for slot, graph_input in enumerate(subgraph["inputs"]):
            with self.subTest(input=graph_input["name"]):
                for link_id in graph_input["linkIds"]:
                    self.assertEqual(links[link_id]["origin_id"], -10)
                    self.assertEqual(links[link_id]["origin_slot"], slot)

    @staticmethod
    def _model_chain_titles(
        graph: dict[str, dict[str, object]], sampling_title: str
    ) -> list[str]:
        node = find_unique_node(graph, sampling_title, "ModelSamplingSD3").node
        chain: list[str] = []
        source_id = str(node["inputs"]["model"][0])
        while True:
            source = graph[source_id]
            chain.append(str(source["_meta"]["title"]))
            if source["class_type"] == "UNETLoader":
                break
            source_id = str(source["inputs"]["model"][0])
        chain.reverse()
        return chain


if __name__ == "__main__":
    unittest.main()
