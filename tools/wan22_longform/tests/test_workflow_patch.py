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
            vae="wan_2.1_vae.safetensors",
            text_encoder="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
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
            "wan_2.1_vae.safetensors",
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
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
            models=Models(
                "configured-high.safetensors",
                "configured-low.safetensors",
                "configured-vae.safetensors",
                "configured-text-encoder.safetensors",
            )
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

    def test_builder_patches_all_four_declared_native_model_roles(self) -> None:
        render = base_render_config(
            models=Models(
                "configured-high.safetensors",
                "configured-low.safetensors",
                "configured-vae.safetensors",
                "configured-text-encoder.safetensors",
            )
        )

        graph = build_api_graph(
            load_fixture("native_segment_api.json"),
            render,
            available_files(
                "configured-high.safetensors",
                "configured-low.safetensors",
                "configured-vae.safetensors",
                "configured-text-encoder.safetensors",
            ),
        )

        self.assertEqual(
            find_unique_node(graph, "VAE", "VAELoader").node["inputs"]["vae_name"],
            "configured-vae.safetensors",
        )
        self.assertEqual(
            find_unique_node(graph, "TEXT_ENCODER", "CLIPLoader").node["inputs"][
                "clip_name"
            ],
            "configured-text-encoder.safetensors",
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

    def test_persisted_i2v_and_flf_graphs_are_clean_and_executable(self) -> None:
        ui_path = PROJECT_DIR / "workflows" / "ui" / "wan22_segment_i2v_native.json"
        api_path = PROJECT_DIR / "workflows" / "api" / "wan22_segment_i2v_native_api.json"
        bridge_ui = PROJECT_DIR / "workflows" / "ui" / "wan22_bridge_flf2v_native.json"
        bridge_api = PROJECT_DIR / "workflows" / "api" / "wan22_bridge_flf2v_native_api.json"
        bridge_fixture = PROJECT_DIR / "tests" / "fixtures" / "native_bridge_api.json"

        self.assertTrue(ui_path.is_file())
        self.assertEqual(json.loads(api_path.read_text(encoding="utf-8")), load_fixture("native_segment_api.json"))
        self.assertNotIn("lightx2v", ui_path.read_text(encoding="utf-8").casefold())
        self.assertTrue(bridge_ui.is_file())
        self.assertEqual(
            json.loads(bridge_api.read_text(encoding="utf-8")),
            load_fixture("native_bridge_api.json"),
        )
        self.assertNotIn("lightx2v", bridge_ui.read_text(encoding="utf-8").casefold())
        bridge_classes = {
            node["class_type"] for node in load_fixture("native_bridge_api.json").values()
        }
        self.assertNotIn("Note", bridge_classes)
        self.assertNotIn("MarkdownNote", bridge_classes)
        for note_path in (
            bridge_ui.with_suffix(".NOT_BUILT.md"),
            bridge_api.with_suffix(".NOT_BUILT.md"),
            bridge_fixture.with_suffix(".NOT_BUILT.md"),
        ):
            self.assertFalse(note_path.exists())

    def test_quality_graphs_use_their_verified_normal_template_baselines(self) -> None:
        for fixture_name, high_sampler, low_sampler, shift, cfg in (
            ("native_segment_api.json", "SAMPLER_HIGH", "SAMPLER_LOW", 5.0, 3.5),
            (
                "native_bridge_api.json",
                "BRIDGE_SAMPLER_HIGH",
                "BRIDGE_SAMPLER_LOW",
                8.0,
                4.0,
            ),
        ):
            with self.subTest(fixture=fixture_name):
                graph = load_fixture(fixture_name)
                self.assertEqual(
                    find_unique_node(graph, "MODEL_SAMPLING_HIGH", "ModelSamplingSD3").node[
                        "inputs"
                    ]["shift"],
                    shift,
                )
                self.assertEqual(
                    find_unique_node(graph, "MODEL_SAMPLING_LOW", "ModelSamplingSD3").node[
                        "inputs"
                    ]["shift"],
                    shift,
                )
                self.assertEqual(
                    find_unique_node(graph, high_sampler, "KSamplerAdvanced").node["inputs"],
                    {
                        "model": ["3", 0],
                        "positive": ["10" if fixture_name == "native_segment_api.json" else "11", 0],
                        "negative": ["10" if fixture_name == "native_segment_api.json" else "11", 1],
                        "latent_image": ["10" if fixture_name == "native_segment_api.json" else "11", 2],
                        "add_noise": "enable",
                        "noise_seed": 0,
                        "steps": 20,
                        "cfg": cfg,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "start_at_step": 0,
                        "end_at_step": 10,
                        "return_with_leftover_noise": "enable",
                    },
                )
                low_inputs = find_unique_node(
                    graph, low_sampler, "KSamplerAdvanced"
                ).node["inputs"]
                self.assertEqual(low_inputs["add_noise"], "disable")
                self.assertEqual(low_inputs["steps"], 20)
                self.assertEqual(low_inputs["cfg"], cfg)
                self.assertEqual(low_inputs["start_at_step"], 10)
                self.assertEqual(low_inputs["end_at_step"], 20)
                self.assertEqual(low_inputs["return_with_leftover_noise"], "disable")

    def test_template_specific_quality_profiles_are_enforced(self) -> None:
        i2v = load_fixture("native_segment_api.json")
        find_unique_node(i2v, "MODEL_SAMPLING_HIGH", "ModelSamplingSD3").node[
            "inputs"
        ]["shift"] = 8.0
        with self.assertRaisesRegex(WorkflowError, "model sampling high shift must be 5.0"):
            validate_two_stage_graph(i2v)

        bridge = load_fixture("native_bridge_api.json")
        find_unique_node(bridge, "BRIDGE_SAMPLER_HIGH", "KSamplerAdvanced").node[
            "inputs"
        ]["cfg"] = 3.5
        with self.assertRaisesRegex(WorkflowError, "bridge sampler high cfg must be 4.0"):
            validate_two_stage_graph(bridge)

    def test_bridge_routes_both_canonical_endpoints_into_flf_conditioning(self) -> None:
        graph = load_fixture("native_bridge_api.json")
        conditioning = find_unique_node(
            graph, "FLF_CONDITIONING", "WanFirstLastFrameToVideo"
        ).node

        self.assertEqual(
            conditioning["inputs"]["start_image"],
            [find_unique_node(graph, "BRIDGE_FIRST_IMAGE", "LoadImage").node_id, 0],
        )
        self.assertEqual(
            conditioning["inputs"]["end_image"],
            [find_unique_node(graph, "BRIDGE_LAST_IMAGE", "LoadImage").node_id, 0],
        )
        patch_map = (PROJECT_DIR / "workflows" / "PATCH_MAP.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("BRIDGE_START", patch_map)
        self.assertIn("BRIDGE_END", patch_map)
        self.assertIn("52a53af170145cfd579e6e6f6334ce25e9b8cf10", patch_map)
        self.assertIn(
            "9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf",
            patch_map,
        )

    def test_clean_bridge_fixture_matches_installed_schema_and_native_topology(self) -> None:
        graph = load_fixture("native_bridge_api.json")

        validate_graph_against_object_info(graph, load_native_schema())
        validate_two_stage_graph(graph)
        self.assertNotIn("lightx2v", json.dumps(graph).casefold())
        self.assertNotIn("loraloadermodelonly", json.dumps(graph).casefold())

    def test_bridge_ui_subgraph_preserves_both_exposed_endpoint_slots(self) -> None:
        ui = json.loads(
            (
                PROJECT_DIR / "workflows" / "ui" / "wan22_bridge_flf2v_native.json"
            ).read_text(encoding="utf-8")
        )
        subgraph = ui["definitions"]["subgraphs"][0]
        links = {link["id"]: link for link in subgraph["links"]}

        for slot, input_name in enumerate(("BRIDGE_FIRST_IMAGE", "BRIDGE_LAST_IMAGE")):
            graph_input = next(
                entry for entry in subgraph["inputs"] if entry["name"] == input_name
            )
            self.assertEqual(len(graph_input["linkIds"]), 1)
            link = links[graph_input["linkIds"][0]]
            self.assertEqual(link["origin_id"], -10)
            self.assertEqual(link["origin_slot"], slot)
            self.assertEqual(link["target_id"], 11)
            self.assertEqual(link["target_slot"], 5 + slot)

        self.assertLessEqual(
            {node["type"] for node in subgraph["nodes"]},
            set(load_native_schema()),
        )

    def test_bridge_rejects_invalid_endpoint_routes(self) -> None:
        graph = load_fixture("native_bridge_api.json")
        conditioning = find_unique_node(
            graph, "FLF_CONDITIONING", "WanFirstLastFrameToVideo"
        ).node
        conditioning["inputs"]["end_image"] = conditioning["inputs"]["start_image"]

        with self.assertRaisesRegex(WorkflowError, "BRIDGE_LAST_IMAGE"):
            validate_two_stage_graph(graph)

    def test_bridge_supports_the_same_dynamic_lora_chain_policy(self) -> None:
        graph = build_api_graph(
            load_fixture("native_bridge_api.json"),
            base_render_config(
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
            ),
            available_files(),
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
        validate_two_stage_graph(graph)

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
