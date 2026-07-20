from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.config import (  # noqa: E402
    ModelFiles,
    ProjectConfig,
    load_presets,
    load_project,
    resolve_preset,
    validate_lora_policy,
)


class ConfigResolutionTests(unittest.TestCase):
    def test_load_project_and_resolve_keeps_source_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.yaml"
            project_path.write_text(
                """{
  "preset": "P3_MYSTIC_MOTION",
  "models": {"high": "wan-high.safetensors", "low": "wan-low.safetensors", "vae": "wan-vae.safetensors", "text_encoder": "wan-text.safetensors"},
  "loras": {"identity": {"mode": "single_both", "file": "identity.safetensors"}}
}
""",
                encoding="utf-8",
            )

            project = load_project(project_path)
            presets = load_presets(PROJECT_DIR / "config" / "presets.yaml")
            resolved = resolve_preset(project, presets)

            self.assertEqual(project.source["preset"], "P3_MYSTIC_MOTION")
            self.assertEqual(resolved.models.high, "wan-high.safetensors")
            self.assertEqual(resolved.models.low, "wan-low.safetensors")
            self.assertEqual(resolved.models.vae, "wan-vae.safetensors")
            self.assertEqual(resolved.models.text_encoder, "wan-text.safetensors")
            self.assertEqual(resolved.permissiveness.mode, "mystic")
            self.assertEqual(resolved.vbvr.strength, 0.25)
            self.assertEqual(resolved.motion.strength, 0.25)
            self.assertEqual(resolved.permissiveness.active.strength, 0.25)
            self.assertEqual(resolved.identity.high_file, "identity.safetensors")
            self.assertEqual(resolved.identity.low_file, "identity.safetensors")

    def test_resolved_config_uses_exact_project_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.yaml"
            project_path.write_text(
                """{
  "preset": "P0_IDENTITY_BASELINE",
  "models": {"high": "exact-high.safetensors", "low": "exact-low.safetensors", "vae": "exact-vae.safetensors", "text_encoder": "exact-text.safetensors"}
}
""",
                encoding="utf-8",
            )
            resolved = resolve_preset(
                load_project(project_path),
                load_presets(PROJECT_DIR / "config" / "presets.yaml"),
            )

            self.assertEqual(resolved.models.high, "exact-high.safetensors")
            self.assertEqual(resolved.models.low, "exact-low.safetensors")
            self.assertEqual(resolved.models.vae, "exact-vae.safetensors")
            self.assertEqual(resolved.models.text_encoder, "exact-text.safetensors")

    def test_indented_yaml_manifest_is_loaded_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.yaml"
            project_path.write_text(
                """preset: P0_IDENTITY_BASELINE
models:
  high: wan-high.safetensors
  low: wan-low.safetensors
  vae: wan-vae.safetensors
  text_encoder: wan-text.safetensors
""",
                encoding="utf-8",
            )

            project = load_project(project_path)

            self.assertEqual(project.source["models"]["high"], "wan-high.safetensors")

    def test_corrective_low_defaults_to_zero_and_preserves_configured_weight(self) -> None:
        presets = load_presets(PROJECT_DIR / "config" / "presets.yaml")
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.yaml"
            project_path.write_text(
                """{
  "preset": "P0_IDENTITY_BASELINE",
  "models": {"high": "wan-high.safetensors", "low": "wan-low.safetensors", "vae": "wan-vae.safetensors", "text_encoder": "wan-text.safetensors"},
  "loras": {
    "corrective": {
      "enabled": true,
      "file": "corrective.safetensors",
      "weight": 0.20
    }
  }
}
""",
                encoding="utf-8",
            )
            configured = resolve_preset(load_project(project_path), presets)

        for preset_name in ("P0_IDENTITY_BASELINE", "P1_BASE_CONTROL"):
            project = ProjectConfig(
                path=Path("project.yaml"),
                source={
                    "preset": preset_name,
                    "models": {"high": "wan-high.safetensors", "low": "wan-low.safetensors", "vae": "wan-vae.safetensors", "text_encoder": "wan-text.safetensors"},
                },
            )
            self.assertEqual(resolve_preset(project, presets).corrective.strength, 0.0)
        self.assertEqual(configured.corrective.strength, 0.20)
        self.assertTrue(configured.corrective.is_enabled)

    def test_catalog_contains_the_five_exact_non_escalating_presets(self) -> None:
        catalog = load_presets(PROJECT_DIR / "config" / "presets.yaml")

        self.assertEqual(
            {
                name: (
                    preset.vbvr_high,
                    preset.permissiveness_mode,
                    preset.permissiveness_low,
                    preset.motion_high,
                    preset.corrective_low,
                    preset.identity_high,
                    preset.identity_low,
                )
                for name, preset in catalog.presets.items()
            },
            {
                "P0_IDENTITY_BASELINE": (0.0, "none", 0.0, 0.0, 0.0, 0.90, 0.90),
                "P1_BASE_CONTROL": (0.25, "none", 0.0, 0.0, 0.0, 0.90, 0.90),
                "P2_BALANCED_MYSTIC": (0.25, "mystic", 0.25, 0.0, 0.0, 0.90, 0.90),
                "P3_MYSTIC_MOTION": (0.25, "mystic", 0.25, 0.25, 0.0, 0.90, 0.90),
                "P4_GENERAL_FALLBACK": (0.20, "wan_general", 0.20, 0.0, 0.0, 0.90, 0.90),
            },
        )

    def test_example_config_is_valid_without_optional_loras(self) -> None:
        project = load_project(PROJECT_DIR / "config" / "models.example.yaml")
        resolved = resolve_preset(
            project, load_presets(PROJECT_DIR / "config" / "presets.yaml")
        )

        self.assertEqual(resolved.corrective.strength, 0.0)

        validate_lora_policy(
            resolved,
            ModelFiles.from_names(
                {
                    "wan-high.safetensors",
                    "wan-low.safetensors",
                    "wan-vae.safetensors",
                    "wan-text-encoder.safetensors",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
