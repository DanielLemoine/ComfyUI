from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.config import (  # noqa: E402
    ConfigError,
    IdentityLora,
    LoraSlot,
    ModelFiles,
    Models,
    Permissiveness,
    ResolvedRenderConfig,
    validate_lora_policy,
)


def resolved_config(**changes: object) -> ResolvedRenderConfig:
    values: dict[str, object] = {
        "models": Models(
            high="wan-high.safetensors",
            low="wan-low.safetensors",
            vae="wan-vae.safetensors",
            text_encoder="wan-text.safetensors",
        ),
        "permissiveness": Permissiveness(mode="none"),
        "identity": IdentityLora(),
        "vbvr": LoraSlot(branch="high"),
        "motion": LoraSlot(branch="high"),
        "corrective": LoraSlot(branch="low"),
        "dangerous_override": False,
        "denylist": ("body", "breast", "bust"),
    }
    if "permissiveness_mode" in changes:
        values["permissiveness"] = Permissiveness(
            mode=str(changes.pop("permissiveness_mode"))
        )
    if changes.pop("wan_general_enabled", False):
        values["permissiveness"] = Permissiveness(
            mode=str(getattr(values["permissiveness"], "mode")),
            wan_general=LoraSlot(file="wan-general.safetensors", branch="low"),
        )
    if "identity_mode" in changes:
        values["identity"] = IdentityLora(
            mode=str(changes.pop("identity_mode")),
            file=str(changes.pop("identity_file")),
        )
    values.update(changes)
    return ResolvedRenderConfig(**values)  # type: ignore[arg-type]


class LoraPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_files = ModelFiles.from_names(
            {
                "wan-high.safetensors",
                "wan-low.safetensors",
                "wan-vae.safetensors",
                "wan-text.safetensors",
                "id.safetensors",
                "id-high.safetensors",
                "id-low.safetensors",
                "wan-general.safetensors",
                "body-focus.safetensors",
            }
        )

    def test_permissiveness_modes_are_exclusive(self) -> None:
        config = resolved_config(permissiveness_mode="mystic", wan_general_enabled=True)
        with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
            validate_lora_policy(config, self.model_files)

    def test_single_identity_routes_the_same_file_to_both_experts(self) -> None:
        config = resolved_config(identity_mode="single_both", identity_file="id.safetensors")
        validate_lora_policy(config, self.model_files)
        self.assertEqual(config.identity.high_file, "id.safetensors")
        self.assertEqual(config.identity.low_file, "id.safetensors")

    def test_split_identity_routes_explicit_branch_files(self) -> None:
        config = resolved_config(
            identity=IdentityLora(
                mode="split", high_file="id-high.safetensors", low_file="id-low.safetensors"
            )
        )
        validate_lora_policy(config, self.model_files)
        self.assertEqual(config.identity.high_file, "id-high.safetensors")
        self.assertEqual(config.identity.low_file, "id-low.safetensors")

    def test_missing_enabled_lora_is_rejected(self) -> None:
        config = resolved_config(
            vbvr=LoraSlot(file="missing.safetensors", branch="high")
        )
        with self.assertRaisesRegex(ConfigError, "configured LoRA is missing"):
            validate_lora_policy(config, self.model_files)

    def test_disabled_lora_does_not_need_a_file(self) -> None:
        config = resolved_config(vbvr=LoraSlot(file="missing.safetensors", branch="high", enabled=False))
        validate_lora_policy(config, self.model_files)

    def test_disabled_identity_does_not_need_branch_files(self) -> None:
        config = resolved_config(
            identity=IdentityLora(mode="single_both", enabled=False)
        )
        validate_lora_policy(config, self.model_files)

    def test_body_emphasis_lora_requires_dangerous_override(self) -> None:
        config = resolved_config(
            motion=LoraSlot(file="body-focus.safetensors", branch="high")
        )
        with self.assertRaisesRegex(ConfigError, "body-emphasis LoRA rejected"):
            validate_lora_policy(config, self.model_files)

    def test_named_dangerous_override_allows_denylisted_lora(self) -> None:
        config = resolved_config(
            motion=LoraSlot(file="body-focus.safetensors", branch="high"),
            dangerous_override=True,
        )
        validate_lora_policy(config, self.model_files)

    def test_high_and_low_models_must_differ(self) -> None:
        config = resolved_config(
            models=Models("same.safetensors", "same.safetensors", "wan-vae.safetensors", "wan-text.safetensors")
        )
        with self.assertRaisesRegex(ConfigError, "high and low model files must differ"):
            validate_lora_policy(config, self.model_files)

    def test_high_and_low_models_are_distinct_case_insensitively(self) -> None:
        config = resolved_config(
            models=Models("WAN-HIGH.safetensors", "wan-high.safetensors", "wan-vae.safetensors", "wan-text.safetensors")
        )
        with self.assertRaisesRegex(ConfigError, "high and low model files must differ"):
            validate_lora_policy(config, self.model_files)

    def test_configured_models_must_exist_in_discovery(self) -> None:
        config = resolved_config(
            models=Models("missing-high.safetensors", "wan-low.safetensors", "wan-vae.safetensors", "wan-text.safetensors")
        )
        with self.assertRaisesRegex(ConfigError, "configured high model is missing"):
            validate_lora_policy(config, self.model_files)

    def test_discovery_matches_model_names_case_insensitively(self) -> None:
        config = resolved_config(
            models=Models("WAN-HIGH.SAFETENSORS", "wan-low.safetensors", "wan-vae.safetensors", "wan-text.safetensors")
        )

        validate_lora_policy(config, self.model_files)

    def test_vbvr_and_motion_cannot_be_routed_to_low_noise(self) -> None:
        for name in ("vbvr", "motion"):
            with self.subTest(name=name):
                config = resolved_config(
                    **{name: LoraSlot(file="id.safetensors", branch="low")}
                )
                with self.assertRaisesRegex(ConfigError, "must be high-noise only"):
                    validate_lora_policy(config, self.model_files)

    def test_corrective_cannot_be_routed_to_high_noise(self) -> None:
        config = resolved_config(
            corrective=LoraSlot(file="id.safetensors", branch="high")
        )
        with self.assertRaisesRegex(ConfigError, "must be low-noise only"):
            validate_lora_policy(config, self.model_files)

    def test_permissiveness_adapters_cannot_be_routed_to_high_noise(self) -> None:
        for mode, slot_name in (("mystic", "mystic"), ("wan_general", "wan_general")):
            with self.subTest(mode=mode):
                config = resolved_config(
                    permissiveness=Permissiveness(
                        mode=mode,
                        **{slot_name: LoraSlot(file="id.safetensors", branch="high")},
                    )
                )
                with self.assertRaisesRegex(ConfigError, "must be low-noise only"):
                    validate_lora_policy(config, self.model_files)


if __name__ == "__main__":
    unittest.main()
