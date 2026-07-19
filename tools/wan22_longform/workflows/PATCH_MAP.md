# Wan2.2 workflow patch map

## Segment I2V API titles

- Models: `MODEL_HIGH`, `MODEL_LOW`
- Sampling wrappers: `MODEL_SAMPLING_HIGH`, `MODEL_SAMPLING_LOW`
- Two-stage samplers: `SAMPLER_HIGH`, `SAMPLER_LOW`
- Inputs: `SEGMENT_FIRST_IMAGE`, `POSITIVE_PROMPT`
- Native conditioning: `I2V_CONDITIONING`

The clean base contains no `LoraLoaderModelOnly` nodes. After `validate_lora_policy` confirms configured model and LoRA filenames against discovered local model files, `build_api_graph` inserts enabled adapters in this fixed order:

- High expert: `MODEL_HIGH` → `LORA_VBVR_HIGH` → `LORA_MOTION_HIGH` → `LORA_IDENTITY_HIGH` → `MODEL_SAMPLING_HIGH`
- Low expert: `MODEL_LOW` → one `LORA_PERMISSIVENESS_LOW` → `LORA_CORRECTIVE_LOW` → `LORA_IDENTITY_LOW` → `MODEL_SAMPLING_LOW`

Disabled or zero-strength adapters are omitted rather than represented with placeholder filenames.

## Bridge FLF titles — blocked

No native Wan FLF graph was emitted because no verified official template contains `WanFirstLastFrameToVideo`.

When an official native graph becomes available, its canonical input titles are `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE`. `BRIDGE_START` and `BRIDGE_END` are documentation aliases only; executable graphs must use the canonical titles.
