# Wan2.2 workflow patch map

## Segment I2V API titles

- Models: `MODEL_HIGH`, `MODEL_LOW`
- Sampling wrappers: `MODEL_SAMPLING_HIGH`, `MODEL_SAMPLING_LOW`
- Two-stage samplers: `SAMPLER_HIGH`, `SAMPLER_LOW`
- Stable segment titles: `START_IMAGE`, `PROMPT_POSITIVE`, `PROMPT_NEGATIVE`, `VIDEO_PREVIEW`
- Native conditioning: `I2V_CONDITIONING`

The clean base contains no `LoraLoaderModelOnly` nodes. `build_api_graph(base_graph, render)` remains valid when no optional adapter is enabled. If an enabled adapter is requested, callers must supply an explicit `ModelFiles` available-file inventory; the builder independently refuses an adapter whose filename is absent from it. `validate_lora_policy` remains the configuration-policy gate. With both checks satisfied, the builder inserts enabled adapters in this fixed order:

- High expert: `MODEL_HIGH` → `LORA_VBVR_HIGH` → `LORA_MOTION_HIGH` → `LORA_IDENTITY_HIGH` → `MODEL_SAMPLING_HIGH`
- Low expert: `MODEL_LOW` → one `LORA_PERMISSIVENESS_LOW` → `LORA_CORRECTIVE_LOW` → `LORA_IDENTITY_LOW` → `MODEL_SAMPLING_LOW`

Disabled or zero-strength adapters are omitted rather than represented with placeholder filenames.

The manual I2V UI is intentionally the exception: it visibly routes the
installed `wan22-i2v-a14b\\sgfw\\wan22_i2v_a14b_sgfw.safetensors` adapter from
`MODEL_HIGH` through `LORA_IDENTITY_HIGH` and from `MODEL_LOW` through
`LORA_IDENTITY_LOW` before the two sampling wrappers. Both nodes start at
strength `0.8` and can be set to `0` together for a manual neutral run. This
does not change the clean API graph or the runner's manifest policy.

## Quality baseline

Both quality graphs use the verified normal 20-step two-stage topology: high sampling uses Euler/simple, steps 0–10, and leftover noise enabled; low sampling adds no fresh noise and uses steps 10–20 with leftover noise disabled. The I2V baseline remains faithful to its installed official normal branch (`ModelSamplingSD3` shift `5`, CFG `3.5`); the official FLF baseline uses shift `8`, CFG `4`. The 4-step LightX2V lane remains preview-only and is not present in either quality graph.

## Official I2V provenance and normal branch

- Installed source: `D:\AI\ComfyUI\venv\Lib\site-packages\comfyui_workflow_templates_json\templates\video_wan2_2_14B_i2v.json`
- Template identifier: `video_wan2_2_14B_i2v`
- Verified SHA-256: `6eea9b627b10fcfaf3e75a43aad2c58d8daabdbf72b32ede1602c668cac376bb`
- Canonical subgraph: `84e2cf3f-de93-40ef-ab22-b9375296917b` — `Image to Video (Wan2.2)`

The packaged UI is a direct capture of that canonical normal branch with the entire 4-step LoRA lane removed. The normal high/low models, 20 steps, CFG 3.5, and split step 10 connect directly to the two-stage pipeline; no incomplete selector node remains. `PrimitiveFloat Duration` and `PrimitiveFloat FPS` feed `ComfyMathExpression` `floor (a * b + 1)`; its integer output feeds `WanImageToVideo.length`, while the FPS primitive feeds `CreateVideo.fps`. The API graph keeps the same normal 20-step semantics as concrete executable values.

## Bridge FLF API titles

- Models and sampling wrappers: `MODEL_HIGH`, `MODEL_LOW`, `MODEL_SAMPLING_HIGH`, `MODEL_SAMPLING_LOW`
- Canonical endpoint image loaders: `BRIDGE_FIRST_IMAGE`, `BRIDGE_LAST_IMAGE`
- Native conditioning: `FLF_CONDITIONING` (`WanFirstLastFrameToVideo`)
- Two-stage samplers: `BRIDGE_SAMPLER_HIGH`, `BRIDGE_SAMPLER_LOW`
- Output stages: `BRIDGE_DECODE`, `BRIDGE_CREATE_VIDEO`, `BRIDGE_SAVE_VIDEO`

`BRIDGE_START` and `BRIDGE_END` are documentation aliases only. Executable graphs and runner patching use `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE` exclusively.

The bridge base remains free of `LoraLoaderModelOnly` nodes. `build_api_graph(base_graph, render, available_files)` applies the same dynamic high/low LoRA ordering and inventory checks as the segment graph.

## Official FLF provenance

- Source: `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/52a53af170145cfd579e6e6f6334ce25e9b8cf10/templates/video_wan2_2_14B_flf2v.json`
- Upstream pin: `Comfy-Org/workflow_templates@52a53af170145cfd579e6e6f6334ce25e9b8cf10`
- Verified SHA-256: `9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf`
- Installed source: `D:\AI\ComfyUI\venv\Lib\site-packages\comfyui_workflow_templates_json\templates\video_wan2_2_14B_flf2v.json`

The UI bridge is a native LiteGraph subgraph adapted from that official template’s normal branch. Its API counterpart contains only executable core nodes; frontend-only notes are not included in the API graph.

## Six-segment manual sequence UI

`wan22_six_segment_sequence_native.json` reuses the canonical I2V subgraph six
times without modifying its normal 20-step high-to-low topology. Each section
has visible top-level `PROMPT_POSITIVE_nn` and `PROMPT_NEGATIVE_nn` nodes. They
feed the matching subgraph instance, so prompt edits remain per segment rather
than changing the shared subgraph definition. Each section also has
`ENABLE_SEGMENT_nn` and `USE_EXISTING_SEGMENT_nn`: disabled sections do not
enter the final assembly, while existing sections load their selected MP4 via
`CACHED_SEGMENT_nn` instead of requesting a Wan render. Segments 2–6 also have
`USE_PREVIOUS_TAIL_nn` to select the final `CONTINUATION_TAIL_nn` frames of the
accumulated prior sequence (default: 8) or the final frame from `RESUME_VIDEO_nn`.
Every `ComfySwitchNode` supplies both branches, so its
lazy branch selection protects disabled Wan and resume-video work. Fresh enabled
sections remove the exact number of conditioned input frames before frame
assembly: the selected tail length for a previous-tail continuation or one for
a saved-MP4 resume. They then pass through `Wan22ConditionalSaveVideo`; cached
sections are not rewritten. `ASSEMBLE_ENABLED_SEGMENTS` and `SAVE_FINAL_VIDEO`
make one direct frame-concatenated MP4. Generated FLF transitions remain
deliberate separate operations rather than silently inserted cuts.
