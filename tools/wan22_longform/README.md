# Wan2.2 long-form operator package

This package supplies two native ComfyUI API/UI render workflows, two keyframe utility workflows, one six-segment manual sequence workflow, and a local-only operator runner:

- `wan22_segment_i2v_native`: a quality-first Wan 2.2 image-to-video segment.
- `wan22_bridge_flf2v_native`: a native first/last-frame bridge with two explicit endpoint images.
- `wan22_extract_last_frame`: saves the exact final frame from a completed MP4 for use as a later shot's start image.
- `wan22_keyframe_author_qwen_edit`: uses the installed Qwen Image Edit model to author a planned FLF end keyframe from a supplied source image.
- `wan22_six_segment_sequence_native`: six optionally rendered or reused I2V sections with one assembled final MP4.
- `wan22_longform.cli`: immutable attempts, QC evidence, accepted-only assembly records, and copy-only workflow deployment.

It is intentionally more than a single graph. The graphs create media; the runner preserves the evidence needed to choose continuation frames, retry safely, and assemble only reviewed outputs. Each assembly is a separate immutable record, so accepting a segment never turns that reusable source attempt into an assembly state.

## Local setup

Run the commands from this package with its source directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\tools\wan22_longform\src"
python -m wan22_longform.cli preflight --comfy-root D:\AI\ComfyUI --comfy-url http://127.0.0.1:8188 `
  --project .\tools\wan22_longform\projects\example\project.yaml --bind-project
python -m wan22_longform.cli validate-project .\tools\wan22_longform\projects\example\project.yaml
```

The active local ComfyUI installation uses shared model roots under `D:\AI\models`; its launch configuration uses `D:\AI\input`, `D:\AI\outputs`, and `D:\AI\temp` rather than the default `ComfyUI\input` and `ComfyUI\output` folders. The runner uploads only to the configured loopback ComfyUI API and keeps project attempts under the project directory.

Install the model files before rendering:

- `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `vae/wan_2.1_vae.safetensors`

Run preflight again after any ComfyUI, frontend, PyTorch/CUDA, custom-node, model, or template change. READY requires a trustworthy interpreter under `comfy_root`, available ComfyUI/runtime and custom-node revision evidence, the supplied project's exact high/low/VAE/text-encoder filenames in role-correct roots, enabled optional LoRAs in LoRA roots, native schemas, and the registered canonical I2V and FLF package assets with their pinned manifest/local SHA-256s. Preflight materializes enabled optional LoRAs into the schema-checked API graph. Inventory combines configured extra model paths with default ComfyUI model roots; high/low must come from diffusion-model or UNET roots, VAE from VAE roots, and text encoders from text/CLIP roots. Without `--project`, model-role proof is unavailable and preflight remains BLOCKED. Workflow discovery enumerates local profile candidates when it cannot prove which profile is active; it never assumes `user/default`.

`preflight` is a machine-readable gate: it writes evidence either way and prints JSON with `artifact_dir`, `status`, and `blockers`. It exits `0` only for `READY`; `BLOCKED` exits nonzero and no render command should follow. With both `--project` and `--bind-project`, a `READY` preflight atomically updates that manifest's `object_info` path and SHA-256 to the snapshot it just recorded, then revalidates the bound manifest. A blocked preflight leaves the manifest untouched. Pass every active ComfyUI configuration with `--extra-model-paths-config <path> [<path> ...]` (repeat the option as needed); canonical YAML block-scalar paths and the default `models/` roots are both inventoried.

## UI workflows

Deploy only the package UI files to an explicit directory. The command never deletes target files and refuses an existing target unless `--force` is supplied.

```powershell
python -m wan22_longform.cli deploy-workflows `
  --target D:\AI\ComfyUI\user\default\workflows\wan22_longform
```

Use `--force` only to replace the package's matching files in that target. It preserves unrelated user files. The API JSON files are for the runner; the UI JSON files are for inspecting or manually executing the same native topology.

The persisted I2V graph uses stable executable titles: `PROMPT_POSITIVE`, `PROMPT_NEGATIVE`, `START_IMAGE`, and `VIDEO_PREVIEW`. These are the runner's patch points; do not rename them in a deployed copy.

### Manual ComfyUI execution

The deployed render UI workflows are complete manual canvases: their `LoadImage` and
`SaveVideo` nodes are already wired to the native Wan subgraph. Open them from
ComfyUI's **Workflows** sidebar under `wan22_longform`; do not open the
`.NOT_BUILT` marker.

- **`wan22_segment_i2v_native`**: choose/upload the opening frame in
  `START_IMAGE`, write the action in `PROMPT_POSITIVE`, and leave the verified
  model selections at their defaults unless changing them deliberately. The
  visible `LORA_IDENTITY_HIGH` and `LORA_IDENTITY_LOW` nodes use the installed
  `sgfw` LoRA at `0.8`; include `sgfw` in the positive prompt and keep both
  strengths equal. Set both strengths to `0` to disable the manual identity
  adapter. The default is 640x640 for 5 seconds. Click **Run**; the MP4 is
  saved below `D:\AI\outputs\wan22_longform\segment_`.
- **`wan22_bridge_flf2v_native`**: this is the single-clip first/last-frame
  workflow. Choose the start and planned end keyframes in `BRIDGE_FIRST_IMAGE`
  and `BRIDGE_LAST_IMAGE`, set `PROMPT_POSITIVE`, then click **Run**. Its
  default is 640x640 and 81 frames; the MP4 is saved below
  `D:\AI\outputs\wan22_longform\bridge_`.
- **`wan22_extract_last_frame`**: after a clip is complete, paste its absolute
  MP4 path into `EXTRACT_EXACT_LAST_FRAME` and click **Run**. It writes exactly
  one PNG below `D:\AI\outputs\wan22_longform\flf_last_frame_`. Upload that
  PNG into the next FLF clip's first-image loader. This utility does not render
  video and is intentionally a separate workflow.
- **`wan22_keyframe_author_qwen_edit`**: choose/upload the current shot's
  opening frame (or the previous shot's extracted final frame) in
  `KEYFRAME_SOURCE_IMAGE`, then describe only the desired final action/pose in
  `END_KEYFRAME_EDIT_INSTRUCTION`. The official Qwen Edit 2509 path preserves
  the source image's framing while producing a planned endpoint PNG below
  `D:\AI\outputs\wan22_longform\flf_planned_end_`. Use that PNG in
  `BRIDGE_LAST_IMAGE` of `wan22_bridge_flf2v_native`. Its Lightning mode is on
  by default (4 steps, CFG 1); turn `Enable Lightning LoRA` off inside the
  Qwen subgraph for the normal 20-step, CFG 4 route. Do not load the Wan-only
  `sgfw` video LoRA into this image-edit graph.

An FLF end image is a planned keyframe, not something that can be extracted
from an as-yet-unrendered clip. Create or edit that future endpoint first, then
render the FLF shot. After the shot is rendered, extract its actual last frame
to become the next shot's start image; independently plan the next shot's end
keyframe before rendering that shot.
- **`wan22_six_segment_sequence_native`**: this is the long-form manual
  canvas. Each `SEGMENT_nn` has its own private subgraph definition: enter the
  segment to edit its `PROMPT_POSITIVE`, `PROMPT_NEGATIVE`, or add a LoRA.
  Those edits affect that segment only. `ENABLE_SEGMENT_nn` decides whether that numbered section appears in
  the final sequence. `USE_EXISTING_SEGMENT_nn` means use an already rendered
  MP4 instead of generating that section again: upload/select that MP4 in its
  matching `CACHED_SEGMENT_nn` Video Helper Suite node. Leave it false to
  generate from the segment's own prompt. Segment 1 always starts with
  `SEED_IMAGE`. For segments 2–6, keep `USE_PREVIOUS_TAIL_nn` true to continue
  from the exact final frames selected by `CONTINUATION_TAIL_nn` (default: 8),
  or set it false and paste a saved MP4 path into `RESUME_VIDEO_nn` to start a
  new continuation point. Wan conditions the next render on that input batch;
  the workflow then removes exactly those repeated conditioning frames before
  it appends the fresh motion (eight with the default tail, or one when resuming
  from a saved MP4). Each freshly
  generated enabled section is saved below `D:\AI\outputs\wan22_longform\segments\`;
  `SAVE_FINAL_VIDEO` writes the direct assembled sequence. For example, to
  preserve clips 1–3 and rerender only 4–5, enable 1–5, set `USE_EXISTING_SEGMENT_01`
  through `_03` true and choose their MP4s, leave 4–5 false, keep their previous-tail
  toggles true, and disable 6. This workflow performs direct frame assembly;
  use the separate FLF workflow only when you deliberately want a generated
  transition rather than a cut.

If ComfyUI was already open when deploying an update, use the **Refresh** button
in the Workflows sidebar and reopen the workflow. A missing-image warning before
you choose an input image is expected; it is not a graph-load error.

## Project manifest

Start at `projects/example/project.yaml`. It is deliberately identity-disabled and uses an adult, fully clothed, neutral subject. Add the two user-owned anchor files named in its `shots[].anchor_image` fields before validation or rendering.

The companion `prompts/*.txt` files are editable operator references. The sample manifest repeats its active prompt text intentionally; this release does not pretend to have an implicit prompt-file loader.

Important fields:

- The strict v1 manifest requires project identity/mode/duration, a resolvable `preset`, render dimensions/FPS and the fixed V1 codec declarations, a seed family, exact high/low/VAE/text-encoder filenames, output declarations, QC/retry policy, workflow hashes, a hash-bound local `object_info` snapshot, and an environment snapshot. The checked-in example intentionally starts without an `object_info` binding: run `preflight --project <manifest> --bind-project` before `validate-project`. `validate-project` checks the contract without changing the YAML.
- `workflow_api` and `bridge_workflow_api` point at the clean native API graphs.
- `models` names the high-/low-noise Wan UNETs, VAE, and text encoder; all four exact filenames are recorded.
- `model_files` is only a portable declaration. Normal validation and every render require role-correct, on-disk `model_roots`; the runner independently verifies each declared high/low UNET, VAE, and text-encoder file before upload or submission.
- `shots[].segments[]` is the explicit I2V render plan. A normal opening image resolves in this order: `segment.opening_image`, then `shot.anchor_image`, then the optional project fallback `inputs.opening_frame`.
- A `continue_from` segment is different: it can only use exactly one accepted same-shot upstream attempt's selected, hash-verified **tail** candidate. The attempt must match the current project id and source-manifest SHA-256, and its sealed provenance/render metadata/QC/acceptance evidence must still match. It cannot combine an opening-image override with continuation.
- Strict-manifest seed families use `render.seed_base + segment.seed_offset`; an explicit `segment.seed` is the only override. A legacy top-level request seed is only a low-level compatibility fallback.
- Frame timing is a contract: `render.frames` must equal `floor(duration_seconds * render.generation_fps + 1)`. Segments and the top-level request may not override `frames` or `length`; each segment derives its exact frame count from the project render contract. A bridge's declared `frames`, its duration, and the project FPS must agree, and the explicit `assembly_order` must agree with that timing within one frame. The runner seals the submitted `length` and `CreateVideo.fps`, then verifies fetched media frames, FPS, duration, width, and height before metadata or QC is written. Resume re-verifies that sealed timing against the video before QC can continue.
- V1 assembly codecs are deliberately fixed: `review_mp4_codec: h264` and `master_codec: ffv1`; the ProRes edit master uses the fixed `prores_ks` profile 3 command. Other manifest codec values are rejected rather than silently ignored.
- A story `flf2v` bridge may use literal accepted-QC paths, but new immutable projects should use the manifest-stable pair `first_image: accepted_selected_tail` and `last_image: accepted_selected_head`. At render time those selectors require exactly one current-lineage accepted `from_segment`/`to_segment` attempt and resolve its sealed tail/head selections. The bridge attempt records the resolved literal paths, hashes, upstream attempts, and acceptance-decision evidence. `base_source_image` is allowed only with `purpose: technical_smoke`, never as a story-bridge fallback.
- A story `flf2v` bridge also requires `from_segment` and `to_segment` in its own shot. Its entry in `assembly_order` must sit directly between those exact segment entries. The runner verifies that its selected tail/head inputs come from those declared accepted attempts.
- `direct` and `intentional_cut` are non-rendered transition policies. Each declares same-shot `from_segment` and `to_segment`; those two segment entries must be adjacent in `assembly_order`. They do not create a bridge attempt or an assembly media item.
- `external_control` follows the same adjacent-segment policy contract. This runner neither renders nor imports external-control media; an external clip needs a future explicit imported-output and provenance contract, so it must not be represented as a synthetic bridge item today.
- A bridge id may not reuse a segment id in the same shot. Technical-smoke bridges are never allowed in `assembly_order`.
- `assembly_order` is mandatory for project and shot assembly; it contains production segment media and rendered story FLF bridges only. A shot must include every one of those media items exactly once, so a declared story FLF bridge cannot silently become a direct cut while a policy transition cannot become a phantom clip.
- `output_root` and the three `outputs` paths are immutable output-role templates. Each declared role must live inside `output_root`; the runner records their resolved root and role filenames, then writes to a unique record-local directory by default (or an explicit `--output-dir`). It never writes directly into the declared root. `--rife-review-mp4` is honored only with `--request-rife`.

The first bridge in the example is deliberately a `technical_smoke` configuration. For a new story bridge, declare the semantic selector pair in the original manifest before rendering either source segment; do not edit the manifest after acceptance just to insert runtime attempt paths.

```yaml
first_image: accepted_selected_tail
last_image: accepted_selected_head
from_segment: S010_C001
to_segment: S010_C002
```

## Operator flow

```powershell
# Validate without changing the source manifest.
python -m wan22_longform.cli validate-project .\project.yaml

# I2V: one segment or every declared segment in a shot.
python -m wan22_longform.cli render-segment .\project.yaml S010 S010_C001
# render-shot is only for independent segments. It refuses a shot with continue_from entries.
python -m wan22_longform.cli render-shot .\project.yaml S020

# FLF: semantic accepted selectors or literal accepted-QC endpoints; 17, 33, 49, 65, or 81 frames are legal.
python -m wan22_longform.cli render-bridge .\project.yaml B010 --timeout 1800

# Review and record human selections. The continuation is a tail candidate; --head-frame seals the head used by a later semantic FLF bridge. Acceptance writes an atomic terminal decision plus an immutable marker that hashes it; each new evidence path is claimed without replacement, so concurrent conflicting decisions cannot overwrite one another. If an interruption leaves only the terminal decision, evidence recovery revalidates it and idempotently publishes the missing marker.
python -m wan22_longform.cli qc-contact-sheet .\attempts\S010\S010_C001\attempt-...
python -m wan22_longform.cli accept .\attempts\S010\S010_C001\attempt-... `
  --note "clean motion and stable anatomy" `
  --continuation-frame .\attempts\...\candidate-frames\tail\frame-000016.png `
  --head-frame .\attempts\...\candidate-frames\head\frame-000000.png
# This accepted tail is now the immutable input for the dependent segment.
python -m wan22_longform.cli render-segment .\project.yaml S010 S010_C002
# Accept S010_C002 with --head-frame before its semantic story bridge is rendered.
python -m wan22_longform.cli accept .\attempts\S010\S010_C002\attempt-... `
  --note "clean entry frame for FLF bridge" `
  --head-frame .\attempts\...\candidate-frames\head\frame-000000.png
python -m wan22_longform.cli reject .\attempts\... --note "camera jump at frame 8"
python -m wan22_longform.cli retry .\attempts\... --note "retry with lower motion control"

# Resume only existing immutable planned or safely queued rendering attempts. Before upload or submission it re-hashes the sealed provenance, source manifest, workflow/request snapshots, selected inputs, configured graph, request, and submission provenance. Upload-only interruption can retry from hash-verified prepared evidence; a recorded queue identity reattaches to that prompt and polls without submitting a duplicate. Output MP4s, history, metadata, QC, candidate frames, and contact sheets are staged and atomically published, so a partial output is replaced on resume instead of being trusted. A submit intent without a prompt ID is an unknown outcome and deliberately stops for manual ComfyUI lookup. Resume never rerenders accepted work or runs FFmpeg assembly automatically.
python -m wan22_longform.cli resume .\project.yaml --timeout 1800
python -m wan22_longform.cli status .\project.yaml

# Assemble only hash-verified accepted outputs. No timeline order is inferred for a project or shot.
# Each command creates a new immutable assembly record under assembly-records/.
python -m wan22_longform.cli assemble-shot .\project.yaml S010 --qc-approved
python -m wan22_longform.cli assemble-project .\project.yaml --qc-approved
# Optional RIFE target: it is an explicit protected target, never an ignored flag.
python -m wan22_longform.cli assemble-project .\project.yaml --qc-approved --request-rife `
  --rife-review-mp4 .\review\rife-review.mp4
```

For a reviewed direct boundary that must retain both endpoint frames, create a new immutable project assembly record with:

```powershell
python -m wan22_longform.cli assemble-project project.yaml --qc-approved --approve-no-trim-boundary 1 "Reviewed at 200%; preserve both frames."
```

This approval preserves both frames; it cannot authorize trimming. Only a full-fidelity, reverified exact duplicate can trim a frame automatically.

`render-shot` deliberately refuses any dependent continuation before submitting anything. Review and accept the upstream tail, then submit the dependent segment with `render-segment`. It never chooses a frame or silently creates a bridge. For semantic story bridges, keep the selector pair in the manifest from the beginning, accept the declared source tail and destination head, then submit `render-bridge` without rewriting the manifest.

The `--timeout` default is 1800 seconds because a cold two-UNET Wan load can take longer than a short HTTP timeout on smart-offload hardware. The CLI still permits only loopback ComfyUI URLs.

### Assembly records and recovery

`assemble-shot` and `assemble-project` first validate the strict manifest and require every accepted source to match the current project lineage (`project_id` plus source-manifest SHA-256). Each input binds and re-hashes the source snapshot, base workflow/request snapshots, `provenance.json`, input-upload evidence, configured graph, submission request/provenance, persisted queue identity, history, `render-metadata.json`, `qc.yaml`, terminal acceptance decision, and selected video before creating an append-only record in `assembly-records/<scope>/assembly-...`. The root digest covers the complete request, approvals, lineage, and inputs. Every lifecycle decision binds that root, its own digest, and the predecessor decision hash across `planned → assembling → assembled → final` or any `failed` transition. Reviewed approvals always retain `requires_review=true` and zero trim. A failed plan or FFmpeg run is recorded as `failed` and its partial targets are never overwritten.

Use `status project.yaml` to inspect every record. It state-validates planned, rendering, rendered, review, and accepted evidence; partial safe recovery is reported as `incomplete`, while a changed hash is reported as `integrity_failed` with its original `recorded_state`. It also rechecks project lineage, the root/decision chain, all accepted-source evidence, the serialized plan, boundary evidence, and finalized outputs read-only. A structurally unreadable record root is reported per-record and does not hide intact siblings. Legacy attempts and assembly records remain readable but report `legacy_unverified`; they cannot drive continuation, FLF endpoint selection, render resume, or reviewed assembly. `resume` excludes unverified, failed, and unknown-submit records from recovery guidance and never restarts an assembly, overwrites a target, or re-renders an accepted segment. Start a new reviewed assembly request with a new `--output-dir` when recovering from a partial explicit target; the default record-local output directory is already unique.

The low-level `assemble` command accepts arbitrary files only for diagnostics and requires `--diagnostic-only`. It is not the reviewed project assembly path and does not produce an accepted-project assembly record.

## Preset escalation and identity

Use the presets in sequence: `P0_IDENTITY_BASELINE`, `P1_BASE_CONTROL`, `P2_BALANCED_MYSTIC`, `P3_MYSTIC_MOTION`, then `P4_GENERAL_FALLBACK`. Do not jump to a stronger adapter merely to conceal a continuity or prompt problem. The policy layer keeps high-/low-noise LoRA routing and mutually exclusive permissiveness adapters checked.

The manual I2V canvas includes the locally installed `sgfw` identity LoRA on
both model branches. Its `0.8` strength is a starting point, not a fidelity
guarantee; inspect the generated subject and adjust both branches together.
The runner remains manifest-controlled: use `single_both` only when one
verified identity LoRA is appropriate for both experts, or `split` when
verified high/low files are available. Never claim identity lock while a
runner manifest has `mode: none` active.

For cinematic continuity, keep one clear camera action per segment, preserve clothing/setting details in the positive prompt, and change only one control dimension per retry. Treat a first/last-frame bridge as a transition tool, not a replacement for reviewing the actual segment tail and next-segment head.

## QC rubric and finishing tools

Inspect the contact sheet before accepting: subject identity/wardrobe stability, hands/face, motion direction, camera move, lighting, background geometry, edge frames, and any abrupt change at the planned boundary. Acceptance stores a non-empty operator note and optional selected candidate frame in the immutable decision chain.

Native assembly creates a review MP4 plus FFV1 and ProRes edit masters after exact boundary evidence and duration checks. It drops audio in V1 rather than claiming A/V timing preservation. RIFE remains a separately requested follow-up only after native assembly validates. Use an NLE for story edit, pacing, titles, audio, and color decisions; use Topaz only as a later upscale/interpolation finishing stage after the edit is locked.

## Extension matrix, limits, and troubleshooting

| Area | Included | Not included |
| --- | --- | --- |
| Wan generation | Native core I2V and FLF nodes | Community node substitutes |
| Optional LoRAs | Policy-routed model-only adapters | Bundled LightX2V in quality graphs |
| QC | Candidate frames, contact sheets, immutable decisions | Automatic artistic approval |
| Assembly | FFmpeg native timeline validation | Safe audio preservation in V1 |
| Upscale/interpolation | Explicit RIFE readiness marker | Automatic RIFE or Topaz execution |

If validation says a model is missing, correct the filename or `model_roots`; do not rename a checkpoint to satisfy the manifest. If local schema validation fails, refresh preflight and use the captured local `object_info`. If a bridge refuses to render, check that it has a stable `id`/`shot_id`, an allowed `frames` count, and either accepted tail/head QC endpoints or an explicitly marked `technical_smoke` base source. If a deployment target exists, choose a new package directory or explicitly use `--force`; the tool will not clear it.

Unverified assumptions remain explicit: production creative quality and identity fidelity require a supplied anchor/identity LoRA and human QC; no benchmark proves a particular prompt will preserve a subject through an arbitrary long sequence; RIFE and Topaz configurations are intentionally outside this local package.
