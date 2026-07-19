# Wan2.2 Long-Form ComfyUI System Design

## Goal

Build a local-only, production-oriented Wan2.2 I2V A14B system that renders short, traceable video units and assembles them into durable cinematic shots. The primary production mode is cut-based cinematic assembly; bounded continuation and first/last-frame bridges are secondary tools for individual shots.

## Scope and source hierarchy

`C:\Users\Daniel\Downloads\wan22_longform_bundle\wan22_longform_comfyui_graph_spec.md` defines the functional behavior. `wan22_longform_codex_build_prompt.md` defines the implementation process, repository layout, validation gates, and final reporting. Where the documents differ, the functional result stays compatible with the graph specification while the build prompt supplies executable naming and process requirements.

V1 includes native I2V and FLF2V render paths, the API runner, QC state and artifacts, deterministic FFmpeg assembly, and unit/integration test coverage. Optional Lightning, context-window, VACE, Fun, ATI, ReCamMaster, Wan-Animate, and Fun/InP workflows are documented but not dependencies of the core.

## Operating constraints

- Target repository: `D:\AI\ComfyUI`; implementation branch: `codex/wan22-longform-v1` in a linked worktree.
- Active runtime: `http://127.0.0.1:8188`, ComfyUI 0.27.0, Python 3.11.6, PyTorch 2.13.0+cu130, RTX 5090.
- Shared model root: `D:\AI\models`; selected files are discovered and recorded rather than assumed.
- The current live ComfyUI render must not be stopped, restarted, queued behind, or otherwise disturbed until its queue is idle. Worktree-only development and non-GPU validation may continue meanwhile.
- No model or LoRA download is automatic. Missing artifacts produce inventory/checklist evidence and a clear validation error.
- Existing worktree and runtime user files are preserved. The only runtime installation action is copying validated UI workflow files into the discovered active workflow folder under `wan22_longform`.

## Architecture

```text
project.yaml + preset + prompt files
              |
              v
    Python runner / validation / state machine
              |
              +--> base workflow + dynamic optional-LoRA graph builder
              |                   |
              |                   v
              |               local ComfyUI /prompt
              |
              +--> immutable attempt evidence + head/tail frames + QC record
              |
              v
      FFmpeg normalization, boundary diagnostics, and assembly
              |
              v
         review MP4 + edit-friendly/lossless master
```

### 1. Workflow layer

The canonical UI sources live in `tools/wan22_longform/workflows/ui/`. They are cloned from the locally captured official native Wan templates after node schemas, topology, widget settings, and sampler wiring have been recorded in preflight evidence.

`wan22_segment_i2v_native.json` is a clean base workflow with no optional LoRA dependency. It retains the verified two-stage topology: high-noise model and sampler create the leftover-noise latent, then the low-noise model and sampler finish it without new initial noise. The source has unique, stable symbolic titles for all directly patchable base nodes.

`wan22_bridge_flf2v_native.json` is the same verified model/text/sampler/decode topology, replacing I2V conditioning with the installed first/last-frame conditioning node. Its canonical input titles are `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE`. `BRIDGE_START` and `BRIDGE_END` remain documented compatibility aliases for the graph-spec terminology; no ambiguous lookup is allowed.

The runner derives API-format graphs from the captured base API template. It only inserts a LoRA node if the corresponding feature is both enabled and backed by a discovered file. Inserted nodes receive stable title/type metadata and are inserted in this order:

```text
high: MODEL_HIGH -> VBVR_HIGH -> MOTION_HIGH -> IDENTITY_HIGH -> MODEL_SAMPLING_HIGH
low:  MODEL_LOW -> PERMISSIVENESS_LOW -> CORRECTIVE_LOW -> IDENTITY_LOW -> MODEL_SAMPLING_LOW
```

This dynamic strategy is deliberate: executable graphs never contain fake filenames or disabled optional nodes that require unavailable assets. The UI source contains a patch-map note that documents all optional symbolic titles, while actual graph nodes exist only when safely configured.

### 2. Domain package

`tools/wan22_longform/src/wan22_longform/` is a focused Python package:

- `config.py`: typed YAML loading, preset resolution, strict policy checks, and model-file validation.
- `inventory.py`: ComfyUI/runtime/model/custom-node discovery and evidence writing.
- `workflow.py`: title/type lookup, uniqueness checks, base-template validation, and dynamic API graph construction.
- `comfy_client.py`: local HTTP upload, queue, history, and optional WebSocket polling boundary.
- `project.py`: project manifest parsing and attempt/status persistence.
- `render.py`: segment and bridge request orchestration, immutable attempt creation, and result collection.
- `frames.py`, `qc.py`, `ffmpeg.py`, and `assembly.py`: deterministic frame extraction, contact sheets, QC records, media inspection, normalization plans, and assembly.
- `metadata.py`, `hashing.py`, and `errors.py`: provenance records, SHA-256 utilities, and concise domain errors.
- `cli.py`: explicit, non-destructive CLI commands; only `render-*`, decision, and assembly actions change project output state.

Configuration is YAML and remains readable outside Python. A project snapshot is copied into each output root; source manifests never mutate during render/preset resolution.

### 3. Policy enforcement

The manifest exposes exactly one `permissiveness.mode`: `none`, `mystic`, or `wan_general`. Schema and graph construction reject configurations that attempt to enable both broad permissiveness adapters. VBVR is high-noise only; `MOTION_HIGH` is high-noise only and disabled by default; `CORRECTIVE_LOW` is low-noise only and disabled until a vetted file is supplied.

Identity supports `single_both` and `split`. A single identity file produces both a high and a low node; a split configuration uses the explicit branch file. Identity is never silently applied to one expert only.

`denylist.yaml` holds case-insensitive patterns for known body-emphasis adapters. Any configured optional adapter matching the list raises a policy error unless the project opts into a named dangerous override. The default prompt templates explicitly protect adult identity, natural anatomy, clothing, continuity, and non-glamour framing.

### 4. Project and state model

The project manifest owns render settings, exact discovered model names, LoRA configuration, prompts, anchors, shots, segment seed offsets, continuation links, bridge choices, QC policy, and output location.

Each render creates a new immutable attempt directory. The status transition is constrained to:

```text
planned -> rendering -> rendered -> needs_review
needs_review -> accepted | rejected | retry_requested
accepted -> assembling -> assembled -> final
```

Decision commands write the operator decision, note, timestamp, parent attempt, and selected continuation frame. `resume` reads existing state and never rerenders an accepted attempt. Automated progress is permitted only when the manifest explicitly enables it.

### 5. QC and assembly

After a successful render, the runner writes the exact API request, rendered graph, hashes, metadata, output paths, first and last five decoded frames, a contact sheet, and `qc.yaml` initialized to `needs_review`. It records output duration as `frame_count / fps`.

The assembly planner first probes every accepted media file. It refuses direct concat for mismatched geometry, FPS/time base, pixel format, codec/profile, color metadata, or audio policy. It emits a normalization plan and runs all FFmpeg calls through one wrapper. Exact duplicate boundary frames are removed once; perceptually similar frames are diagnostics only unless a documented threshold and operator action approve removal. RIFE may be scheduled only after native QC and assembly; Topaz remains a finishing-only documented stage.

## Preflight and template evidence

Before writing executable workflow JSON, preflight captures:

```text
artifacts/preflight/environment.json
artifacts/preflight/model_inventory.json
artifacts/preflight/custom_nodes.json
artifacts/preflight/object_info.json
artifacts/preflight/official_template_inventory.md
artifacts/preflight/preflight_report.md
```

The implementation derives exact class names, input names, two-stage sampler boundary, and workflow representation from this evidence. If either required official native template is unavailable, workflow construction stops and the report becomes an actionable blocker; environment-independent runner components still build and test.

## Testing and verification

Unit tests use fixtures and never require a GPU. They cover mutual exclusivity, deny-list enforcement, branch-specific identity routing, missing/disabled LoRA behavior, title/type lookup uniqueness, distinct high/low models, two-stage no-fresh-noise topology, duplicate boundary handling, duration math, immutable retries, accepted-attempt resume behavior, normalization refusal, and RIFE ordering.

Installed-node validation reads captured `object_info.json` and confirms every emitted class exists. ComfyUI smoke tests are marked separately and will run only after the active task is idle. The base smoke test uses 17 or 33 low-resolution frames and all optional LoRAs disabled. LoRA and identity smoke tests are conditional on file availability. The full two-segment/bridge gate remains the prerequisite for any final-resolution sample claim.

## Delivery

The tracked package, source workflows, configs, tests, docs, and preflight/validation artifacts are committed from the isolated branch. Validated UI workflow copies are installed into the active runtime workflow folder only after they pass static and installed-node validation. The final report states exact evidence, missing model/LoRA requirements, tests and smoke outputs, pending identity integration, and the next operator command.
