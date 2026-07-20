# Wan2.2 long-form operator package

This package supplies two native ComfyUI API/UI workflows and a local-only operator runner:

- `wan22_segment_i2v_native`: a quality-first Wan 2.2 image-to-video segment.
- `wan22_bridge_flf2v_native`: a native first/last-frame bridge with two explicit endpoint images.
- `wan22_longform.cli`: immutable attempts, QC evidence, accepted-only assembly, and copy-only workflow deployment.

It is intentionally more than a single graph. The graphs create media; the runner preserves the evidence needed to choose continuation frames, retry safely, and assemble only reviewed outputs.

## Local setup

Run the commands from this package with its source directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\tools\wan22_longform\src"
python -m wan22_longform.cli preflight --comfy-root D:\AI\ComfyUI --comfy-url http://127.0.0.1:8188
python -m wan22_longform.cli validate-project .\tools\wan22_longform\projects\example\project.yaml
```

The active local ComfyUI installation uses shared model roots under `D:\AI\models`; its launch configuration uses `D:\AI\input`, `D:\AI\outputs`, and `D:\AI\temp` rather than the default `ComfyUI\input` and `ComfyUI\output` folders. The runner uploads only to the configured loopback ComfyUI API and keeps project attempts under the project directory.

Install the model files before rendering:

- `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `vae/wan_2.1_vae.safetensors`

Run preflight again after any ComfyUI, model, or template change. It is the gate that confirms native node availability and the registered official FLF template provenance.

## UI workflows

Deploy only the package UI files to an explicit directory. The command never deletes target files and refuses an existing target unless `--force` is supplied.

```powershell
python -m wan22_longform.cli deploy-workflows `
  --target D:\AI\ComfyUI\user\default\workflows\wan22_longform
```

Use `--force` only to replace the package's matching files in that target. It preserves unrelated user files. The API JSON files are for the runner; the UI JSON files are for inspecting or manually executing the same native topology.

## Project manifest

Start at `projects/example/project.yaml`. It is deliberately identity-disabled and uses an adult, fully clothed, neutral subject. Add a real anchor file at `projects/example/inputs/anchor.png` before validation or rendering.

The companion `prompts/*.txt` files are editable operator references. The sample manifest repeats its active prompt text intentionally; this release does not pretend to have an implicit prompt-file loader.

Important fields:

- `workflow_api` and `bridge_workflow_api` point at the clean native API graphs.
- `models` names the high- and low-noise Wan UNET files; they must differ.
- `model_files` is a portable inventory declaration. Replace it with local `model_roots` when you want the runner to discover installed names from disk.
- `shots[].segments[]` is the explicit I2V render plan.
- `bridges[]` requires either both `first_image` and `last_image`, or an explicit `base_source_image` used deliberately for both endpoints. The runner never guesses a tail or head frame.
- `assembly_order` is mandatory for `assemble-project`; it prevents chronology from being inferred from filesystem timestamps.

The first bridge in the example uses `base_source_image` only as a technical smoke configuration. Before a story bridge, replace it with the two exact QC-approved endpoint paths.

## Operator flow

```powershell
# Validate without changing the source manifest.
python -m wan22_longform.cli validate-project .\project.yaml

# I2V: one segment or every declared segment in a shot.
python -m wan22_longform.cli render-segment .\project.yaml S010 S010_C001
python -m wan22_longform.cli render-shot .\project.yaml S010

# FLF: only explicit endpoints; 17, 33, 49, 65, or 81 frames are legal.
python -m wan22_longform.cli render-bridge .\project.yaml B010 --timeout 1800

# Review and record a human decision. A continuation frame must be a hash-verified QC candidate.
python -m wan22_longform.cli qc-contact-sheet .\attempts\S010\S010_C001\attempt-...
python -m wan22_longform.cli accept .\attempts\S010\S010_C001\attempt-... `
  --note "clean motion and stable anatomy" `
  --continuation-frame .\attempts\...\candidate-frames\tail\frame-000016.png
python -m wan22_longform.cli reject .\attempts\... --note "camera jump at frame 8"
python -m wan22_longform.cli retry .\attempts\... --note "retry with lower motion control"

# Resume only existing immutable planned attempts; it never rerenders accepted work.
python -m wan22_longform.cli resume .\project.yaml --timeout 1800
python -m wan22_longform.cli status .\project.yaml

# Assemble only hash-verified accepted outputs. No timeline order is inferred for a project.
python -m wan22_longform.cli assemble-shot .\project.yaml S010 --qc-approved
python -m wan22_longform.cli assemble-project .\project.yaml --qc-approved
```

`render-shot` deliberately submits only declared I2V segments. It does not choose continuation frames or silently create a bridge. Submit `render-bridge` after you have explicitly written the selected endpoints into the manifest.

The `--timeout` default is 1800 seconds because a cold two-UNET Wan load can take longer than a short HTTP timeout on smart-offload hardware. The CLI still permits only loopback ComfyUI URLs.

## Preset escalation and identity

Use the presets in sequence: `P0_IDENTITY_BASELINE`, `P1_BASE_CONTROL`, `P2_BALANCED_MYSTIC`, `P3_MYSTIC_MOTION`, then `P4_GENERAL_FALLBACK`. Do not jump to a stronger adapter merely to conceal a continuity or prompt problem. The policy layer keeps high-/low-noise LoRA routing and mutually exclusive permissiveness adapters checked.

Identity LoRA remains pending until a real local file is installed and named in `loras.identity`. Use `single_both` only when one verified identity LoRA is appropriate for both experts, or `split` when verified high/low files are available. Never claim identity lock while `mode: none` is active.

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

If validation says a model is missing, correct the filename or `model_roots`; do not rename a checkpoint to satisfy the manifest. If local schema validation fails, refresh preflight and use the captured local `object_info`. If a bridge refuses to render, check that it has a stable `id` and `shot_id`, two explicit existing endpoint files (or one explicit base source), and an allowed length. If a deployment target exists, choose a new package directory or explicitly use `--force`; the tool will not clear it.

Unverified assumptions remain explicit: production creative quality and identity fidelity require a supplied anchor/identity LoRA and human QC; no benchmark proves a particular prompt will preserve a subject through an arbitrary long sequence; RIFE and Topaz configurations are intentionally outside this local package.
