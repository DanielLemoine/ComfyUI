# Wan2.2 Long-Form Whole-Branch Hardening Report

Date: 2026-07-19

Branch: `codex/wan22-longform-v1`

Base: `38e4a3a7e698a66f35b29adaa6582a9a2499c669`

Implementation commit: `be90af9b` (`Harden Wan22 provenance and record integrity`)

## Outcome

Completed the final-review hardening wave for the Wan2.2 long-form helper. The implementation now fails closed when a project, attempt, accepted segment, or assembly record crosses project boundaries or loses cryptographic continuity. Legacy attempt and assembly records remain readable for inspection but are explicitly `legacy_unverified` and cannot be used for privileged resume, continuation, final-frame-first-frame, or assembly operations.

The existing reviewed-boundary policy is preserved: reviewed joins still require `requires_review=True`, and reviewed boundaries still require `trim_right_frames=0`.

## Implemented Changes

### Registered template and preflight hardening

- Added a pinned native Wan2.2 I2V template identity and SHA-256 alongside the existing FLF identity.
- Restricted READY template discovery to the canonical registered package path with its sibling core manifest, pinned local file hash, and native-node validation.
- Rejected generic blueprint JSON and altered or manifest-less registered templates.
- Made project-aware preflight validate the configured high-noise UNet, low-noise UNet, VAE, and text encoder against discovered local model inventory.
- Added explicit ComfyUI revision, frontend, PyTorch, and CUDA evidence fields, including explicit unavailable states when evidence cannot be collected.
- Added custom-node revision evidence while excluding test and cache directories.
- Replaced the hard-coded `user/default` workflow assumption with discovery across `user/*/workflows` and explicit candidate reporting.

### Project, attempt, and accepted-evidence integrity

- Added a stable project lineage identifier containing the project ID and current project-manifest SHA-256.
- Stored lineage in newly created attempt and provenance records.
- Added project-bound attempt verification and surfaced attempt integrity in project status.
- Sealed accepted evidence for the source manifest, provenance, render metadata, QC record, selected output, lineage, and terminal acceptance decision.
- Revalidated accepted evidence before resume, continuation, FLF endpoint use, accepted-input selection, and assembly.
- Made cross-project reuse fail before ffprobe or renderer activity.
- Preserved legacy readability while preventing legacy-unverified evidence from privileged use.

### Assembly record integrity

- Bound each assembly root to project lineage and a complete canonical snapshot of accepted input evidence.
- Added a root digest over the full assembly request/root.
- Added predecessor hashes and self-digests to each transition, forming a verified decision chain.
- Revalidated root lineage, all accepted input evidence, root digest, and every decision link during status and execution paths.
- Detected root approval-list tampering, decision-chain mutation, source/provenance/metadata/QC/output mutation, and terminal acceptance mutation.
- Kept old records readable as `legacy_unverified`; partially upgraded or inconsistent records fail integrity validation.

### Documentation and tests

- Updated `tools/wan22_longform/README.md` with project-aware preflight, READY semantics, evidence requirements, legacy behavior, and assembly-chain behavior.
- Added the controlling execution plan at `docs/superpowers/plans/2026-07-19-whole-branch-hardening.md`.
- Expanded focused unit coverage across inventory, project state, render resume/continuity, assembly records, CLI status, legacy compatibility, and cross-project rejection.
- Updated the continuity fixture to represent the new project-bound provenance contract.

## Changed Files

- `docs/superpowers/plans/2026-07-19-whole-branch-hardening.md`
- `tools/wan22_longform/README.md`
- `tools/wan22_longform/src/wan22_longform/cli.py`
- `tools/wan22_longform/src/wan22_longform/inventory.py`
- `tools/wan22_longform/src/wan22_longform/project.py`
- `tools/wan22_longform/src/wan22_longform/render.py`
- `tools/wan22_longform/tests/test_cli_assembly_records.py`
- `tools/wan22_longform/tests/test_continuity_contract.py`
- `tools/wan22_longform/tests/test_inventory.py`
- `tools/wan22_longform/tests/test_project_state.py`
- `tools/wan22_longform/tests/test_render_client.py`

This report is finalized in a separate report-only commit so that it can name the immutable implementation commit above.

## TDD Evidence

### Focused RED

Command:

```powershell
C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_inventory tools.wan22_longform.tests.test_cli_assembly_records tools.wan22_longform.tests.test_project_state tools.wan22_longform.tests.test_render_client
```

Observed before production edits: 73 tests ran with 7 expected failures. The failures demonstrated the missing controls:

- canonical I2V identity was absent;
- runtime environment keys and revisions were absent;
- a blueprint I2V document was accepted;
- a cross-project assembly input was accepted;
- mutation of assembly requested approvals was not detected;
- a new attempt did not contain lineage;
- cross-project resume proceeded as far as ffprobe.

### Focused GREEN

The same focused area, expanded by the new regression cases, completed successfully:

```text
Ran 84 tests in 11.937s
OK
```

### Full suite

```text
Ran 198 tests in 20.726s
OK (skipped=1)
```

### Static checks

- `C:\Program Files\Python311\python.exe -m compileall -q tools\wan22_longform\src` — exit 0.
- `C:\Program Files\Python311\python.exe -m ruff check tools/wan22_longform/src tools/wan22_longform/tests` — `All checks passed!`.
- `git diff --check` — exit 0 after removing one Markdown EOF whitespace issue.

## Self-Review

- Confirmed all privileged consumers use project-bound verified evidence rather than trusting path shape or mutable JSON fields.
- Confirmed new writes are sealed while legacy records remain inspectable without being silently trusted.
- Confirmed decision-chain validation covers both predecessor continuity and each decision's own payload.
- Confirmed preflight reports unavailable evidence explicitly rather than manufacturing READY evidence.
- Confirmed reviewed-boundary invariants remain unchanged.
- Confirmed no networking, GPU execution, ComfyUI startup, or runtime-artifact modification was used during implementation or validation.
- Confirmed `tools/wan22_longform/artifacts/` remains untracked and untouched.

## Residual Concerns

None within the requested static/unit-test scope. A live ComfyUI/GPU validation was intentionally excluded by the task constraints.

## Fix Report: Whole-Branch Re-Review

Date: 2026-07-19

### Findings addressed

- **C1 — trusted template root:** Registered I2V/FLF candidates must now be direct children of an exact, separately enumerated `comfyui_workflow_templates_json/templates` root under a known ComfyUI embedded/virtual environment. Package-shaped ancestry under `blueprints`, `workflow_templates`, or web assets cannot qualify, even when it contains matching bytes and a fabricated sibling manifest.
- **C2 — immutable resume evidence:** New attempts seal `provenance.json` in `attempt.json`. Before any render upload or submission, the runner re-hashes the planned source-manifest snapshot, workflow snapshot, request snapshot, provenance seal/lineage, project inputs, and selected inputs. Stored `accepted_tail` and `accepted_qc_candidate` inputs reload the recorded upstream attempt, require current-project accepted-evidence verification, and must reproduce the exact candidate path, SHA-256, and kind.
- **I1 — fail-closed READY evidence:** Runtime queries use only a known embedded or virtual-environment interpreter below `comfy_root`; they never fall back to the helper process interpreter. READY now requires the runtime interpreter plus available ComfyUI revision, frontend, PyTorch, CUDA, and enumerated custom-node revision evidence. Model inventory combines configured extras and default roots, and model-role proof is root-kind aware: high/low use diffusion-model or UNET roots, VAE uses VAE roots, and text uses text/CLIP roots.

Reviewed-boundary behavior was not changed: reviewed joins still require `requires_review=true` and `trim_right_frames=0`.

### TDD RED

Command:

```powershell
C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_inventory tools.wan22_longform.tests.test_cli_assembly_records tools.wan22_longform.tests.test_project_state tools.wan22_longform.tests.test_render_client
```

Observed before follow-up production edits:

```text
Ran 93 tests in 12.525s
FAILED (failures=13)
```

The failures covered the package-shaped blueprint spoof, configured-extra/default-root omission, wrong-root role acceptance, missing runtime/custom-node READY blockers, helper-interpreter leakage, four planned snapshot/provenance mutations, two upstream accepted-evidence mutations, and an incorrect accepted QC candidate kind.

### GREEN verification

Focused:

```text
Ran 93 tests in 12.482s
OK
```

Full:

```text
Ran 207 tests in 20.203s
OK (skipped=1)
```

Static validation:

- `C:\Program Files\Python311\python.exe -m compileall -q tools\wan22_longform\src`
- `C:\Program Files\Python311\python.exe -m ruff check tools/wan22_longform/src tools/wan22_longform/tests`
- `git diff --check`

All follow-up work remained local and static/unit-test only. No GPU, ComfyUI runtime, network service, or untracked runtime artifact was used or modified.

### CUDA sentinel follow-up

The final Important re-review found that the CUDA probe's literal `unavailable` output still had exit code 0 and was therefore represented as available evidence. CUDA evidence now fails closed when the probe returns empty output or the case-insensitive `unavailable` sentinel; the existing runtime evidence blocker then prevents READY.

TDD RED:

```text
Ran 30 inventory tests in 0.982s
FAILED (failures=1)
```

Final GREEN:

```text
Ran 30 inventory tests in 1.184s
OK

Ran 208 tests in 21.856s
OK (skipped=1)
```

No reviewed-boundary, runtime-execution, or artifact behavior changed.

## Fix Report: Task 2 Final Traceability, Model Roles, and Resume Recovery

Date: 2026-07-19

### Scope and guardrails

This follow-up remained static and unit-test only. It did not start, stop, query, or modify ComfyUI, the GPU, network services, or the ignored `tools/wan22_longform/artifacts/` runtime directory.

### TDD RED

Before the production edits, the focused Task 2 command exposed the missing model-role, YAML, lifecycle, preflight, and recovery behavior:

```powershell
C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_workflow_patch tools.wan22_longform.tests.test_inventory tools.wan22_longform.tests.test_project_state tools.wan22_longform.tests.test_render_client tools.wan22_longform.tests.test_cli
```

```text
Ran 110 tests
FAILED (failures=15, errors=6, skipped=1)
```

### Corrections

- Accepted evidence is now transitive. The immutable planned snapshot, selected input provenance, configured workflow, submission request, versioned submission provenance, queue identity, history, metadata, QC, and output are rehashed and compared before status, continuation/FLF selection, or assembly can use an accepted attempt.
- Resolved model configuration is strict for high-noise UNet, low-noise UNet, VAE, and text encoder. Native I2V and FLF graph patching now targets the unique `UNETLoader`, `VAELoader`, and `CLIPLoader` nodes, and missing VAE/text files fail before upload or submission.
- Render preparation is immutable and restart-safe. A submission intent is written before queue submission; a returned prompt ID is sealed in `queue.json` before polling or the rendering transition. Planned upload interruptions can repeat matching evidence, queued rendering attempts reattach to their recorded ID, and an intent without a queue ID fails closed as an unknown submit outcome.
- Lifecycle integrity now validates state-appropriate evidence. Tampered planned data becomes `integrity_failed` in CLI payloads with its recorded state preserved; incomplete and legacy evidence remain distinct and cannot be promoted to trusted execution paths.
- Extra model paths now use `yaml.safe_load`, accept canonical block scalars and multiple active files, and combine role-correct configured roots with standard `models/*` roots.
- `preflight` emits JSON containing `status`, `blockers`, and `artifact_dir`; it returns zero only for `READY` and nonzero for `BLOCKED`.

### GREEN verification

Focused Task 2 modules:

```text
Ran 131 tests
OK (skipped=1)
```

Full suite:

```text
Ran 217 tests in 28.727s
OK (skipped=1)
```

Static checks passed:

```powershell
C:\Program Files\Python311\python.exe -m compileall -q tools\wan22_longform\src
C:\Program Files\Python311\python.exe -m ruff check tools/wan22_longform/src tools/wan22_longform/tests
git diff --check
```

Implementation commit: this Task 2 follow-up commit on `codex/wan22-longform-v1`.

## Supplement: Gate 5 Story-FLF Binding and Final Review Corrections

### Manifest-stable story endpoints

The prior story-FLF contract required literal QC file paths in `first_image` and `last_image`. Those paths do not exist until the source attempts render, while changing the manifest afterward changes its source hash and invalidates the source attempts' lineage.

Story bridges now support the immutable semantic pair:

```yaml
first_image: accepted_selected_tail
last_image: accepted_selected_head
```

The pair resolves only the declared `from_segment` tail and `to_segment` head, requires exactly one accepted current-lineage source attempt for each, rehashes their full accepted evidence, and writes the resolved path, SHA-256, upstream attempt, and immutable acceptance-decision evidence into the bridge attempt. `accept --head-frame <QC-head>` seals the chosen destination head alongside the existing tail selection. Accepted bridge and assembly verification rechecks that upstream relationship transitively.

Regression coverage proves that the original manifest bytes remain unchanged through two source acceptances and a 33-frame semantic FLF render; that the bridge can enter the accepted shot timeline; and that tampered, multiple, wrong-kind, or wrong-lineage sources fail before upload/submission.

### Independent-review corrections

- Preflight now reads frozen manifest model mappings through `Mapping`, so a normal `load_project()` object supplies all four model roles instead of creating a false `BLOCKED` result.
- Only `extra_model_paths.yaml` is implicitly trusted. Backup/glob lookalikes are ignored, and every CLI-supplied active config must exist or preflight fails closed.
- Recovery is now idempotent across the metadata/QC boundary. An interrupted rendering attempt with sealed metadata advances without re-polling or rewriting it; an interrupted rendered attempt reuses/verifies existing QC and completes only the review transition. CLI resume admits those verified incomplete states without resubmitting the prompt.
- Accepted-submission tampering coverage now includes the `input_upload` referenced hash as well as request, source manifest, base workflow, and configured workflow hashes.

### Verification

RED coverage was observed before the semantic selector implementation (missing `--head-frame` and selector lifecycle support) and before the final-review fixes (frozen mappings, stale configs, and metadata/QC resume all failed).

Final focused checks:

```text
Ran 62 inventory/render tests
OK

Ran 85 continuity/project/render/CLI tests
OK (skipped=1)
```

Full suite completed successfully with 230 collected tests and one intentional skip. `compileall`, Ruff, and `git diff --check` also passed. All validation remained local/static/unit-test only; no ComfyUI, GPU, network service, or runtime artifact was touched.

## Supplement: Canonical I2V, Schema, Timing, and Recovery Hardening

Date: 2026-07-20

### Completed controls

- Rebased the packaged I2V UI subgraph onto the installed registered canonical `video_wan2_2_14B_i2v.json` asset (`6eea9b627b10fcfaf3e75a43aad2c58d8daabdbf72b32ede1602c668cac376bb`), subgraph `84e2cf3f-de93-40ef-ab22-b9375296917b` / `Image to Video (Wan2.2)`.
- Preserved the official normal quality path and its false-selected 4-step switches while excluding all LightX/LoRA loader nodes and true branches. The UI retains `Duration × FPS → floor(a*b+1) → WanImageToVideo.length`, with the same FPS routed to `CreateVideo`.
- Made the normal preflight gate validate both hash-pinned API graphs against the captured local `/object_info`, including numeric bounds/steps and choice widgets. A CLI-level regression proves an invalid pinned native input returns `BLOCKED`.
- Made `generation_frames` and `generation_fps` declarative graph inputs, enforced frame/duration/assembly contracts, verified fetched media timing before metadata/QC, fixed V1 codec policy, and rejected dead `seed_increment` configuration.
- Required on-disk, role-correct model roots before normal validation/render submission; `model_files` remains declaration-only.
- Added atomic staged publication for output fetches and evidence artifacts. Resume replaces nonzero partial segment and bridge outputs and recovers history, metadata, QC, candidate-frame, and contact-sheet interruption boundaries without duplicate submission.

### Final verification

```text
Ran 246 tests in 46.194s
OK (skipped=1)
```

- `ruff check src tests` — passed.
- `git diff --check` — passed.
- Canonical UI provenance/hash/topology verification — passed (`25` retained normal-branch nodes, `50` links).

This supplement remained static/unit-test only. It did not start, stop, query, or modify ComfyUI, the GPU, external services, or the ignored runtime-artifact directory.
