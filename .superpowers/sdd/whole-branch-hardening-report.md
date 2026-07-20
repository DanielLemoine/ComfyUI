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
