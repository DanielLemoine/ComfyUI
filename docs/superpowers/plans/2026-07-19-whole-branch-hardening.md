# Wan2.2 Long-Form Whole-Branch Hardening Plan

For agentic workers: use subagent-driven-development or executing-plans. This is the single final-review fix wave; one implementer must address the complete set of findings before re-review.

Goal: Close the independent review's template-provenance, project-lineage, and lifecycle-integrity gaps without weakening local-only or immutable-record guarantees.

Architecture: Treat the registered workflow-template package as the sole proof of either official Wan template. Treat the project ID plus SHA-256 of the source manifest as a lineage key: new attempts, accepted render evidence, and assembly roots must agree on it. Bind an assembly root digest into every lifecycle decision so a failed-before-planning approval request is no longer silently mutable.

Tech Stack: Python 3.11, argparse, dataclasses, JSON/YAML, SHA-256, unittest, local ComfyUI filesystem only.

## Global Constraints

- Do not contact the internet, change core ComfyUI, stop/restart ComfyUI, or run GPU/live render work.
- Keep all runtime artifacts untracked and do not mutate existing smoke evidence.
- Official I2V and FLF template discovery must require the canonical registered package asset, sibling core-manifest entry, pinned SHA-256, local SHA-256, and matching native node; topology-only blueprints/user JSON are never proof.
- A project lineage is project_id plus the SHA-256 of the immutable source manifest bytes. Cross-project or legacy-unbound attempts may remain readable but cannot drive rendering continuation, bridge endpoint selection, or assembly.
- Assembly records must retain requires_review=True; reviewed approvals preserve zero frames only and never authorize trimming.
- Legacy attempts/assembly records must report an explicit unverified/legacy integrity outcome rather than a false verified result.
- Write tests first, observe focused red failures, then make the smallest passing implementation. Keep the fix in one coherent commit.

---

### Task 1: Final review hardening wave

Files:

- Modify: tools/wan22_longform/src/wan22_longform/inventory.py
- Modify: tools/wan22_longform/src/wan22_longform/cli.py
- Modify: tools/wan22_longform/src/wan22_longform/project.py
- Modify: tools/wan22_longform/src/wan22_longform/render.py
- Modify: tools/wan22_longform/README.md
- Modify: tools/wan22_longform/tests/test_inventory.py
- Modify: tools/wan22_longform/tests/test_cli_assembly_records.py
- Modify: tools/wan22_longform/tests/test_project_state.py
- Modify: tools/wan22_longform/tests/test_render_client.py

Interfaces:

- project_lineage(project) returns project_id and source_manifest_sha256 and is the canonical project identity used by new attempts and assembly roots.
- Accepted attempt/assembly input evidence includes the lineage plus SHA-256s for source-manifest, provenance.json, render-metadata.json, qc.yaml, the terminal accepted decision, and the selected video.
- collect_preflight accepts an optional ProjectConfig. A preflight without model-role proof cannot say READY.

- [ ] Add failing I2V provenance tests: a matching blueprints JSON is rejected; a missing manifest entry is rejected; a changed registered package I2V byte is rejected; the registered video_wan2_2_14B_i2v.json manifest SHA and local SHA are accepted only when equal to 6eea9b627b10fcfaf3e75a43aad2c58d8daabdbf72b32ede1602c668cac376bb and it contains WanImageToVideo.
- [ ] Add failing preflight tests: no project/model-role proof is BLOCKED; a required high/low/VAE/text model absent from the configured inventory is BLOCKED; environment/custom-node evidence records ComfyUI, frontend, PyTorch, CUDA, and node revisions or explicit unavailable values; non-node __pycache__ and tests directories are excluded.
- [ ] Add failing lineage tests: an accepted attempt copied from another project or with a changed source-manifest/provenance/render-metadata/QC/acceptance-decision hash cannot be selected for project assembly, continuation, or FLF endpoint use. A new assembly root serializes the current lineage and each input's immutable evidence.
- [ ] Add failing lifecycle tests: change assembly.json requested boundary approvals on a planned to failed record and require integrity failure; change an intermediate decision/root linkage and require integrity failure; legacy records without the root/chain fields remain readable but report explicit legacy/unverified integrity rather than verified.
- [ ] Run the focused modules before implementation:

    C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_inventory tools.wan22_longform.tests.test_cli_assembly_records tools.wan22_longform.tests.test_project_state tools.wan22_longform.tests.test_render_client

Expected: red failures caused by missing registered-I2V verification, missing project lineage, missing lifecycle root/chain checks, and incomplete preflight gate.

- [ ] Implement minimal registered-I2V verification parallel to FLF. Refactor only enough shared code to make the canonical template ID/filename/pinned hash/manifest entry/local-byte check identical in behavior for both node families. Do not accept blueprints or generic template roots as official proof.
- [ ] Implement and persist project lineage at attempt creation and assembly-record creation. Verify it while enumerating accepted attempts, resolving accepted video/bridge endpoints, rehashing assembly inputs, and inspecting a record through a project. Bind the exact immutable attempt evidence listed above. Keep legacy objects loadable but reject them from privileged paths.
- [ ] Add an immutable assembly-root digest and append-only lifecycle linkage. Each transition must reference the current root digest and predecessor decision hash; integrity checks must validate the complete linkage for final and failed records. Mark records lacking those fields as legacy/unverified, never verified.
- [ ] Extend preflight evidence and blockers for project model roles and runtime/custom-node revisions. Discover workflow profiles without assuming user/default; if the active profile cannot be proven, report candidates rather than claiming one.
- [ ] Update README only where needed so its integrity and preflight statements match the actual exact behavior and legacy status.
- [ ] Re-run the focused modules; expect green. Then run:

    C:\Program Files\Python311\python.exe -m unittest discover -s tools\wan22_longform\tests -p "test_*.py"
    C:\Program Files\Python311\python.exe -m compileall -q tools\wan22_longform\src
    C:\Program Files\Python311\python.exe -m ruff check tools/wan22_longform/src tools/wan22_longform/tests
    git diff --check

Expected: full suite green, compile/lint/diff checks clean.

- [ ] Commit all task files plus this plan using a direct subject such as Harden Wan22 provenance and record integrity.

## Execution Handoff

Run one final-review fix implementer against this complete task, then perform a fresh broad whole-branch review and live read-only validation. Do not reuse the older cross-project smoke assembly as proof after lineage enforcement; it must fail closed.

---

### Task 2: Final traceability, model-role, and resume recovery wave

**Files:**

- Modify: `tools/wan22_longform/src/wan22_longform/project.py`
- Modify: `tools/wan22_longform/src/wan22_longform/render.py`
- Modify: `tools/wan22_longform/src/wan22_longform/config.py`
- Modify: `tools/wan22_longform/src/wan22_longform/workflow.py`
- Modify: `tools/wan22_longform/src/wan22_longform/inventory.py`
- Modify: `tools/wan22_longform/src/wan22_longform/cli.py`
- Modify: focused project/render/config/workflow/inventory/CLI test modules and `README.md` where behavior changes.

**Requirements:**

- [x] Write failing tests that mutate every accepted-attempt provenance artifact: source manifest, workflow snapshot, request snapshot, configured workflow, submission request, submission provenance, and their referenced hashes. `status`, continuation/FLF selection, and project assembly must reject the altered attempt.
- [x] Write failing tests that declare non-default VAE/text-encoder model names and verify the submitted native API graph patches those exact loader targets; missing declared files must fail before submission.
- [x] Write failing segment and bridge recovery tests for upload failure, prompt-submission failure, and wait-timeout interruption. Prepared evidence must be hash-verified/idempotent; a recorded prompt ID must permit polling a rendering attempt without duplicate queue submission.
- [x] Write a failing planned-state integrity test: changed provenance/workflow/request/selected-input evidence must produce `integrity_failed`, never `verified`.
- [x] Write failing extra-model-path parser tests using canonical YAML block scalar model paths, multiple config inputs, and role-correct root discovery. Do not treat the literal `|` as a directory.
- [x] Write a failing CLI preflight test where `BLOCKED` returns nonzero and emits status/blockers; a ready fixture remains success.
- [x] Run the focused modules and observe red before production changes.
- [x] Implement complete transitive accepted-evidence verification and serialize it into accepted/assembly evidence. Do not accept a video merely because its final file hash still matches.
- [x] Retain all four declared model roles in resolved config, validate them against inventory, and patch unique VAE and text-encoder loader targets in both native graph templates.
- [x] Make interruption recovery explicit and idempotent: persist queue identity immediately after submission, retry safely only when nothing was submitted, and reattach rendering attempts through their recorded prompt ID.
- [x] Make `inspect_attempt_integrity()` validate the evidence appropriate to every lifecycle state and report incomplete/legacy evidence distinctly.
- [x] Parse ComfyUI extra-model path YAML with `yaml.safe_load`, including block scalars and active config files; combine configured and default roots with correct role kinds.
- [x] Make `preflight` a true gate: structured status/blockers output and nonzero exit for BLOCKED.
- [x] Re-run focused tests, full suite, compileall, Ruff, and `git diff --check`; commit a coherent follow-up and append the hardening report.
