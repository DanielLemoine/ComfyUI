# Wan2.2 Long-Form Post-Repair Correction Plan

Goal: close every P1 and P2 item from the 2026-07-20 static re-review before an isolated live smoke is considered.

Scope: Python helper, packaged native workflows, tests, and concise operator documentation only. No ComfyUI request, GPU work, service action, or runtime-artifact mutation is permitted.

## Contracts to restore

- A `LoadImage.image` enum marked `image_upload: true` is dynamic only through an explicit trusted mapping. Preflight may trust the clean graph's empty placeholder; submission may trust only the sanitized names returned by the active upload calls. Other enums remain strict.
- `preflight --project ... --bind-project` is the supported atomic manifest-binding path. It writes the captured local `object_info` reference and SHA-256 only after a READY preflight; validate/render require that binding.
- Strict V1 timing has one resolved frame count for each submitted segment/bridge. Segment/request frame overrides must either be represented in the project duration contract or be rejected; this wave keeps a single global segment frame count and calculates bridge duration from its explicit frame count.
- Returned media must match the configured native width, height, frame count, FPS, and one-frame duration tolerance before metadata or QC is written. The sealed timing record contains both expected and actual dimensions.
- Acceptance uses an append-only terminal decision plus an immutable decision-hash marker. A crash between those atomic writes is recoverable: after complete evidence revalidation, the reader idempotently publishes only the missing marker.
- Every non-replacing evidence publish atomically claims its final path. Concurrent contenders receive `FileExistsError`; they cannot overwrite an already sealed numeric decision or acceptance marker.
- Ordinary preflight materializes enabled optional model-only LoRAs and validates both their local LoRA-root inventory and their resulting graph schema.
- `render.workflow` is rejected as obsolete, and persisted I2V titles are normalized to `PROMPT_POSITIVE`, `PROMPT_NEGATIVE`, `START_IMAGE`, and `VIDEO_PREVIEW`.

## Test-first execution

1. Add real-choice-list `LoadImage.image` tests proving that only an explicit empty placeholder or freshly trusted sanitized upload name bypasses the stale enum; confirm static enum choices still fail.
2. Add CLI binding coverage that starts from an unbound manifest, uses `preflight --bind-project`, then validates the newly bound manifest without a manual edit. Add a blocked-preflight no-write assertion.
3. Add timing tests for a mismatched segment frame override, returned width/height mismatch, and sealing of dimensions.
4. Add accepted-attempt recovery coverage which removes the terminal decision marker after the decision and proves that evidence revalidates and reconstructs that marker without rerendering.
5. Add enabled-optional-LoRA preflight/schema coverage and exact persisted-title tests.
6. Run the focused test set and capture the intentional RED result before implementation.
7. Implement the smallest owner-layer changes: validator policy in `workflow.py`; manifest contract/binding in `config.py` and `cli.py`; project-aware graph materialization in `inventory.py`; timing/submission policy in `render.py`; recoverable terminal acceptance marking in `project.py`; non-overwriting immutable publication in `atomic.py`; and normalized graph/document references.
8. Re-run focused tests, full discovery, compileall, Ruff, and `git diff --check`. Record evidence in the branch hardening report and the final Documents report, then commit and push the named branch explicitly.

## Expected file set

- `tools/wan22_longform/src/wan22_longform/{atomic,workflow,config,cli,inventory,render,project}.py`
- `tools/wan22_longform/{README.md,projects/example/project.yaml,workflows/PATCH_MAP.md}`
- `tools/wan22_longform/workflows/{api,ui}/wan22_segment_i2v_native*.json`
- focused `atomic`, `workflow`, `inventory`, `cli`, `config`, `render`, and `project-state` tests
- `.superpowers/sdd/whole-branch-hardening-report.md` and this plan

## Verification commands

```powershell
C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_workflow_patch tools.wan22_longform.tests.test_inventory tools.wan22_longform.tests.test_cli tools.wan22_longform.tests.test_config tools.wan22_longform.tests.test_render_client tools.wan22_longform.tests.test_project_state
C:\Program Files\Python311\python.exe -m unittest discover -s tools\wan22_longform\tests -p "test_*.py"
C:\Program Files\Python311\python.exe -m compileall -q tools\wan22_longform\src
C:\Program Files\Python311\python.exe -m ruff check tools/wan22_longform/src tools/wan22_longform/tests
git diff --check
```
