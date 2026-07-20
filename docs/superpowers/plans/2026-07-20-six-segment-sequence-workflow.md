# Six-Segment Sequence Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Deliver one manually usable ComfyUI workflow that can render any enabled subset of six sequential Wan I2V clips, resume from selected existing MP4s, save enabled clips, and produce one merged final video.

**Architecture:** Reuse the verified native Wan I2V subgraph, adding an `IMAGE` frames output alongside its video output. Six outer subgraph instances are chained by `ImageFromBatch` tail frames. `ComfySwitchNode` supplies lazy source, cached-clip, and enable branches so disabled or reused segments do not request model execution. A local gated-save output node avoids writing a placeholder clip when a segment is disabled or reused; a final `ImageBatch` chain creates the assembled image sequence passed to `CreateVideo` and `SaveVideo`.

**Tech stack:** ComfyUI 0.27 local workflow JSON, core `ComfySwitchNode`, `ImageFromBatch`, `ImageBatch`, `CreateVideo`, `SaveVideo`, installed Video Helper Suite loader, Python unittest.

## Global Constraints

- Keep the native 20-step Wan I2V high-to-low sampling topology unchanged.
- Do not restart ComfyUI, submit GPU work, or alter an active queue until the graph is validated.
- All switch `on_false` and `on_true` inputs must be connected.
- A disabled segment must not save a clip or force its Wan render branch.
- Resume video frames are selected only when the corresponding resume-source switch is selected.

---

### Task 1: Add a frame-producing native segment subgraph

**Files:**
- Modify: `tools/wan22_longform/workflows/ui/wan22_segment_i2v_native.json`
- Test: `tools/wan22_longform/tests/test_workflow_patch.py`

- [ ] Write a failing test that requires `FRAMES: IMAGE` beside the existing `VIDEO` output and checks that it originates at `DECODE`.
- [ ] Add the output mapping with an array-form link field and leave the existing video output untouched.
- [ ] Run the focused workflow test and require it to pass.

### Task 2: Add the local gated clip saver

**Files:**
- Create: `custom_nodes/wan22_longform_sequence/__init__.py`
- Test: `tools/wan22_longform/tests/test_sequence_workflow.py`

- [ ] Write a failing test for a `Wan22ConditionalSaveVideo` node contract: `enabled`, `video`, prefix/format/codec inputs; it saves only when enabled and forwards `VIDEO`.
- [ ] Implement the node by delegating enabled writes to the installed core `SaveVideo` behavior and returning the video unchanged when disabled.
- [ ] Run the focused test and require it to pass.

### Task 3: Generate the six-segment UI workflow

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/sequence_workflow.py`
- Create: `tools/wan22_longform/workflows/ui/wan22_six_segment_sequence_native.json`
- Test: `tools/wan22_longform/tests/test_sequence_workflow.py`

- [ ] Write failing structural tests for six titled segment instances, six `ENABLE_SEGMENT_nn` booleans, six `USE_EXISTING_SEGMENT_nn` cached-video selectors, five lazy continuation-tail routes, six gated clip savers, and one final `SaveVideo`.
- [ ] Generate the workflow from the native I2V definition, preserving exact model/sampler/title contracts while adding the outer sequence controls.
- [ ] Make the aggregate frame chain select only enabled clips and pass the combined frames to the final `CreateVideo`/`SaveVideo` pair.
- [ ] Run focused tests and JSON/link validation.

### Task 4: Deploy, visually validate, document, and verify

**Files:**
- Modify: `tools/wan22_longform/README.md`
- Modify: `tools/wan22_longform/workflows/PATCH_MAP.md`
- Modify: `tools/wan22_longform/src/wan22_longform/cli.py` only if deployment discovery does not already include the new JSON.

- [ ] Deploy only package workflows with `deploy-workflows --force`.
- [ ] Open the deployed graph in the local ComfyUI frontend and confirm all six titled sections and sequence controls are visible.
- [ ] Run the complete `tools/wan22_longform` test suite, `compileall`, Ruff, JSON parsing, link-contract checks, and `git diff --check`.
- [ ] Commit and push with an explicit refspec.
