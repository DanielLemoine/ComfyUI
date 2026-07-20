# Reviewed Boundary Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit an explicit reviewed no-trim direct cut while preserving exact-duplicate-only automatic trimming.

**Architecture:** Keep `BoundaryDecision.requires_review` true. Add a `BoundaryApproval` value object that can only approve a review-required, non-alpha, zero-trim boundary. Project/shot commands persist requested approvals in a new record, re-read them after source rehashing, and bind approval plus diagnostic evidence into plan and integrity records.

**Tech Stack:** Python 3.11, argparse, dataclasses, FFmpeg/ffprobe, unittest, PyYAML.

## Global Constraints

- Never clear `requires_review` to represent approval.
- Only a full-fidelity reverified exact duplicate may delete a frame.
- Approval preserves both frames and is unavailable to raw diagnostic `assemble`.
- Assembly records remain append-only; failed records stay failed.
- Test first, then implement the smallest safe behavior.

---

### Task 1: Planner approval model

**Files:**
- Modify: `tools/wan22_longform/src/wan22_longform/assembly.py`
- Modify: `tools/wan22_longform/tests/test_assembly_plan.py`

**Interfaces:**
- `BoundaryApproval(boundary_index: int, note: str)`.
- `approve_reviewed_boundaries(decisions, approvals, inputs) -> list[BoundaryDecision]`.
- `plan_assembly(..., boundary_approvals=())` and `AssemblyPlan.boundary_approvals`.

- [ ] Write failing tests that prove a review-required near duplicate can plan only with `BoundaryApproval(1, "Reviewed at 200%; retain both frames.")`, produces no `trim_boundary`, and still has `requires_review=True`.
- [ ] Add failing tests for blank, duplicate, out-of-range, alpha, and nonzero-trim approvals.
- [ ] Run `C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_assembly_plan`; expect red for missing approval API.
- [ ] Implement the dataclass and validation. `_trim_counts()` accepts approved review boundaries only when trim is zero; `_verify_exact_duplicate_boundary()` remains the only route to trim one frame.
- [ ] Re-run the test module; expect green.
- [ ] Commit with `git add tools/wan22_longform/src/wan22_longform/assembly.py tools/wan22_longform/tests/test_assembly_plan.py` and `git commit -m "Add reviewed no-trim boundary approvals"`.

### Task 2: Record-bound project/shot command

**Files:**
- Modify: `tools/wan22_longform/src/wan22_longform/cli.py`
- Modify: `tools/wan22_longform/src/wan22_longform/project.py`
- Modify: `tools/wan22_longform/tests/test_cli_assembly_records.py`
- Modify: `tools/wan22_longform/tests/test_project_state.py`

**Interfaces:**
- Repeatable `--approve-no-trim-boundary INDEX NOTE` exists only on `assemble-project` and `assemble-shot`.
- Parsed approvals have positive unique indexes and stripped nonempty notes.
- `assembly.json.requested.boundary_approvals` is re-read after accepted source rehashing.

- [ ] Write failing tests that assert an approved request, plan, and boundary decision log contain the same index/note and source-boundary evidence.
- [ ] Write a failing integrity test that changes serialized plan approval evidence and gets `integrity_failed` from `status` and `resume`.
- [ ] Run `C:\Program Files\Python311\python.exe -m unittest tools.wan22_longform.tests.test_cli_assembly_records tools.wan22_longform.tests.test_project_state`; expect red.
- [ ] Add the parser option to project/shot commands only. Persist a canonical list of approvals before the record exists; re-read that list from `assembly.json` after `verify_assembly_record_inputs()`; pass only record-derived approvals to `_plan_assembly()`.
- [ ] Serialize effective approvals with 1-based boundary index, note, left/right diagnostic hashes, and source attempt/output identity in `assembly-plan.json`; cross-check it against `assembly.json` in `project.py` integrity verification.
- [ ] Re-run focused tests; expect green.
- [ ] Commit with `git add tools/wan22_longform/src/wan22_longform/cli.py tools/wan22_longform/src/wan22_longform/project.py tools/wan22_longform/tests/test_cli_assembly_records.py tools/wan22_longform/tests/test_project_state.py` and `git commit -m "Record reviewed boundary approvals"`.

### Task 3: Operator docs, static gates, and live assembly

**Files:**
- Modify: `tools/wan22_longform/README.md`
- Modify: `tools/wan22_longform/CHANGELOG.md`
- Modify: `tools/wan22_longform/src/wan22_longform/frames.py`
- Modify: `tools/wan22_longform/tests/test_frame_boundaries.py`

- [ ] Add the documented command `assemble-project project.yaml --qc-approved --approve-no-trim-boundary 1 "Reviewed at 200%; preserve both frames."`, explaining this creates a new record, preserves both frames, and cannot authorize trimming.
- [ ] Run static gates: `C:\Program Files\Python311\python.exe -m unittest discover -s tools\wan22_longform\tests -p "test_*.py"`; `ruff check tools/wan22_longform/src tools/wan22_longform/tests`; `git diff --check`.
- [ ] Assemble the synthetic direct project with one reviewed-boundary approval and the synthetic FLF project with two approvals, each into a new output directory. Inspect frame counts and `status` integrity payloads.
- [ ] Commit docs, contact-sheet repair, and plan with `git add tools/wan22_longform/README.md tools/wan22_longform/CHANGELOG.md tools/wan22_longform/src/wan22_longform/frames.py tools/wan22_longform/tests/test_frame_boundaries.py docs/superpowers/plans/2026-07-19-reviewed-boundary-assembly.md` and `git commit -m "Document reviewed boundary assembly"`.

## Execution Handoff

The authorized task will execute this plan inline using `superpowers:executing-plans`, retaining test-first checkpoints before final integration review.
