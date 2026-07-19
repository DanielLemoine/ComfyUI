# Wan2.2 Long-Form V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only, testable Wan2.2 I2V long-form render, QC, and assembly system that preserves official two-stage Wan topology and supports safe optional LoRA routing.

**Architecture:** Keep ComfyUI workflows responsible for one native segment or FLF bridge. A project-local Python package validates immutable YAML manifests, derives executable API graphs from verified title/type anchors, tracks attempts and QC decisions, and invokes one FFmpeg wrapper for deterministic post-processing. The package must keep optional LoRAs absent until a configured file exists.

**Tech Stack:** Python 3.11, standard library HTTP/subprocess/pathlib/dataclasses, PyYAML, local ComfyUI API, FFmpeg, `unittest`, JSON/YAML workflow files.

## Global Constraints

- Work only on `codex/wan22-longform-v1` in `D:\AI\ComfyUI-wan22-longform`; never stage or change the active `D:\AI\ComfyUI` checkout's unrelated work.
- Do not stop, restart, or queue GPU work on `http://127.0.0.1:8188` until `/queue` reports no running task.
- Capture local `object_info`, model inventory, custom-node revisions, and native template evidence before emitting executable workflow JSON.
- Preserve the official high-to-low sampling boundary; the low stage must receive the high latent and must not add fresh initial noise.
- `mystic` and `wan_general` are mutually exclusive low-noise alternatives. Identity applies to both experts when enabled. VBVR and motion are high-only; corrective is low-only.
- No placeholder filename may appear in an executable API workflow. Disabled or missing optional LoRAs do not produce nodes.
- No automatic model/LoRA download, external telemetry, overwrite of an attempt, or hidden preset escalation.
- UI workflow deployment into the runtime is a copy after validation, never a move or overwrite of existing user workflows.

---

### Task 1: Establish the project package and deterministic preflight

**Files:**
- Create: `tools/wan22_longform/pyproject.toml`
- Create: `tools/wan22_longform/requirements.txt`
- Create: `tools/wan22_longform/src/wan22_longform/__init__.py`
- Create: `tools/wan22_longform/src/wan22_longform/errors.py`
- Create: `tools/wan22_longform/src/wan22_longform/inventory.py`
- Create: `tools/wan22_longform/src/wan22_longform/cli.py`
- Create: `tools/wan22_longform/tests/test_inventory.py`
- Create: `tools/wan22_longform/tests/fixtures/object_info.json`

**Interfaces:**
- Produces `collect_preflight(comfy_root: Path, comfy_url: str | None, artifact_dir: Path) -> PreflightResult`.
- Produces `python -m wan22_longform.cli preflight --comfy-root ... --comfy-url ...`.
- Produces the six required files under `artifacts/preflight/`.

- [ ] **Step 1: Write the failing preflight test.**

```python
def test_collect_preflight_writes_required_evidence(tmp_path, object_info_fixture):
    result = collect_preflight(
        comfy_root=tmp_path / "ComfyUI",
        comfy_url=None,
        artifact_dir=tmp_path / "artifacts" / "preflight",
        object_info=object_info_fixture,
    )
    assert result.object_info_path.name == "object_info.json"
    assert {path.name for path in result.artifact_dir.iterdir()} >= {
        "environment.json", "model_inventory.json", "custom_nodes.json",
        "object_info.json", "official_template_inventory.md", "preflight_report.md",
    }
```

- [ ] **Step 2: Run the test and verify it fails because the package does not exist.**

Run: `python -m unittest tools/wan22_longform/tests/test_inventory.py -v`

Expected: import failure for `wan22_longform.inventory`.

- [ ] **Step 3: Implement the smallest preflight boundary.**

```python
@dataclass(frozen=True)
class PreflightResult:
    artifact_dir: Path
    object_info_path: Path
    active_workflow_dir: Path | None
    native_i2v_template: Path | None
    native_flf_template: Path | None

def collect_preflight(*, comfy_root: Path, comfy_url: str | None,
                      artifact_dir: Path, object_info: dict[str, object] | None = None) -> PreflightResult:
    """Write reproducible local evidence without downloading any artifact."""
```

Use `urllib.request` only for the local URL, collect system/Python/GPU/FFmpeg data without mutating the live server, list exact model files relative to configured paths, and write UTF-8 JSON/Markdown deterministically. Make unavailable template/schema state an explicit report finding rather than silently substituting a community graph.

- [ ] **Step 4: Re-run the focused test and inspect artifact names.**

Run: `python -m unittest tools/wan22_longform/tests/test_inventory.py -v`

Expected: `OK`; all six evidence files exist.

- [ ] **Step 5: Commit the foundation.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 preflight tooling"
```

### Task 2: Add strict configuration, model discovery, and LoRA policy validation

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/config.py`
- Create: `tools/wan22_longform/config/models.example.yaml`
- Create: `tools/wan22_longform/config/presets.yaml`
- Create: `tools/wan22_longform/config/denylist.yaml`
- Create: `tools/wan22_longform/tests/test_config.py`
- Create: `tools/wan22_longform/tests/test_lora_policy.py`

**Interfaces:**
- Produces `load_project(path: Path) -> ProjectConfig` and `resolve_preset(project: ProjectConfig, presets: PresetCatalog) -> ResolvedRenderConfig`.
- Produces `validate_lora_policy(config: ResolvedRenderConfig, files: ModelFiles) -> None`.
- `ResolvedRenderConfig` contains one `permissiveness_mode`, branch-specific optional LoRA settings, and exact high/low model names.

- [ ] **Step 1: Write policy tests first.**

```python
def test_permissiveness_modes_are_exclusive(model_files):
    config = resolved_config(permissiveness_mode="mystic", wan_general_enabled=True)
    with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
        validate_lora_policy(config, model_files)

def test_single_identity_routes_the_same_file_to_both_experts(model_files):
    config = resolved_config(identity_mode="single_both", identity_file="id.safetensors")
    validate_lora_policy(config, model_files)
    self.assertEqual(config.identity.high_file, "id.safetensors")
    self.assertEqual(config.identity.low_file, "id.safetensors")
```

- [ ] **Step 2: Run the policy tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_config.py tools/wan22_longform/tests/test_lora_policy.py -v`

Expected: import failure for `wan22_longform.config`.

- [ ] **Step 3: Implement immutable config resolution.**

```python
PERMISSIVENESS_MODES = frozenset({"none", "mystic", "wan_general"})

def validate_lora_policy(config: ResolvedRenderConfig, files: ModelFiles) -> None:
    if config.models.high == config.models.low:
        raise ConfigError("high and low model files must differ")
    if config.permissiveness.mode not in PERMISSIVENESS_MODES:
        raise ConfigError("unknown permissiveness mode")
    for slot in config.enabled_loras():
        if slot.matches_denylist(config.denylist) and not config.dangerous_override:
            raise ConfigError(f"body-emphasis LoRA rejected: {slot.file}")
        if not files.contains(slot.file):
            raise ConfigError(f"configured LoRA is missing: {slot.file}")
```

Store the preset result separately from the parsed source YAML. Populate `P0_IDENTITY_BASELINE` through `P4_GENERAL_FALLBACK` exactly as required, with no automatic escalation.

- [ ] **Step 4: Re-run tests and validate the example config.**

Run: `python -m unittest tools/wan22_longform/tests/test_config.py tools/wan22_longform/tests/test_lora_policy.py -v`

Expected: all tests pass, including deny-list rejection, split identity routing, missing-file failure, and high/low distinction.

- [ ] **Step 5: Commit policy support.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 configuration policy"
```

### Task 3: Capture and validate official templates, then build clean API graphs dynamically

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/workflow.py`
- Create: `tools/wan22_longform/workflows/ui/wan22_segment_i2v_native.json`
- Create: `tools/wan22_longform/workflows/ui/wan22_bridge_flf2v_native.json`
- Create: `tools/wan22_longform/workflows/api/wan22_segment_i2v_native_api.json`
- Create: `tools/wan22_longform/workflows/api/wan22_bridge_flf2v_native_api.json`
- Create: `tools/wan22_longform/tests/test_workflow_patch.py`
- Create: `tools/wan22_longform/tests/fixtures/native_segment_api.json`
- Create: `tools/wan22_longform/tests/fixtures/native_bridge_api.json`

**Interfaces:**
- Produces `find_unique_node(graph: ApiGraph, title: str, class_type: str) -> NodeRef`.
- Produces `build_api_graph(base_graph: ApiGraph, render: ResolvedRenderConfig) -> ApiGraph`.
- Produces `validate_two_stage_graph(graph: ApiGraph) -> None`.

- [ ] **Step 1: Write failing graph tests.**

```python
def test_disabled_loras_are_absent_from_executable_graph(self):
    graph = build_api_graph(load_fixture("native_segment_api.json"), base_render_config())
    self.assertNotIn("LORA_VBVR_HIGH", titles(graph))
    self.assertNotIn("LORA_IDENTITY_LOW", titles(graph))

def test_lookup_requires_exactly_one_title_and_type(self):
    with self.assertRaisesRegex(WorkflowError, "multiple matches"):
        find_unique_node(graph_with_duplicate_title(), "MODEL_HIGH", "UNETLoader")

def test_low_sampler_has_no_new_noise_and_uses_high_latent(self):
    graph = load_fixture("native_segment_api.json")
    validate_two_stage_graph(graph)
```

- [ ] **Step 2: Run graph tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_workflow_patch.py -v`

Expected: import failure for `wan22_longform.workflow`.

- [ ] **Step 3: Implement graph validation and dynamic insertion.**

```python
def find_unique_node(graph: ApiGraph, title: str, class_type: str) -> NodeRef:
    matches = [node for node in graph.values()
               if node["class_type"] == class_type and node.get("_meta", {}).get("title") == title]
    if len(matches) != 1:
        raise WorkflowError(f"expected one {class_type} titled {title}, found {len(matches)}")
    return NodeRef(matches[0])

def build_api_graph(base_graph: ApiGraph, render: ResolvedRenderConfig) -> ApiGraph:
    graph = deepcopy(base_graph)
    for slot in render.enabled_loras_in_chain_order():
        insert_model_only_lora(graph, slot)
    validate_two_stage_graph(graph)
    return graph
```

Adapt only captured native templates. Persist `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE` as canonical input titles and record `BRIDGE_START`/`BRIDGE_END` aliases in the patch map note. If capture shows unavailable native FLF2V topology, write the actionable preflight blocker and leave its workflow absent rather than fabricating it.

- [ ] **Step 4: Run graph tests and installed-schema validation.**

Run: `python -m unittest tools/wan22_longform/tests/test_workflow_patch.py -v`

Expected: clean optional graph, correct high/low routes, uniqueness failures on zero/multiple matches, and low sampler validation.

- [ ] **Step 5: Commit native workflow tooling.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 native workflow builder"
```

### Task 4: Implement immutable project state, provenance, and decisions

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/hashing.py`
- Create: `tools/wan22_longform/src/wan22_longform/metadata.py`
- Create: `tools/wan22_longform/src/wan22_longform/project.py`
- Create: `tools/wan22_longform/tests/test_project_state.py`

**Interfaces:**
- Produces `create_attempt(project: ProjectConfig, shot_id: str, segment_id: str) -> Attempt`.
- Produces `transition_attempt(attempt: Attempt, target: AttemptState, note: str | None) -> Attempt`.
- Produces `sha256_file(path: Path) -> str` and `write_metadata(attempt: Attempt, metadata: RenderMetadata) -> Path`.

- [ ] **Step 1: Write state tests.**

```python
def test_retry_creates_a_distinct_immutable_attempt(self):
    first = create_attempt(project, "S010", "S010_C001")
    retry = create_attempt(project, "S010", "S010_C001")
    self.assertNotEqual(first.path, retry.path)
    self.assertTrue((first.path / "request.json").parent.exists())

def test_resume_skips_accepted_attempt(self):
    accepted = transition_attempt(attempt, AttemptState.ACCEPTED, "approved")
    self.assertFalse(needs_render(accepted))
```

- [ ] **Step 2: Run state tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_project_state.py -v`

Expected: import failure for `wan22_longform.project`.

- [ ] **Step 3: Implement append-only attempt records.**

```python
class AttemptState(StrEnum):
    PLANNED = "planned"
    RENDERING = "rendering"
    RENDERED = "rendered"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRY_REQUESTED = "retry_requested"
    ASSEMBLING = "assembling"
    ASSEMBLED = "assembled"
    FINAL = "final"
```

Use a monotonically increasing attempt number plus UTC timestamp in each path. Snapshot the manifest, workflow API JSON, request payload, input/workflow/output hashes, rendered output metadata, selected continuation frame, operator note, and parent attempt without modifying a prior attempt.

- [ ] **Step 4: Re-run the test module.**

Run: `python -m unittest tools/wan22_longform/tests/test_project_state.py -v`

Expected: immutable retry and accepted-resume tests pass.

- [ ] **Step 5: Commit traceability support.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 attempt tracking"
```

### Task 5: Add local ComfyUI rendering, frame extraction, and QC artifacts

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/comfy_client.py`
- Create: `tools/wan22_longform/src/wan22_longform/render.py`
- Create: `tools/wan22_longform/src/wan22_longform/frames.py`
- Create: `tools/wan22_longform/src/wan22_longform/qc.py`
- Create: `tools/wan22_longform/tests/test_frame_boundaries.py`
- Create: `tools/wan22_longform/tests/test_render_client.py`

**Interfaces:**
- Produces `ComfyClient.submit(graph: ApiGraph) -> str`, `ComfyClient.wait(prompt_id: str) -> HistoryResult`, and `ComfyClient.upload_image(path: Path) -> str`.
- Produces `render_segment(project, shot_id, segment_id, client) -> Attempt`.
- Produces `extract_candidate_frames(video: Path, count: int, where: Literal["head", "tail"], destination: Path) -> list[Path]`.

- [ ] **Step 1: Write HTTP and frame tests with a fake local transport.**

```python
def test_render_initializes_qc_needs_review(fake_client, project):
    attempt = render_segment(project, "S010", "S010_C001", fake_client)
    self.assertEqual(read_qc(attempt.path / "qc.yaml")["status"], "needs_review")

def test_tail_selection_uses_length_minus_count(self):
    self.assertEqual(candidate_start_index(frame_count=81, count=5, where="tail"), 76)
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_render_client.py tools/wan22_longform/tests/test_frame_boundaries.py -v`

Expected: import failures for the render modules.

- [ ] **Step 3: Implement local-only clients and QC records.**

```python
class ComfyClient:
    def submit(self, graph: ApiGraph) -> str:
        return self._post_json("/prompt", {"prompt": graph})["prompt_id"]

def candidate_start_index(*, frame_count: int, count: int, where: str) -> int:
    if frame_count < count:
        raise FrameError("video is shorter than requested candidate count")
    return 0 if where == "head" else frame_count - count
```

Use only `127.0.0.1` or configured loopback URLs. Save every request before submission. Use FFmpeg after render when the captured native template lacks an installed batch-selector node. The renderer stops after `needs_review` unless explicit policy authorizes automatic continuation.

- [ ] **Step 4: Re-run tests.**

Run: `python -m unittest tools/wan22_longform/tests/test_render_client.py tools/wan22_longform/tests/test_frame_boundaries.py -v`

Expected: mocked submission/history, QC initialization, candidate index math, and contact-sheet planning all pass.

- [ ] **Step 5: Commit render/QC support.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 render and QC support"
```

### Task 6: Implement FFmpeg probing, boundary diagnostics, and assembly planning

**Files:**
- Create: `tools/wan22_longform/src/wan22_longform/ffmpeg.py`
- Create: `tools/wan22_longform/src/wan22_longform/assembly.py`
- Create: `tools/wan22_longform/tests/test_assembly_plan.py`
- Create: `tools/wan22_longform/tests/test_duration_math.py`

**Interfaces:**
- Produces `probe_media(path: Path) -> MediaSpec`, `duration_seconds(frame_count: int, fps: Fraction) -> Fraction`.
- Produces `compare_boundary(left: Path, right: Path) -> BoundaryDecision`.
- Produces `plan_assembly(inputs: list[MediaSpec], output: AssemblyTargets) -> AssemblyPlan`.

- [ ] **Step 1: Write assembly tests.**

```python
def test_exact_duplicate_is_trimmed_once(self):
    decision = compare_frame_hashes("same", "same", perceptual_distance=0)
    self.assertEqual(decision.trim_right_frames, 1)

def test_similar_frame_is_not_silently_trimmed(self):
    decision = compare_frame_hashes("different", "different", perceptual_distance=3)
    self.assertEqual(decision.trim_right_frames, 0)
    self.assertTrue(decision.requires_review)

def test_duration_is_correct_within_one_frame(self):
    self.assertEqual(duration_seconds(81, Fraction(16, 1)), Fraction(81, 16))
```

- [ ] **Step 2: Run tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_assembly_plan.py tools/wan22_longform/tests/test_duration_math.py -v`

Expected: import failure for `wan22_longform.assembly`.

- [ ] **Step 3: Implement one FFmpeg command boundary.**

```python
def run_ffmpeg(args: Sequence[str], *, ffmpeg: Path, cwd: Path | None = None) -> CompletedProcess[str]:
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-y", *args]
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)

def plan_assembly(inputs: list[MediaSpec], output: AssemblyTargets) -> AssemblyPlan:
    if not compatible_for_concat(inputs):
        return AssemblyPlan(normalize_first=True, operations=normalization_operations(inputs, output))
    return AssemblyPlan(normalize_first=False, operations=concat_operations(inputs, output))
```

Create review MP4 plus FFV1/ProRes editing master. Enforce that RIFE is a post-assembly optional operation only. Log all trim decisions and refuse blind concat on incompatible media.

- [ ] **Step 4: Re-run tests.**

Run: `python -m unittest tools/wan22_longform/tests/test_assembly_plan.py tools/wan22_longform/tests/test_duration_math.py -v`

Expected: exact-duplicate, near-duplicate, normalization refusal, duration, and RIFE-order tests pass.

- [ ] **Step 5: Commit assembly support.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 assembly planning"
```

### Task 7: Complete CLI, sample project, workflow deployment, and operator documentation

**Files:**
- Create: `tools/wan22_longform/README.md`
- Create: `tools/wan22_longform/CHANGELOG.md`
- Create: `tools/wan22_longform/projects/example/project.yaml`
- Create: `tools/wan22_longform/projects/example/prompts/global_positive.txt`
- Create: `tools/wan22_longform/projects/example/prompts/global_negative.txt`
- Create: `tools/wan22_longform/projects/example/prompts/shot_001.txt`
- Create: `tools/wan22_longform/projects/example/prompts/shot_002.txt`
- Modify: `tools/wan22_longform/src/wan22_longform/cli.py`
- Create: `tools/wan22_longform/tests/test_cli.py`
- Create: `tools/wan22_longform/workflows/ui/wan22_segment_preview_lightx2v.NOT_BUILT.md` when an exact compatible local template cannot be proven.

**Interfaces:**
- Supports `preflight`, `validate-project`, `render-segment`, `render-shot`, `qc-contact-sheet`, `accept`, `reject`, `retry`, `assemble-shot`, `assemble-project`, `status`, and `resume`.
- Supports `deploy-workflows --target <active-user-workflow-dir>` as an explicit copy-only command.

- [ ] **Step 1: Write CLI parsing and deployment tests.**

```python
def test_validate_project_does_not_mutate_source_manifest(tmp_path):
    source = write_example_manifest(tmp_path)
    before = source.read_bytes()
    self.assertEqual(main(["validate-project", str(source)]), 0)
    self.assertEqual(source.read_bytes(), before)

def test_deploy_rejects_existing_target_without_force(tmp_path):
    with self.assertRaisesRegex(DeploymentError, "already exists"):
        deploy_workflows(source_dir, tmp_path / "wan22_longform", force=False)
```

- [ ] **Step 2: Run the CLI tests and verify they fail.**

Run: `python -m unittest tools/wan22_longform/tests/test_cli.py -v`

Expected: missing command behavior or import failure.

- [ ] **Step 3: Implement explicit commands and concise documentation.**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
```

README coverage must include architecture, local setup, model placement, UI workflows, CLI examples, manifest reference, preset escalation, identity installation, cinematic/continuous guidance, QC rubric, troubleshooting, NLE/RIFE/Topaz division, extension matrix, limitations, and unverified assumptions. Mark initial release `0.1.0` in the changelog. The sample remains fully clothed, adult, neutral, and identity-disabled until an actual identity file is supplied.

- [ ] **Step 4: Re-run CLI tests and syntax checks.**

Run: `python -m unittest tools/wan22_longform/tests/test_cli.py -v`

Expected: all commands parse, source manifests remain unchanged, and deployment cannot silently overwrite a runtime workflow.

- [ ] **Step 5: Commit the operator surface.**

```powershell
git add tools/wan22_longform
git commit -m "Add Wan22 long-form CLI"
```

### Task 8: Run static validation, then perform gated runtime validation once the active job is idle

**Files:**
- Create: `tools/wan22_longform/artifacts/validation/static_validation.md`
- Create: `tools/wan22_longform/artifacts/smoke/README.md`
- Modify: `tools/wan22_longform/README.md`

**Interfaces:**
- Produces `validate-workflows` report that maps every emitted class to captured `object_info`.
- Produces smoke output paths and an explicit pending record when no identity LoRA exists.

- [ ] **Step 1: Write a static gate test.**

```python
def test_every_executable_workflow_class_exists_in_object_info(self):
    object_info = load_object_info(preflight_path)
    for graph_path in executable_workflows():
        self.assertTrue(set(class_types(graph_path)) <= set(object_info))
```

- [ ] **Step 2: Run the complete unit suite before live rendering.**

Run: `python -m unittest discover -s tools/wan22_longform/tests -v`

Expected: all environment-independent tests pass with no GPU use.

- [ ] **Step 3: Run static workflow validation and inspect the report.**

Run: `python -m wan22_longform.cli validate-workflows --preflight tools/wan22_longform/artifacts/preflight/object_info.json`

Expected: every executable class is present, every required title/type is unique, and both sampler invariants hold.

- [ ] **Step 4: Wait for `/queue` to be idle, then run the smallest base smoke render.**

Run: `python -m wan22_longform.cli render-segment tools/wan22_longform/projects/example/project.yaml --shot S010 --segment S010_C001 --frames 17 --width 480 --height 272`

Expected: one low-resolution base I2V attempt with optional LoRAs disabled, complete metadata, five head/tail candidates, contact sheet, and `needs_review` QC state.

- [ ] **Step 5: Exercise a two-segment and 33-frame FLF bridge only after the base smoke is accepted.**

Run: `python -m wan22_longform.cli render-shot tools/wan22_longform/projects/example/project.yaml --shot S010`

Expected: direct-cut and bridged assembly evidence; no one-minute or identity-ready claim unless their gates have actually passed.

- [ ] **Step 6: Commit validation evidence and push the branch explicitly.**

```powershell
git add tools/wan22_longform
git commit -m "Validate Wan22 long-form tooling"
git push origin codex/wan22-longform-v1:codex/wan22-longform-v1
```

## Plan Self-Review

- Each functional requirement has a task: discovery (1), policy (2), workflows (3), traceability (4), render/QC (5), assembly (6), operator surface (7), and verification (8).
- The plan never fabricates node schemas; Task 3 blocks executable workflow construction on captured local evidence.
- The plan preserves explicit artifacts, immutable attempts, exclusive permissiveness modes, both-branch identity routing, and post-QC RIFE ordering.
- No unbounded continuation loop, implicit overwrite, automatic download, or active-render interruption is included.
