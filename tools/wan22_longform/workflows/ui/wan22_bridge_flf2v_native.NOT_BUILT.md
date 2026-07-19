# Wan2.2 native FLF UI workflow — NOT_BUILT

The verified official template inventory contains no workflow with a `WanFirstLastFrameToVideo` node. `D:\AI\ComfyUI\blueprints\First-Last-Frame to Video.json` is an LTX workflow, not a native Wan workflow, so it cannot be adapted for this lane.

Actionable blocker: provide an official ComfyUI template under `blueprints`, `workflow_templates`, or `web/assets/workflow_templates` whose graph contains `WanFirstLastFrameToVideo`, then rerun preflight and capture that exact topology. No community or fabricated graph may be substituted.

When the native bridge is available, its canonical input titles are `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE`. `BRIDGE_START` and `BRIDGE_END` are documentation aliases only.
