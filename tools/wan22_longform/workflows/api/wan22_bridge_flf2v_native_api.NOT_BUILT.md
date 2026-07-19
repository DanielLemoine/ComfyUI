# Wan2.2 native FLF API workflow — NOT_BUILT

No official API graph can be derived because the verified official template inventory contains no workflow with a `WanFirstLastFrameToVideo` node. `D:\AI\ComfyUI\blueprints\First-Last-Frame to Video.json` is an LTX workflow and is not a valid Wan source.

Actionable blocker: provide an official ComfyUI template under `blueprints`, `workflow_templates`, or `web/assets/workflow_templates` whose graph contains `WanFirstLastFrameToVideo`, then rerun preflight and derive the API map from that captured native topology. No community or fabricated graph may be substituted.

The future canonical API input titles are `BRIDGE_FIRST_IMAGE` and `BRIDGE_LAST_IMAGE`. `BRIDGE_START` and `BRIDGE_END` are documentation aliases only.
