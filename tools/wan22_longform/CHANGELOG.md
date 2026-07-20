# Changelog

## 0.1.1 - 2026-07-19

- Added strict v1 project-manifest validation, including workflow hashes and environment snapshots.
- Made continuous I2V handoffs consume only one accepted, hash-verified upstream tail and preserve the selected input provenance in every downstream attempt.
- Bound story FLF endpoints to accepted tail/head QC evidence; retained `base_source_image` only for explicitly labeled technical smoke runs.
- Made `render-shot` reject dependent continuations before it queues any segment and made strict seed offsets deterministic.

## 0.1.0 - 2026-07-19

- Added native Wan 2.2 I2V segment and FLF bridge UI/API workflows.
- Added local preflight, immutable attempt/QC records, safe assembly, and copy-only workflow deployment.
- Added a documented, identity-disabled sample project and prompt set.
- Kept LightX2V preview work intentionally unbuilt pending exact local template provenance.
