# Changelog

## 0.1.3 - 2026-07-19

- Made `status` and `resume` read-only integrity checks: source video hashes, serialized plan evidence, boundary decisions, and finalized outputs now fail closed as `integrity_failed` instead of appearing as clean final records.
- Bound the assembly record, plan, output-evidence file, boundary decisions, and final output hashes into append-only transition evidence.
- Made shot assembly consume its validated `assembly_order`, including declared story FLF bridges, and reject incomplete or ambiguous shot timelines.
- Bound manifest output declarations as safe record-local role/name templates and honored explicit `--rife-review-mp4` targets only with `--request-rife`.

## 0.1.2 - 2026-07-19

- Replaced source-attempt assembly states with separate immutable assembly records containing accepted source hashes, requested targets, plans, boundary evidence, output hashes, and recoverable success/failure decisions.
- Made `status` and `resume` report assembly-record state without automatically restarting FFmpeg work or re-rendering accepted segments.
- Made project and shot assembly validate the strict manifest before selecting sources, and made arbitrary-file `assemble` explicitly diagnostic-only.
- Rejected same-shot bridge/segment id collisions, required declared source/destination metadata and timeline adjacency for story FLF bridges, and prohibited technical-smoke bridges from production assembly order.

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
