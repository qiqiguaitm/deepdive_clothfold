# TG2 Initialization Sidecar Snapshot Audit

Captured: 2026-08-07 17:42 UTC

This audit preserves the initialization sidecars produced by the four active
North runtime-v7 TG2 jobs before any detached superseded attempt can overwrite
the fixed remote sidecar paths. Read-only copies and a machine-readable
manifest are stored under
`logs/resource_scheduler_local/temporal_grounding_tg2_initialization_snapshots/20260807T1742Z`.

| Arm / seed | Formal Job ID | Sidecar SHA-256 |
|---|---|---|
| `future_off/1000` | `t-20260807223242-bt4fv` | `a93415823a61c204aa26725b01311e34cffab4e498a512b27c147d4dabd572c9` |
| `future_off/1001` | `t-20260807223247-sjmmb` | `3eec2d08a6e806a8114227115f9308710388b8bcf2fa726251b6c3ec48528032` |
| `raw_milestone/1001` | `t-20260807221612-kpqwj` | `cd520a3cdcc4a855d45bd69c83d091329dbb02750a06de25c6bc94cd40621d70` |
| `raw_milestone/1002` | `t-20260807221617-7hcmw` | `5bf34bef71c24e85a1206b8cdfa77b4ec7d79ed7e93cd9af46084c4d4fa8d366` |

All four sidecars agree on:

- initialization payload: `26e1de2e990379d7c727588b32c8822989cc543dbad65a0a6112a323dc8559ff`;
- parameter tree: `142be83fcbb30e6befa9ac43a3508bcc5321be95d12508f8cf7ba7c40858902e`;
- trainable tree: `14c7acfbb64b4bbef8c2397a05a2654b59a71b7b1831c9377ad032e2e787014f`;
- optimizer tree: `c1ce78a0b11d92c83239aadf786cc48536be3958b2c8f29793fed96db4cd78fb`;
- `dual_route=false`.

The two future-off jobs have `lawam_future_off=true`, no milestone target, and
no full-coverage requirement. The two raw-milestone jobs have
`lawam_future_off=false`, the frozen raw milestone target and compact index,
and `require_full_target_coverage=true`.

The two active East fixed-endpoint sidecars remain root-owned mode `0600` on
the shared filesystem, so the development user cannot copy their contents.
Their distinct seed-qualified checkpoint roots, per-run configs, statistics,
and W&B directories were inspected separately at the step-5000 audit. The
joint post-training verifier remains authoritative for all nine completed
runs; this snapshot is preservation evidence, not a substitute for that gate.
