# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-11 04:16 UTC

This document is the active GPU evidence plan for the temporal-grounding
paper. It contains only unfinished training and closed-loop evaluation jobs,
their dependencies, and the gates that determine later GPU work. Completed
CPU audits, source preparation, admission manifests, writing, figures, builds,
and scheduler telemetry are not active checklist items.

For execution dependencies and stop rules, this TODO supersedes the older task
ordering in `PAPER_REPLAN_TEMPORAL_GROUNDING_2026-08-07.md`; that document
remains the manuscript-level narrative plan.

No checkbox authorizes manual execution. The existing resource-aware scheduler
is the sole execution owner. Do not launch, stop, restart, reprioritize, or
replace a job outside that scheduler. On 2026-08-07 the operator authorized the
minimal scheduler change required to register the 16 already-frozen first-wave
TG1A/TG1B/TG2 jobs below. The authorization does not permit changes to source,
data, recipes, interventions, dependencies, gates, or result selection.

## 0. Current status override (2026-08-10)

This section supersedes stale live-state wording and checkboxes later in this
document. Historical job IDs and operational incidents remain below as an
audit trail; they are not the current execution plan.

### TG2 parent matrix: complete checkpoints, rejected comparison

All nine original TG2 training cells reached durable step-20,000 artifacts and
all nine location-aware materializers completed. The strict v4 joint integrity
worker then rejected the matrix because `exact_rank_data_order_within_seed`
failed for every seed: the four-rank ordered episode/frame hashes differ across
arms. The frozen loader used eight workers per rank with `in_order=false`, so
worker-completion timing changed the consumed order even under a shared seed.

This is a scientific protocol failure, not a missing-checkpoint failure. The
original nine checkpoints are retained as audit artifacts, but all original
TG2-E01--E09 evaluations are retired and must not run. They cannot establish a
matched comparison.

The scheduler-owned recovery probe `t-20260810090151-5ph8w` completed on East.
Two independent four-rank launches, each consuming 256 microbatches and 4,096
samples per rank with `in_order=true`, matched exactly on every rank. Canonical
evidence is
`logs/temporal_grounding/tg2/data_order_recovery_probe_v1/matched.json`.
This validates the deterministic recovery path but is not policy evidence.

### TG2R recovery matrix: active highest-priority training wave

The versioned recovery contract is
`lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_v1.json`.
It preserves all original arms, seeds, initialization, target routes, batch
128, four-GPU world size, eight workers per rank, 20,000 updates, H=E=50, and
fixed-final-checkpoint selection. Its only training change is
`datasets.vla_data.in_order: false -> true`.

All nine cells use the same detached source identity (`outer 11fb843`, LaWAM
`71803a3`) to eliminate source drift. North uses the atomically staged source;
the two East cells use a detached worktree at the same commits, the same frozen
recovery payload, and the byte-verified Transformers 5.2 runtime overlay. The
East entrypoint repeats both Git checks, all overlay hashes, the parent TG2
bundle verifier, the TG2R verifier, and the model/data hashes before training.
The nine independent four-GPU jobs may run concurrently. Backup-profile
submission is enabled with an eight-GPU identity-wide cap. At the 04:16 UTC
audit, the primary identity uses 4/25 GPUs, the backup identity uses 0/8 GPUs,
and East is idle at 0/8 GPUs. The superseded backup seed-1002 request terminated
`Failed` without becoming admissible. No TG2R
evaluation is admissible until all nine cells
pass a new joint exact-order integrity gate. Completion discovery is
location-aware: both East and North final-model paths are checked, and an East
parent directly satisfies its materializer dependency without a redundant
North transfer. The post-training v2 amendment also makes the joint integrity
gate location-aware without changing any scientific check: complete North
staging takes precedence, partial staging is rejected, and canonical shared
sidecars are accepted only when no staging directory exists, as required for
East-trained cells. A post-training v3 amendment handles the instrumentation
defect described below without changing a checkpoint, data order, or scientific
gate.

- [x] **TG2R-S0 [COMPLETE]** Staged and byte-verified the recovery payload on
  North at 01:17 UTC; both pinned Git identities and all four payload hashes
  passed.
- [x] **TG2R-T01 [COMPLETE: `t-20260810091838-d5ds7`, backup]** `future_off`,
  seed 1000, North 4 GPU; durable completion at 01:39 UTC and verified
  materialization at 02:49 UTC.
- [x] **TG2R-T02 [COMPLETE: `t-20260810091842-8p7bt`, backup]** `future_off`,
  seed 1001, North 4 GPU; durable completion at 01:45 UTC and verified
  materialization at 03:59 UTC.
- [ ] **TG2R-T03 [RUNNING: `t-20260811075607-jqpjx`, primary]** `future_off`,
  seed 1002, North 4 GPU. The operator explicitly authorized a primary-identity
  duplicate. The scheduler staged and hash-verified the amendment, detached the
  unkillable backup Queueing attempt `t-20260810091846-g8fpd`, and dispatched
  the primary job at 23:56 UTC. The formal checkpoint and audit sidecars use
  unique `primarydup` names; the old backup attempt is excluded from completion,
  materialization, integrity, and evaluation provenance. At 04:16 UTC the
  formal run is healthy at step 7,477/20,000 and 2.05 seconds/step, ETA 7.13
  hours; the detached backup request has terminated `Failed`.
- [x] **TG2R-T04 [COMPLETE: `t-20260810091825-6cgzh`, primary]**
  `fixed_endpoint`, seed 1000, North 4 GPU; durable completion at 14:08 UTC.
- [x] **TG2R-T05 [COMPLETE: `t-20260810091829-vnvpv`, backup]**
  `fixed_endpoint`, seed 1001, North 4 GPU; durable completion at 14:07 UTC.
- [x] **TG2R-T06 [COMPLETE: `t-20260810091834-hfkvq`, backup]**
  `fixed_endpoint`, seed 1002, North 4 GPU; durable completion at 14:18 UTC.
- [x] **TG2R-T07 [COMPLETE: `t-20260810102331-ktjk6`, primary]**
  `raw_milestone`, seed 1000, East 4 GPU; North attempt
  `t-20260810101504-w2mj8` was stopped before execution. The East replacement
  completed durably at 15:22 UTC and its location-aware materializer completed
  without a redundant transfer.
- [x] **TG2R-T08 [COMPLETE: `t-20260810102335-57b27`, primary]**
  `raw_milestone`, seed 1001, East 4 GPU; North attempt
  `t-20260810091854-v7vbr` was stopped before execution. The East replacement
  completed durably at 15:19 UTC and its location-aware materializer completed
  without a redundant transfer.
- [x] **TG2R-T09 [COMPLETE: `t-20260810091945-2h6rw`, primary]**
  `raw_milestone`, seed 1002, North 4 GPU; durable completion at 17:36 UTC. Its
  23.49 GB serialized full-state transfer and v3 sidecar verification completed
  with `rc=0` at 19:18 UTC.
- [ ] **TG2R-I1 [ADMITTED; BLOCKED by T01--T09]** Run nine serialized,
  full-state, hash-verified North-to-East materializers, then require exact
  initialization and rank-order equality within each seed plus distinct order
  across seeds. Eight materializers are complete: all fixed-endpoint and
  raw-milestone rows plus `future_off` seeds 1000--1001. Both newly materialized
  checkpoints have 7.17 GB final models, 9.13 GB optimizer states, and complete
  initialization/four-rank order sidecars. The gate explicitly checks
  `in_order=true` and eight workers in every persisted full config.
- [ ] **TG2R-E01--E09 [ADMITTED; BLOCKED by I1]** Run the unchanged frozen
  paired evaluation protocol on East only after joint integrity acceptance.

The 04:19 UTC downstream readiness audit rebuilt all ten scheduler tasks from
their frozen manifests. The East one-GPU integrity worker has no missing input
or hash failure. All nine East four-GPU evaluations likewise have no hash
failure; their only missing ready file is the prespecified
`temporal_grounding_tg2r_training_integrity.ok` gate marker. Thus no source,
YAML, renderer, or runtime repair is pending behind the final training and
materialization dependency.

The first materialization attempts for the three completed fixed-endpoint rows
all failed closed and exhausted their three local retries. The frozen TG2R
launcher exports `TG2R_ARM`, while the inherited audit writer reads `TG2_ARM`,
so the redundant `arm` field is null in the initialization and four rank-order
sidecars. Seed, route, rank, world size, counts, and ordered-sample digests are
present. Post-training amendment v3 preserves every raw sidecar byte and its
SHA-256, rejects any non-null arm mismatch or independent-field mismatch, and
sets the prespecified arm only in a temporary normalized integrity overlay.
It changes no training tensor, checkpoint, recipe, target route, data order,
evaluation, or acceptance criterion. The scheduler re-armed all exhausted
materializers after commit `84510b76`. All three completed fixed-endpoint
materializations then passed v3 validation: each report preserves the raw
initialization and four rank-file hashes, records arm recovery on all four
ranks, and leaves the raw files byte-identical with `arm:null`.

A read-only 19:22 UTC preflight found that the two East raw-milestone
initialization records and all eight rank-order records are root-owned mode
`0600`. The previously configured local zero-GPU integrity worker would
therefore fail after all dependencies completed. Post-training amendment v4
moves only the unchanged v3 verifier into a one-GPU `Robot-East-H20` platform
worker, whose root execution context can read those canonical records. Its
marker, all integrity checks, and every downstream evaluation dependency remain
unchanged; focused and full tests pass (`219 passed`).

### TG1 current incomplete evidence

TG1A has 98 valid frozen-manifest summaries in total, but no condition is a
complete 24-cell matrix. Normal is 20/24, null is 19/24, persistence is 21/24,
and shuffled is 0/24 because it remains dependent on complete normal capture.
The three platform jobs ended `Failed` after producing partial valid cells;
partial success rates are not interpretable. Each East candidate has exhausted
the frozen `max_failures=1`, and shuffled remains blocked by incomplete normal
capture. East idleness therefore does not authorize another attempt. TG1B remains disabled pending the
fixed-scene validity protocol decision: `future_off,E=36` is 20/24 and
`local_wm,E=50` is 18/24; the other two cells are incomplete. The unactivated
retry-cap amendment remains audit-only.

Task_N remains excluded by operator instruction. TG2R training and recovery of
the incomplete TG1 evidence are the active local-TODO priorities; TG3 and all
claim-expanding branches remain gated by the unchanged scientific criteria.

## 1. Scientific question and evidence boundary

The paper asks:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

The completed local audit establishes three descriptive facts:

- The released RoboTwin LaWAM checkpoint has `H=E=36` under its frozen 30 Hz,
  1.2 s inference contract. The current two-frame loader implies endpoint
  offsets `[0,35]`, but the original training data and exact training source
  are not local.
- The historical local all6 matrix has a 50-action training window but executes
  36 actions per query (`H=50,E=36`). Its fixed training target lies beyond the
  historical executed prefix.
- Only 19.35% of the 420,238 frozen milestone pairs lie within the 50-action
  training window, and 12.50% lie within the 36-action executed prefix.

These facts do not establish control utility, correct-content use, or a causal
temporal-grounding mechanism. Canonical local evidence is in
`lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_local_audit_v1.json` and
Section 41 of
`lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`.

## 2. Status vocabulary

- **READY:** its frozen admission bundle passes verification and no scientific
  dependency remains; the scheduler may admit it when policy and resources
  allow.
- **BLOCKED:** its immutable upstream artifact or gate does not yet exist. Do
  not submit it early.
- **CONDITIONAL:** it is not an active job. Create its admission bundle only
  after the named result gate passes.
- **COMPLETE:** all required cells and artifact checks passed. A wrapper exit
  code alone cannot establish completion.

Checkboxes below correspond one-to-one with GPU jobs or closed-loop evaluation
cells. A checked item therefore means canonical result completion, not script,
unit-test, dry-run, or admission readiness.

## 3. Completed prerequisites; not GPU progress

The following immutable admission bundles are frozen and were reverified on
2026-08-07 against the current outer HEAD:

| Bundle | Manifest | Verified files | Status |
|---|---|---:|---|
| TG1A | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json` | 14 | Passed |
| TG1B | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1b_admission_v1.json` | 11 | Passed |
| TG2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_admission_v1.json` | 23 | Passed |
| TG2 North staging | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_north_staging_amendment_v1.json` | 23 | Passed in detached clean worktree |
| TG2 seed independence | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_seed_independence_amendment_v1.json` | Post-training audit | Admitted; awaits nine data-order records |
| TG2 post-training pipeline | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_posttraining_pipeline_v1.json` | Runtime-only | Admitted; nine materializers, joint gate, and nine evals registered |
| TG2 post-training pipeline v2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_posttraining_pipeline_v2.json` | Runtime-only | Passed; strict sidecar validation, North sidecar staging, East integrity worker, and exact marker contract |
| TG2R post-training v2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v2.json` | Runtime-only | Passed; location-aware sidecar resolution with strict staging precedence and unchanged integrity/eval protocol |
| TG2R post-training v3 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v3.json` | Runtime-only | Passed locally; preserves raw sidecars and recovers only null redundant arm metadata in an audited temporary overlay |
| TG2R post-training v4 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v4.json` | Runtime-only | Passed locally; runs the unchanged v3 integrity verifier in the East root context required to read canonical mode-0600 sidecars |
| TG2R post-training v5 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v5.json` | Runtime-only | Passed locally; adds explicit tagged-source and tagged-sidecar selection so the authorized primary seed-1002 duplicate cannot mix with the detached backup attempt |
| TG2R seed-1002 primary duplicate | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2r_future_off_seed1002_primary_duplicate_v1.json` | Operational-only | Operator-authorized; preserves the complete frozen scientific contract and changes only credential identity, output tag, and superseded-attempt provenance |
| Runtime v2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v2.json` | Runtime-only | Passed; API framework admission |
| Runtime v3 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v3.json` | Runtime-only | Passed; shared Git trust and North mount admission |
| Runtime v4 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v4.json` | Runtime-only | Passed; TG1 policy Python pinned and processor smoke verified |
| Runtime v5 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v5.json` | Runtime-only | Passed; cached North image and deployment timeout |
| Runtime v6 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v6.json` | Runtime-only | Superseded before step 0; Qwen3 package overlay validated |
| Runtime v7 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v7.json` | Runtime-only | Healthy jobs retained; Qwen3 padding bridge passed |
| Runtime v8 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v8.json` | Runtime-only | Active for pending cells; collision-safe arm/seed run timestamps |
| Runtime v9 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v9.json` | Runtime-only | Passed; TG1A metadata batch bridge and strict failed-v4 isolation |
| Runtime v10 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v10.json` | Runtime-only | Passed; exact authorized LaWAM descendant verification for TG1A and TG2 eval |

The bundles pin outer implementation commit `db88e943`, LaWAM commit
`71803a3`, inputs, checkpoints, scene identities, report schemas, analysis
commands, and stop rules. The TG2 North amendment pins the detached staging
commit and dry-run-valid request body. These records make jobs admissible; they
provide no rollout evidence.

At 18:53 UTC, seven TG2 jobs are Running: fixed-endpoint seeds 1000--1001 on
East; raw-milestone seeds 1001--1002 and all three future-off seeds on North.
All seven passed the Qwen3 unequal-length batch smoke and sustained optimization
beyond the first step with frozen global batch 128. North raw seeds 1001--1002
reached steps 7080 and 7114 at 2.29--2.33 s/step; East fixed seeds reached
steps 7120 and 7107 at 2.29--2.30 s/step; North future-off seeds 1000--1001
reached steps 7474 and 7457 at 2.04--2.06 s/step, and newly started seed 1002
reached step 516 at 2.02 s/step. The earlier isolated 2.93 s/step reading for
East fixed seed 1001 returned to its established range on consecutive
heartbeats and is not a sustained slowdown. DataLoader time remains about
0.04 s. The remaining two v8 cells, fixed-endpoint seed 1002 and raw-milestone
seed 1000, are Queueing on North under the backup profile. That profile reports
8/20 active GPUs but seven queued submissions, five of which are obsolete
attempts that the available identities cannot cancel. No final TG2
checkpoint exists yet. Seven detached backup-profile attempts remain visible
but are not experiment progress: two are stuck Deploying and five are Queueing.
The scheduler refuses
to count them as current work and retries conservative cleanup, but the platform
denies `StopJob`; their exact states are reported separately in the resource
scheduler snapshot. The five Queueing attempts remain schedulable when backup
quota is released, including obsolete attempts for all three not-yet-running
cells. They must be stopped through the platform console before that release to
avoid duplicate execution or output-directory races; API credentials available
to the scheduler cannot perform `StopJob` or `DeleteJob`.
At 18:31 UTC, a read-only control-plane check through the gsy development host
confirmed that its default identity can inspect the obsolete jobs and reports
creator `tianming.zhang`, but a targeted cancellation of obsolete Queueing job
`t-20260807223419-bbwfx` was denied by `StopCustomTask`. Thus the gsy identity
does not provide a hidden cleanup path either; the job remained Queueing and no
formal v8 or Running task was touched.
At 19:40 UTC, direct `GetJob` request-body audits for the two current Queueing
jobs, fixed-endpoint seed 1002 (`t-20260807223916-rv9gd`) and raw-milestone
seed 1000 (`t-20260807223926-56pwj`), matched the frozen North runtime-v8 YAML
Entrypoint exactly. Their framework, image, four-H20 flavor, 24-hour deadline,
arm, and seed fields also match. They remain valid queued requests; this audit
does not count as startup or training progress.
The prior primary `future_off/1002` attempt `t-20260808003004-r7g8d` moved from
Queueing to Deploying at 18:32 UTC but was immediately reclaimed because the
scheduler incorrectly included its queue residence in the 900-second deployment
timeout. It produced no training output. Replacement v8 job
`t-20260808023231-z5mn8` entered Running, passed all startup checks, and reached
step 54. The scheduler now records `deploying_started_at` on the actual
Queueing-to-Deploying transition, so deployment timeout excludes prior queue
time; focused and full scheduler tests cover both a fresh transition and a
genuinely stale deployment.
At step 200, the replacement reported total/flow loss `0.063772`, exactly zero
perceptual and distillation losses as required by the `future_off` route, and
data/model times of 0.041/2.024 seconds. This is optimization-health evidence,
not closed-loop policy evidence.
The scheduler now probes every visible superseded attempt every five minutes,
while retaining a 30-minute `StopJob` retry throttle and refusing to stop any
attempt already in a non-waiting state. This shortens detection latency without
increasing denied stop-call pressure or changing experiment execution.
At 18:57 UTC, a completion-capacity audit measured 9.7 TiB available on the
East vePFS and 9.0 TiB on the North GPFS mount
`/vePFS-North-E/vis_robot`, with 91% and 99% of inodes free respectively. A
historical complete LaWAM step-20000 state occupies about 16 GiB (7.30 GB model
plus 9.34 GB optimizer), so nine comparable final states require about 140 GiB.
Checkpoint finalization and North-to-East materialization therefore have ample
filesystem capacity. The 40 GB figure from `/vePFS-North-E` itself describes
the gsy host root filesystem, not the nested North GPFS mount used by jobs.
All active TG2 YAMLs set a 24-hour platform deadline; the current heartbeats
project less than 13 hours from container start through step 20000. Two prior
complete 16 GiB states wrote their model and optimizer files in 12.4--12.5
seconds, leaving more than 11 hours of deadline margin for finalization.
That filesystem-level capacity check did not inspect the nested vePFS Fileset
quota and was therefore insufficient. At 02:06 UTC on 2026-08-08,
`future_off` seeds 1000 and 1001 both reached step 20,000 but failed while
writing `steps_20000_state/pytorch_model.bin`: the enclosing filesystem still
had 8.8 TiB free, while `DescribeFilesets` reported `/vis_robot` exactly at its
30,000/30,000 GiB capacity limit. Neither run produced a durable final model,
and the scheduler correctly rejected both failed partial states. The primary
identity lacked `SetFilesetQuota`; the already-enabled backup identity raised
the limit minimally to 30,500 GiB, after which an 8 MiB write-and-remove probe
passed to refresh and verify the quota cache. The scheduler requeued runtime-v8
retries as `t-20260808100701-hq5cj` and `t-20260808100706-xtw8d`. This is an
operational storage recovery, not training or checkpoint evidence. The
credential-free audit is recorded in
`AUDIT_temporal_grounding_north_fileset_quota_2026-08-08.json`.
The first post-quota v8 attempts (`t-20260808100701-hq5cj` and
`t-20260808100706-xtw8d`) then exited before step 0 because the frozen
overwrite guard found the retained failed-output directories. Those directories
were atomically renamed with `.failed-final-save-<job_id>` suffixes so all
failure evidence remains available while the active exact-name glob is clear.
Touching the already-admitted runtime-v8 readiness marker rearmed the scheduler
without changing its content or hash. Replacement requests
`t-20260808103834-9zf8q` and `t-20260808103839-wdh2t` initially entered the
North queue and moved to Running after the two raw-milestone jobs released
eight GPUs. By 03:16 UTC they reached steps 96 and 65 at 2.08 and 2.05 s/step,
respectively. This confirms startup after the storage and overwrite-guard
recovery, but both remain uncompleted work.
At 03:15 UTC, four runtime-v7 TG2 cells completed cleanly after step 20,000:
fixed-endpoint seeds 1000--1001 on East (`t-20260807221602-j6sww` and
`t-20260807221607-bckk5`) and raw-milestone seeds 1001--1002 on North
(`t-20260807221612-kpqwj` and `t-20260807221617-7hcmw`). Each platform job
reached `Completed`, and the scheduler observed exactly one durable final model
at the job's execution location. The two East location-aware materializers were
satisfied without transfer. Both North materializers are running under the
admitted serialized, hash-verified, atomic transfer path; their completion is
still required by the joint integrity dependency. The freed East slots were
immediately assigned by the scheduler to runtime-v10 TG1A normal and null
evaluations (`t-20260808110854-xxb97` and `t-20260808111025-xbjcc`). These
training completions establish durable checkpoints only, not policy utility.
The North-to-East SSH transport measured 5.6 MiB/s on a read-only 128 MiB
probe, implying about 47 minutes per 15.5 GiB run and 5.5 hours for seven
serialized North runs. Local materializers have no process timeout, and their
incoming-tree hash verification remains atomic. Because one transient SSH
failure would previously exhaust a materializer, the scheduler-only retry cap
is now three with a five-minute cooldown. The two frozen transfer scripts retain
their post-training-v2 hashes; no checkpoint, dependency, integrity gate, or
evaluation protocol changed. A same-poll state transition is also guarded: when
an East parent becomes complete during platform monitoring, its North
materializer is satisfied before candidate selection instead of briefly
launching an unnecessary transfer. A finalization audit also found that the
TG2 completion glob names `final_model/pytorch_model.pt`, which becomes visible
while the multi-gigabyte `torch.save` is still writing it. The scheduler now
records that early artifact as progress but requires platform `Completed` or
`Success` before completing any of the nine TG2 training tasks; it can no longer
issue `StopJob` against an in-progress final save. This is scheduler lifecycle
protection only and does not change the frozen training or completion artifact.
Visible files from `Failed` or `Stopped` jobs are rejected rather than accepted
as complete. A successful platform exit also receives a five-minute metadata
visibility grace before a missing shared-filesystem artifact becomes a retry;
this prevents a clean run from being immediately exhausted or duplicated on a
second resource during propagation. All 162 scheduler tests pass, including
Running, successful-exit, delayed-visibility, grace-expiry, and
failed-finalization regression cases. The same successful-terminal provenance
check is enforced again during pending/restart artifact reconciliation, so a
partial file from a failed attempt cannot bypass the runtime check on the next
state-machine phase. Successful and failed reconciliation cases bring the full
scheduler suite to 164 passing tests.
The first live North materialization refined that planning estimate. The
raw-milestone seed-1001 source tree is 23,489,222,006 bytes, comprising about
6.7 GiB of final-model weights and 16 GiB of step-20,000 checkpoint state. Its
incoming tree grew at 2.15 MiB/s over a 60-second sample while seed 1002 waited
on the serialization lock. If sustained, this projects about 2.9 hours per
North run and roughly 20 hours for seven runs rather than the small-file
probe's optimistic estimate. This affects result availability only; the
admitted full-state, serialized, hash-verified transfer contract is unchanged.
That first-minute projection is not a stable throughput bound. At 04:04 UTC,
after the incoming tree had grown to 11.60 GB, a fresh 60-second byte sample
measured 5.668 MiB/s and projected about 0.56 hours for the remainder of the
seed-1001 tree. Treat this as a current-run ETA only, not completion evidence or
authorization to parallelize, omit files, or weaken verification.
The seed-1001 materializer subsequently finished with `rc=0` at 04:39 UTC, and
the scheduler accepted its atomic local artifact as `1/1` at 04:40 UTC. The
serialized seed-1002 transfer then acquired the lock; its incoming tree had
reached 17.55 GB at 05:31 UTC. Seed-1001 is therefore no longer an integrity
dependency, while seed-1002 remains incomplete. Neither materialization is a
new training or policy-utility result.
The seed-1002 raw-milestone materializer subsequently completed its full-state,
hash-verified atomic transfer with `rc=0` at 05:49 UTC, and the scheduler
accepted `local=1/1` at 05:51 UTC. North `future_off` seed 1002 then completed
step 20,000 and its durable final save at 06:01 UTC; its serialized North
materializer started immediately. The 22.4 GiB full-state transfer, source and
destination hash checks, sidecar validation, and atomic installation completed
with `rc=0` at 07:11 UTC; the scheduler accepted `local=1/1` in the same poll.
The `future_off` seed-1000 and seed-1001 retries concurrently reached steps
7009 and 6980 at 2.14 and 2.04 s/step, with estimated remaining times of 7.72
and 7.38 hours. Thus all completed raw-milestone North transfers and
future-off seed 1002 are local; the remaining integrity dependencies are the
two future-off retries plus fixed-endpoint seed 1002 and raw-milestone seed
1000. None of these transitions is closed-loop policy evidence.
At 07:21 UTC, a second read-only verification of the three user-readable North
sidecar bundles (`future_off` seed 1002 and `raw_milestone` seeds 1001--1002)
passed the frozen initialization route, four-rank data-order schema, and every
recorded SHA-256 comparison. The verifier's eight focused tests also pass.
Canonical East sidecars remain root-owned and are intentionally deferred to
the admitted one-GPU East integrity worker after all nine rows materialize;
local permission denial is not training completion or integrity failure. The
two future-off retries reached steps 7195 and 7163 at 2.04--2.08 s/step, with
current ETAs of about 7.3--7.4 hours. The watcher also sees a fixed-endpoint
seed-1002 training heartbeat from a detached superseded attempt, but the
scheduler has no admissible runtime-v8 binding for that output and correctly
keeps the formal T06 request Queueing. It cannot be counted, materialized, or
used to release the joint gate.
After the transfer marker was accepted, the scheduler was safely reloaded at
07:12 UTC to apply the operator's primary-first credential policy and prior
20-GPU primary-account limit. Two subsequent snapshots report primary `8/20`
GPUs and `2/25` submitted jobs; no platform job or TG1A rollout was interrupted.
At 03:45 UTC, the previously detached runtime-v7 fixed-endpoint seed-1002 job
`t-20260807223419-bbwfx` was Running at step 1256 while the current runtime-v8
job `t-20260807223916-rv9gd` remained Queueing. A request, source, and
initialization audit found identical scientific inputs and parameter-tree
hashes; v8 differs only by its collision-safe run timestamp. The delayed job
nevertheless remains inadmissible under the current v8 invariant because it
was queued when v8 was admitted, is recorded as superseded, and has no final
state or data-order sidecars. Its existing run directory will also trigger the
v8 overwrite guard, while the North materializer requires exactly one matching
directory. No automatic adoption, quarantine, or checkbox credit is
authorized. The exact evidence and decision boundary are recorded in
`AUDIT_temporal_grounding_live_blockers_2026-08-08.json`.
At 08:51 UTC, that separation remained necessary: the detached runtime-v7 job
had reached step 9299 at 2.26 s/step, while direct control-plane queries still
reported the formal runtime-v8 T06 job Queueing. The two formal future-off v8
retries reached steps 9817 and 9771 at 2.04--2.05 s/step. These are healthy
optimization heartbeats only. The runtime-v7 heartbeat remains excluded from
T06, materialization, and the nine-arm integrity gate, and no new backup-profile
submission was made.
The scheduler test suite now redirects `LOG_PATH` into each test's temporary
directory, preventing synthetic retry and materializer messages from entering
the live operational log during future validation runs. The 164-test suite
passes with the stopped scheduler's production log byte count and modification
time unchanged. A one-time cleanup removed 210 historical lines carrying the
three unambiguous synthetic task signatures and left the platform/task state
untouched.
At 19:27 UTC, a live initialization audit covered the five current North
jobs. Their initialization payload, parameter-tree, trainable-tree, and
optimizer-tree SHA-256 values are each identical across arms and seeds
(`26e1de2e...`, `142be83f...`, `14c7acfb...`, and `c1ce78a0...`), all report
zero optimizer-state entries before training, and every record carries the
prespecified seed and route. In particular, all three `future_off` records
enable only `LAWAM_FUTURE_OFF`, while both running `raw_milestone` records
require full target coverage and name the frozen milestone inputs. The two
current East fixed-endpoint jobs also created fresh initialization records
after startup and loaded the same pinned pretraining checkpoint, but those
records are mode 0600 under the platform root identity. Their payload contents
therefore remain deliberately unclaimed until the admitted root-run TG2
integrity worker reads them. Current optimization checkpoints at the same
audit were steps 8000/8000 for East fixed endpoint, 8400/8400/1400 for North
future off, and 7800/8000 for North raw milestone; all reported finite losses,
the expected route-specific auxiliary terms, approximately 0.036--0.041 s
data time, and 2.02--2.31 s model time. This is initialization and training
health evidence only, not a completed TG2 cell or downstream utility result.
Mutable resource counts and platform states remain authoritative only in
`logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`.

At 16:54 UTC, all six Running jobs crossed step 4000 and retained healthy
heartbeats. The frozen `save_interval=20000` intentionally produces no
intermediate step-4000 checkpoint, so resumability and final-checkpoint
integrity cannot be claimed before the fixed final step.

At 17:33 UTC, all six Running jobs crossed step 5000 (25%) with healthy
heartbeats and no non-finite loss report. The exact step-5000 total losses were
0.00524/0.00546 for North future-off seeds 1000--1001,
0.00935/0.01261 for North raw-milestone seeds 1001--1002, and
0.02683/0.02553 for East fixed-endpoint seeds 1000--1001. DataLoader time
remained 0.036--0.041 s and model time 2.03--2.29 s. These optimization losses
are health evidence only and do not determine task utility. The two healthy
East v7 jobs share the auxiliary timestamped `train_lawam_ddp` log directory,
but an artifact audit confirmed distinct seed-qualified checkpoint roots,
configs, statistics, and W&B directories. Thus their model outputs do not
collide; runtime v8 remains required for pending launches because a third
same-second v7 launch previously failed on the shared auxiliary config copy.
The four active North initialization sidecars were also copied read-only and
bound to their formal Job IDs and SHA-256 digests in
`AUDIT_temporal_grounding_tg2_initialization_snapshot_2026-08-07.md`, preserving
their pre-overwrite evidence while detached superseded jobs remain visible.

Runtime v3 attempts produced no summary and exposed one operational mismatch:
the policy server used Transformers 4.53.2 and loaded the frozen Qwen3 weights
through an incompatible tokenizer object. Runtime v4 selects the existing
LaWAM Transformers 5.2.0 environment and fails fast unless it obtains
`Qwen3VLProcessor` with a tokenizer. Failed v3 output roots were preserved
under `.failed_runtime_v3` suffixes; no episode result was reused.

TG2 runtime v5 fixed North image staging but exposed that the repository
environment's Transformers 4.53.2 maps the frozen Qwen3-VL checkpoint to an
incompatible Qwen2.5 processor. Runtime v6 loaded Qwen3 correctly and built
the full model, then failed on the first DataLoader batch because the frozen
`processor_kwargs={"padding": true}` API spelling was not forwarded by the
Qwen3 processor. Runtime v7 adds a hash-pinned compatibility bridge that only
forwards this already-requested padding flag; both 5.2 and 4.57 probes establish
that version rollback alone does not fix it. All v5/v6 attempts stopped before
optimizer step 0. Their output directories were audited and preserved under
`.runtime_v5_pre_step0_quarantine` or `.runtime_v6_pre_step0_quarantine`.
One v7 fixed-endpoint seed-1002 retry passed every runtime and input check but
failed before optimizer step 0 when concurrent jobs selected the same
second-resolution LaWAM log directory. Runtime v8 generates an ephemeral copy
of the unchanged runner and appends the frozen arm and seed to its timestamp;
this isolates operational log/checkpoint directories without changing any
training argument. Healthy Running v7 jobs were not restarted.

The v4 executions exposed two blockers, so failed cells must not be blindly
retried:

- TG1A normal, null, and persistence all reached the policy server, but every
  task failed at first inference because `LatentWorldPolicyInferExample`
  declares `temporal_grounding_context` while the runtime batch builder rejects
  that key. No summary was produced. Runtime v9 admits the reviewed one-key
  metadata allowlist repair, verifies the runner-to-backend context route, and
  permits a retry only after proving and archiving the complete zero-summary
  v4 schema failure. A pre-launch audit then found that the frozen verifier
  would reject this explicitly admitted descendant before execution. Runtime
  v10 verifies the complete base bundle plus the exact two-file v9 LaWAM diff
  and rejects any unlisted descendant change. The scheduler has released all
  four TG1A cells under v10; shuffled remains artifact-blocked until normal
  capture completes. The same amended verifier is already a dependency of the
  nine TG2 evaluations, preventing the identical post-training startup fault.
  A 16:57 UTC preflight reverified all 17 amended bundle files, validated the
  exact prior-failure archive conditions for normal, null, and persistence, and
  passed all 12 retry, bundle, analysis, and context-routing tests.
- TG1B `future_off,E=36` produced 20/24 summaries and `local_wm,E=50` produced
  18/24. Missing cells exhausted the frozen three attempts for fixed scene
  seeds that remained invalid. A completed CPU diagnosis shows all six stopping
  seeds were accepted 45--65 times in other completed runs, and the four shared
  failures are identical across the two TG1B arms. This is stochastic setup
  validity, not a permanently invalid scene identity, but increasing the
  frozen retry cap is still a recipe change and requires an explicit protocol
  decision. Replacing scenes remains forbidden.

The non-activating blocker audit recommends, but does not authorize, one
common `ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500` amendment for TG1A and TG1B. This
matches the repository's existing formal pi0.5 fixed-scene launchers while
preserving every scene identity. A clean amendment must archive incomplete
roots and rerun all conditions in both panels under the same cap; mixing old
partial cells, changing scenes, or amending only one arm remains forbidden.
The read-only machine audit
`AUDIT_temporal_grounding_retry_panel_2026-08-08.json` independently verifies
the exact ordered scene identities of every summary currently present, both
frozen runners' cap of three, all eight required rerun conditions, and
`activated=false`. All 65 summaries present across the eight roots at 09:20 UTC
passed exact ordered-scene verification. The auditor and focused tests pass
without modifying a job, result root, runner, readiness marker, or scheduler
state.
At 07:31 UTC, the scheduler completion gate was hardened so every TG1A, TG1B,
and TG2 evaluation must contain exactly 24 summary files and then pass the
frozen fixed-scene verifier. A mismatched manifest, missing scene identity,
duplicate scene, duplicate cell, or extra summary now leaves the task
incomplete instead of releasing its downstream gate. All 183 scheduler and
recommendation tests pass, and the live partial roots remain correctly
incomplete at 13/24 normal and 6/24 null. This is lifecycle validation only; it
does not change a runner, scene, retry cap, or scientific result.

## 4. Dependency graph and admission waves

```text
TG1A normal ──> verify captured features ──> TG1A shuffled ──> TG1A analysis
TG1A null ──────────────────────────────────────────────────┘
TG1A persistence ───────────────────────────────────────────┘

TG1B four independent cells ───────────────────────────────> TG1B analysis

TG2 nine training jobs ──> one nine-arm integrity gate
                       └──> nine paired evaluations ────────> TG2 analysis
                                                               │
                                                               ├─> stop
                                                               ├─> TG3
                                                               ├─> TG4
                                                               └─> TG5
```

The first scheduler-admissible pool contains 16 independent 4-GPU jobs:

- three TG1A evaluations: normal, null, and persistence;
- four TG1B cadence cells;
- nine TG2 training jobs.

Fixed-checkpoint evaluations have higher information-per-compute than
retraining and are appropriate East candidates whenever the scheduler's
existing policy selects among otherwise admissible work. TG2 training can use
East or the verified North staging candidate. The scheduler's live resource,
quota, and fairness policy remains authoritative.

The dependent pool contains ten 4-GPU evaluations: TG1A shuffled and nine TG2
checkpoint evaluations. Their dependencies are immutable and cannot be waived.

The frozen four-GPU evaluation launchers were also audited against completed
East H20 runs before admitting the dependent wave. Each launcher starts one
eval seed per GPU, with all four workers entering the sweep within about one
minute. The same all6 stack completed its `E=36` sweep in 2.655 hours and its
`E=50` sweep in 2.765 hours, including runtime setup and the frozen three
fixed-scene attempts. The 24-hour platform deadline therefore retains at least
an 8.68x wall-time margin without changing workers, episodes, seeds, or the
frozen evaluation recipe.

## 5. TG1A — released-checkpoint content use

Purpose: determine whether the released checkpoint uses correct predicted
future content at fixed weights. Use only checkpoint SHA-256
`a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb`,
the frozen six-task scene manifest, 30 Hz control, and `H=E=36`.

Admission source:
`lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json`.

### GPU cells

- [ ] **TG1A-E1 [RUNNING-RUNTIME-V10: `t-20260808110854-xxb97`; prior failed
  run: `t-20260807165006-b4pqr`]** Evaluate `normal`; 4 GPUs, 1,200 accepted
  episodes. All four eval seeds completed `beat_block_hammer`; eval seeds 0,
  1, and 2 also completed `blocks_ranking_size`; eval seed 3 stopped that task
  after scene seed 400038 remained invalid for all three frozen setup attempts,
  then completed `blocks_ranking_rgb`. Eval seeds 0, 1, and 2 subsequently
  completed the same RGB task, eval seed 3 completed `handover_block`, and eval
  seed 0 completed that handover cell at 07:27 UTC. Eval seed 2 then completed
  `stack_blocks_two` at 07:43 UTC, followed by eval seed 1 at 07:50 UTC, eval
  seed 3 at 07:53 UTC, and eval seed 0 at 08:16 UTC. All four stack-two cells
  are therefore complete. Eval seeds 2 and 1 completed `stack_blocks_three` at
  09:12 and 09:16 UTC, respectively, producing 19/24 summaries. Eval seed 0
  stopped that task after scene seed 100033 remained invalid for all three
  frozen setup attempts. Eval seeds 1 and 2 stopped their
  handover cells after exhausting the same three-attempt setup limit. The final
  exact-24-cell
  verifier will reject this run without the same explicitly admitted common
  retry amendment required by the null arm. This is runtime progress, not a
  valid partial cross-condition result.
- [ ] **TG1A-E2 [RUNNING-RUNTIME-V10: `t-20260808111025-xbjcc`; prior failed
  run: `t-20260807165010-gk4h7`]** Evaluate `null`; 4 GPUs, 1,200 accepted
  episodes. Eval seeds 0, 2, and 3 completed `beat_block_hammer` and subsequently
  completed `blocks_ranking_size`. Eval seed 1 stopped the first task after scene
  seed 200026 remained invalid for all three frozen setup attempts and also
  failed the size task, but completed `blocks_ranking_rgb` at 08:32 UTC. Eval
  seed 0 completed the same RGB task at 09:20 UTC, producing 8/24 summaries;
  the final exact-24-cell verifier will therefore reject this run
  unless an explicitly admitted protocol amendment resolves the shared
  stochastic-validity blocker. This is runtime progress, not a valid partial
  cross-condition comparison.
- [ ] **TG1A-E3 [READY-RUNTIME-V10; prior failed run: `t-20260807171443-psgh6`]**
  Evaluate `persistence`; 4 GPUs, 1,200 accepted episodes. Runtime input schema
  rejected the frozen intervention field; 0/24 summaries.
- [ ] **TG1A-E4 [BLOCKED by TG1A-E1 capture]** Verify the complete normal
  feature capture, then evaluate the frozen within-task different-episode
  `shuffled` mapping; 4 GPUs, 1,200 accepted episodes.

At 18:45 UTC, fresh `--check-only` retry preflights validated the isolated
failed-runtime archives for normal, null, and persistence. The amendment-aware
TG1A bundle verifier also passed all 17 pinned files under runtime v10. This
confirms launch readiness when East releases capacity; it is not rollout
progress and no checkbox is complete.

The live null failure is the same `(task, eval seed, scene seed)` triple as a
shared TG1B failure, while the concurrent normal arm accepted that scene and
completed the task. This independently supports the existing diagnosis of
stochastic simulator/setup validity rather than a method-specific inference
failure. It does not authorize raising the retry cap, replacing a scene,
dropping the failed cell, or interpreting the partial success rates.

Oracle is closed, not unfinished: no audited exact same-scene expert endpoint
feature mapping exists. Do not substitute a retrieved, cross-scene, or
success-conditioned target and call it oracle.

### Completion and claims

Run the frozen TG1A analysis only after all four cells contain exactly the same
1,200 scene identities.

- Correct-content use requires `normal - shuffled` hierarchical paired 95% CI
  lower bound `>0` and Holm-adjusted paired `p<0.05`.
- Route necessity requires the independent `normal - null` gate.
- Endpoint content beyond persistence requires the independent
  `normal - persistence` gate.
- Report every task. A positive macro cannot erase a task-level regression.

If the content gate fails, retain the null: the released system contains a
future route, but correct future content is not causally identified.

Expected result:
`lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg1a.json`.

## 6. TG1B — execution-cadence sensitivity

Purpose: measure whether the historical `H=50,E=36` contract affects the
seed-2027 local-WM checkpoint differently from its matched future-off
checkpoint. This is a cadence diagnostic, not a future-content intervention.

Admission source:
`lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1b_admission_v1.json`.

### GPU cells

- [ ] **TG1B-E1 [BLOCKED at 20/24: `t-20260807171446-sprsg`]** Evaluate
  `future_off`, `E=36`; 4 GPUs, 1,200 episodes. Four cells exhausted the
  fixed-scene validity retry limit.
- [ ] **TG1B-E2 [NOT RUN under v4]** Evaluate `future_off`, `E=50`; 4 GPUs,
  1,200 episodes. Do not launch until the fixed-scene blocker is resolved.
- [ ] **TG1B-E3 [NOT RUN under v4]** Evaluate `local_wm`, `E=36`; 4 GPUs,
  1,200 episodes. Do not launch until the fixed-scene blocker is resolved.
- [ ] **TG1B-E4 [BLOCKED at 18/24: `t-20260807173250-nfv8r`]** Evaluate
  `local_wm`, `E=50`; 4 GPUs, 1,200 episodes. Six cells exhausted the
  fixed-scene validity retry limit.

### Completion and claims

The primary contrast is the paired difference-in-differences

`(local_wm_E50 - local_wm_E36) - (future_off_E50 - future_off_E36)`.

A positive hierarchical 95% interval establishes checkpoint-specific cadence
sensitivity only. If both checkpoints change similarly, report a general
replanning-cadence effect. Neither outcome proves correct-content use or
explains the released LaWAM system.

Expected result:
`lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg1b.json`.

## 7. TG2 — execution-aligned matched training matrix

Purpose: test active future-objective utility and target-horizon effects after
removing the historical training/execution mismatch. Freeze `H=E=50`, final
step 20,000, seeds 1000--1002, global batch 128, and the admission bundle's
initialization, data, optimizer, capacity, target route, and scene manifest.

Admission sources:

- `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_admission_v1.json`;
- `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_north_staging_amendment_v1.json`
  when the scheduler selects North.

### What is separate and what is jointly trained

TG2 separates the three scientific conditions into distinct checkpoints. This
is required for a causal arm comparison: mixing `future_off`, `fixed_endpoint`,
and `raw_milestone` targets within one checkpoint would no longer identify
which target caused a policy difference. Within each active-target checkpoint,
however, training is already joint: one optimizer minimizes action-flow loss
plus weighted future-perceptual and latent-distillation losses while updating
their shared trainable trunk. `future_off` preserves the parameter and
trainable trees but removes future losses and future conditioning.

All arms start from the same released LaWAM pretraining package. TG2 therefore
tests the effect of **joint downstream fine-tuning from that initialization**;
it does not compare predictor-only pretraining followed by a frozen predictor
against end-to-end joint training, and it does not test a fixed-plus-milestone
target mixture. Those are distinct conditional questions in Section 9.

Seeds 1000--1002 are three stochastic replications of each condition, not
three modules or three sequential training stages. Within a seed, all three
arms receive an identical deterministic initialization payload and rank data
order. The released checkpoint fully determines the observed initialization
payload, so that payload may remain identical across seeds; independence must
instead be demonstrated by distinct rank data-order hashes across seeds.
Training seed is the top statistical unit because one rollout seed panel
measures evaluation noise, not variation in learned policies. Earlier
single-seed screens changed verdict after matched replications, so no method or
training-schedule claim may be based on seed 1000 alone.

### Training jobs

- [ ] **TG2-T01 [RUNNING-V8 RETRY: `t-20260808103834-9zf8q`, North;
  prior V7 reached step 20,000 but failed final save]**
  Train `future_off`, seed 1000; 4 GPUs.
- [ ] **TG2-T02 [RUNNING-V8 RETRY: `t-20260808103839-wdh2t`, North;
  prior V7 reached step 20,000 but failed final save]**
  Train `future_off`, seed 1001; 4 GPUs.
- [x] **TG2-T03 [COMPLETE-V8: `t-20260808023231-z5mn8`, North; final model 1/1;
  materialized]**
  Train `future_off`, seed 1002; 4 GPUs.
- [x] **TG2-T04 [COMPLETE-V7: `t-20260807221602-j6sww`, East; final model 1/1]**
  Train `fixed_endpoint`, seed 1000; 4 GPUs.
- [x] **TG2-T05 [COMPLETE-V7: `t-20260807221607-bckk5`, East; final model 1/1]**
  Train `fixed_endpoint`, seed 1001; 4 GPUs.
- [ ] **TG2-T06 [QUEUEING-V8: `t-20260807223916-rv9gd`, backup]**
  Train `fixed_endpoint`, seed 1002; 4 GPUs.
- [ ] **TG2-T07 [QUEUEING-V8: `t-20260807223926-56pwj`, backup]**
  Train `raw_milestone`, seed 1000; 4 GPUs.
- [x] **TG2-T08 [COMPLETE-V7: `t-20260807221612-kpqwj`, North; final model 1/1;
  materialized]**
  Train `raw_milestone`, seed 1001; 4 GPUs.
- [x] **TG2-T09 [COMPLETE-V7: `t-20260807221617-7hcmw`, North; final model 1/1;
  materialized]**
  Train `raw_milestone`, seed 1002; 4 GPUs.

Each row is one 4-GPU training job. Check a row only after its fixed step-20,000
checkpoint is durable. Losses and checkpoint existence are not policy evidence.

### Matrix integrity dependency

After all nine training rows complete, run the frozen nine-arm integrity audit
and the admitted seed-independence extension. Together they must pass parameter
tree, trainable tree, within-seed initialization payload, exact rank data order,
cross-seed order distinction, normalization, optimizer parameter/state, target
coverage, and final-checkpoint checks. These CPU audits are automatic
dependencies, not GPU TODO items.

The scheduler now owns this entire transition. A checkpoint completed on North
is copied through a hash-verified incoming directory and atomically installed
on East. Its root-owned initialization record and four rank-order records are
strictly validated at the North source, hash-copied into a user-readable East
staging root, and validated again. An East checkpoint requires no transfer; the
joint integrity audit runs as a one-GPU East platform task so it can read the
canonical root-owned East sidecars. A partial staged sidecar set is rejected,
transfers are serialized, and an existing incomplete checkpoint destination is
never overwritten. The North end-to-end materialization probe passed, including
the scheduler-compatible `_train_materialized.ok` marker. Only after all nine
location-aware materializers and both integrity audits complete can any TG2
evaluation enter resource recommendation.

The runtime-v10 TG2 evaluation preflight verifies all 26 amended bundle files;
the evaluation and integrity entrypoints pass shell validation, and the 13
sidecar, training-integrity, seed-independence, and analysis tests pass. This is
startup readiness only and does not relax the nine-arm integrity dependency.

### Evaluation jobs

All rows below are **BLOCKED** until the complete nine-arm integrity audit
passes. Each row is one 4-GPU, 1,200-episode evaluation on the same paired scene
manifest.

- [ ] **TG2-E01 [BLOCKED]** Evaluate `future_off`, seed 1000; 4 GPUs.
- [ ] **TG2-E02 [BLOCKED]** Evaluate `future_off`, seed 1001; 4 GPUs.
- [ ] **TG2-E03 [BLOCKED]** Evaluate `future_off`, seed 1002; 4 GPUs.
- [ ] **TG2-E04 [BLOCKED]** Evaluate `fixed_endpoint`, seed 1000; 4 GPUs.
- [ ] **TG2-E05 [BLOCKED]** Evaluate `fixed_endpoint`, seed 1001; 4 GPUs.
- [ ] **TG2-E06 [BLOCKED]** Evaluate `fixed_endpoint`, seed 1002; 4 GPUs.
- [ ] **TG2-E07 [BLOCKED]** Evaluate `raw_milestone`, seed 1000; 4 GPUs.
- [ ] **TG2-E08 [BLOCKED]** Evaluate `raw_milestone`, seed 1001; 4 GPUs.
- [ ] **TG2-E09 [BLOCKED]** Evaluate `raw_milestone`, seed 1002; 4 GPUs.

### Primary gates

Analyze only after all nine evaluations pass exact scene pairing.

1. Fixed-endpoint utility: the hierarchical 95% CI lower bound for
   `fixed_endpoint - future_off` must exceed zero.
2. Raw-milestone utility: report `raw_milestone - future_off` independently;
   do not infer it from the fixed-versus-raw comparison.
3. Target-horizon effect: the lower bound for
   `fixed_endpoint - raw_milestone` must exceed zero.
4. Task safety: no claimed winning arm may have a training-seed/task effect
   below -5 percentage points against its stated baseline.
5. Training seed is the top resampling unit. Publish the full seed-by-task
   matrix; a macro mean cannot hide a task regression.

Expected result:
`lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg2.json`.

**Analysis execution blocker:** the frozen analyzer currently computes
`tg3_authorized` from `fixed_endpoint - raw_milestone` alone and can therefore
authorize TG3 when fixed endpoint fails against `future_off`. The synthetic
counterexample and required protocol-preserving repair are recorded in
`AUDIT_temporal_grounding_analysis_gate_2026-08-07.md`. Do not execute the TG2
analysis command or admit TG3 until an explicitly authorized amendment repairs
this mismatch and registers the scheduler-owned final CPU analysis task.

## 8. Result-driven branch table

TG1A and TG1B do not gate TG2 completion. They constrain interpretation. TG2
selects later training work:

| Audited outcome | Next GPU work | Allowed interpretation |
|---|---|---|
| Both active targets fail against `future_off` | Stop TG3 and target engineering | Active future objectives lack matched downstream utility in this setting |
| Fixed beats `future_off`, but fixed and raw are unresolved | TG4 eligible; TG3 forbidden; TG5 may replicate fixed utility | Active fixed-target package utility, no horizon mechanism |
| Fixed beats both `future_off` and raw, with task safety | TG3 and TG4 eligible; TG5 waits for one selected contrast | Replicated target-horizon effect; mechanism still unproven |
| Raw beats `future_off` and fixed does not beat raw | TG4 eligible; TG3 temporal-repair branch forbidden; TG5 may replicate raw utility | Milestone utility without support for the proposed alignment explanation |
| Any positive macro violates task safety | No general improvement claim; inspect only prespecified heterogeneity | Task-dependent effect |

TG1A content-gate failure remains binding under every TG2 branch. TG1B cadence
sensitivity never substitutes for a content or target-utility gate.

## 9. Conditional GPU plans; do not admit yet

These are planning envelopes, not active checklists. Freeze new manifests only
after the named upstream gate passes.

### TG3 — temporal-grounding mechanism

Eligibility requires all three TG2 conditions: fixed beats future-off, fixed
beats raw milestone, and task safety passes. The minimum matched matrix is:

- `milestone_time`, `milestone_time_constant`, and `milestone_clipped`;
- seeds 1000--1002: nine 4-GPU training jobs;
- one joint integrity audit, followed by nine paired evaluation jobs;
- for an accepted arm, fixed-checkpoint content-shuffled and time-shuffled
  evaluations on the same scenes.

A repair claim requires a grounded arm to beat raw milestone with positive
95% CI lower bound, Holm-adjusted `p<0.05`, task safety, and recovery of at
least half the TG2 fixed-minus-raw mean gap. A temporal-use claim additionally
requires correct time-to-go to beat time-shuffled timing. If this gate fails,
stop temporal repair; do not add post-hoc horizons, gates, or selectors.

### TG4 — source of active future utility and training protocol

Eligibility requires TG1A content use or a task-safe TG2 active-target utility
gate. Reuse compatible TG2 `future_off` and accepted active checkpoints rather
than retraining them. Freeze only the missing clean-base, auxiliary-only,
conditioning-only, and parameter-matched-null arms at seeds 1000--1002.

Attribute pretraining only from `future_off - clean`, downstream shaping only
from `auxiliary_only - future_off`, and inference content only from
`conditioning_only - parameter_matched_null`. Each label needs its own
positive-lower-bound interval and task-safety check.

#### TG4A — joint versus staged predictor training

This branch is eligible only after one target has task-safe TG2 utility. Select
exactly that target before adding jobs; do not choose the target from schedule
outcomes. Compare three matched training protocols at seeds 1000--1002:

- end-to-end joint optimization of action, future-perceptual, and distillation
  losses;
- target-specific predictor pretraining followed by a frozen-predictor policy
  stage;
- the same predictor pretraining followed by joint unfreezing during the policy
  stage.

Add a compute- and data-exposure-matched staged null so an extra training stage
cannot be credited to future prediction. Every seed must receive its own
predictor pretraining trajectory; reusing one seed-1000 predictor across all
three policy seeds is not an independent end-to-end replication. Freeze equal
sample exposure, optimizer updates, initialization ancestry, checkpoint rule,
and paired scenes before admission. A schedule claim requires its hierarchical
95% interval against end-to-end joint training to exclude zero and task safety
to pass. If compute exposure cannot be matched, report a systems tradeoff, not
a causal schedule advantage.

#### TG4B — fixed-plus-milestone target mixture

This is a different question from staged training. It becomes eligible only if
TG2 shows prespecified task-level complementarity: fixed endpoint and raw
milestone must each beat `future_off` on at least one non-overlapping task
stratum without either failing the global task-safety gate. Otherwise stop; do
not mix two unsupported targets to search for a positive macro.

The minimum matrix is the better single active arm, a fixed-plus-milestone
mixture, and a parameter-matched duplicate-target control at seeds 1000--1002.
The mixture must beat the **better single arm**, not their average, with a
positive hierarchical 95% lower bound and task safety. Report every task and
include route-specific fixed-checkpoint null/shuffle interventions before
claiming that both target contents are used. Mixing losses, alternating target
samples, and adding parallel conditioning routes are different mechanisms;
freeze exactly one implementation before training and do not compare them
post hoc.

### TG5 — external replication

Eligibility requires one task-safe positive TG2 or TG3 contrast. Freeze exactly
one primary contrast before choosing outcomes:

- replicate the accepted grounded-versus-raw contrast if TG3 passes;
- otherwise replicate the accepted fixed-versus-future-off, fixed-versus-raw,
  or raw-versus-future-off contrast selected directly by TG2.

Use either the complete four-suite LIBERO protocol or a prespecified complete
second RoboTwin panel. Cross-benchmark wording requires the hierarchical 95%
interval to retain direction with every suite/task reported. Do not use a
selected positive subset, saturated suite, or single training seed.

## 10. Global stop and reporting rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, outcome weighting,
  oracle-transition, or failed helper jobs to search for a positive result.
- Partial rollouts, smoke tests, training losses, representation metrics,
  checkpoint existence, or unmatched evaluation seeds cannot pass a utility
  gate.
- A representation gain does not establish content use. A public system score
  does not identify its causal component. Cadence sensitivity does not prove
  future-content use.
- Preserve every task-level regression and the fixed -5-point task-safety
  threshold. Do not promote a macro-only improvement.
- Do not tune target horizon, task groups, seeds, checkpoint step, intervention
  mapping, or loss weight against closed-loop outcomes.
- Completed evidence moves to
  `lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`; this file
  should continue to contain only unfinished GPU evidence and active gates.
