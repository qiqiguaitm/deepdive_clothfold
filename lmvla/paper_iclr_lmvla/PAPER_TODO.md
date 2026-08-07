# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-07 08:53 UTC

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
| Runtime v2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v2.json` | Runtime-only | Passed; API framework admission |
| Runtime v3 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v3.json` | Runtime-only | Passed; shared Git trust and North mount admission |
| Runtime v4 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v4.json` | Runtime-only | Passed; TG1 policy Python pinned and processor smoke verified |

The bundles pin outer implementation commit `db88e943`, LaWAM commit
`71803a3`, inputs, checkpoints, scene identities, report schemas, analysis
commands, and stop rules. The TG2 North amendment pins the detached staging
commit and dry-run-valid request body. These records make jobs admissible; they
provide no rollout evidence.

At 08:53 UTC, TG1A normal and null are running on all 8 East GPUs. Both passed
the renderer, frozen-bundle, and pinned-policy-runtime checks and reached the
formal four-seed evaluator. All nine TG2 training jobs have platform IDs on
North: two primary-profile jobs are Deploying and seven jobs are Queueing. The
North queue reports GPU fragmentation for the five backup-profile jobs and
personal quota pressure for two primary-profile jobs. Mutable resource counts
remain authoritative only in `logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`.

Runtime v3 attempts produced no summary and exposed one operational mismatch:
the policy server used Transformers 4.53.2 and loaded the frozen Qwen3 weights
through an incompatible tokenizer object. Runtime v4 selects the existing
LaWAM Transformers 5.2.0 environment and fails fast unless it obtains
`Qwen3VLProcessor` with a tokenizer. Failed v3 output roots were preserved
under `.failed_runtime_v3` suffixes; no episode result was reused.

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

## 5. TG1A — released-checkpoint content use

Purpose: determine whether the released checkpoint uses correct predicted
future content at fixed weights. Use only checkpoint SHA-256
`a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb`,
the frozen six-task scene manifest, 30 Hz control, and `H=E=36`.

Admission source:
`lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json`.

### GPU cells

- [ ] **TG1A-E1 [RUNNING: `t-20260807165006-b4pqr`]** Evaluate `normal`;
  4 GPUs, 1,200 accepted episodes.
- [ ] **TG1A-E2 [RUNNING: `t-20260807165010-gk4h7`]** Evaluate `null`;
  4 GPUs, 1,200 accepted episodes.
- [ ] **TG1A-E3 [READY; waiting for East]** Evaluate `persistence`; 4 GPUs,
  1,200 accepted episodes.
- [ ] **TG1A-E4 [BLOCKED by TG1A-E1 capture]** Verify the complete normal
  feature capture, then evaluate the frozen within-task different-episode
  `shuffled` mapping; 4 GPUs, 1,200 accepted episodes.

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

- [ ] **TG1B-E1 [READY; waiting for East]** Evaluate `future_off`, `E=36`;
  4 GPUs, 1,200 episodes.
- [ ] **TG1B-E2 [READY; waiting for East]** Evaluate `future_off`, `E=50`;
  4 GPUs, 1,200 episodes.
- [ ] **TG1B-E3 [READY; waiting for East]** Evaluate `local_wm`, `E=36`;
  4 GPUs, 1,200 episodes.
- [ ] **TG1B-E4 [READY; waiting for East]** Evaluate `local_wm`, `E=50`;
  4 GPUs, 1,200 episodes.

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

### Training jobs

- [ ] **TG2-T01 [SUBMITTED-QUEUEING: `t-20260807163508-kspsv`, backup]**
  Train `future_off`, seed 1000; 4 GPUs.
- [ ] **TG2-T02 [SUBMITTED-QUEUEING: `t-20260807163513-q6x7f`, backup]**
  Train `future_off`, seed 1001; 4 GPUs.
- [ ] **TG2-T03 [SUBMITTED-QUEUEING: `t-20260807163518-c8vk6`, backup]**
  Train `future_off`, seed 1002; 4 GPUs.
- [ ] **TG2-T04 [SUBMITTED-DEPLOYING: `t-20260807163452-hj776`, primary]**
  Train `fixed_endpoint`, seed 1000; 4 GPUs.
- [ ] **TG2-T05 [SUBMITTED-DEPLOYING: `t-20260807163457-qrwfh`, primary]**
  Train `fixed_endpoint`, seed 1001; 4 GPUs.
- [ ] **TG2-T06 [SUBMITTED-QUEUEING: `t-20260807163503-5h92f`, backup]**
  Train `fixed_endpoint`, seed 1002; 4 GPUs.
- [ ] **TG2-T07 [SUBMITTED-QUEUEING: `t-20260807163641-rk49r`, backup]**
  Train `raw_milestone`, seed 1000; 4 GPUs.
- [ ] **TG2-T08 [SUBMITTED-QUEUEING: `t-20260807163646-dl67q`, primary]**
  Train `raw_milestone`, seed 1001; 4 GPUs.
- [ ] **TG2-T09 [SUBMITTED-QUEUEING: `t-20260807163652-4f72m`, primary]**
  Train `raw_milestone`, seed 1002; 4 GPUs.

Each row is one 4-GPU training job. Check a row only after its fixed step-20,000
checkpoint is durable. Losses and checkpoint existence are not policy evidence.

### Matrix integrity dependency

After all nine training rows complete, run the frozen nine-arm integrity audit.
It must jointly pass parameter tree, trainable tree, within-seed initialization
payload, exact rank data order, normalization, optimizer parameter/state,
target coverage, and final-checkpoint checks. This CPU audit is an automatic
dependency, not a GPU TODO item.

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

### TG4 — source of active future utility

Eligibility requires TG1A content use or a task-safe TG2 active-target utility
gate. Reuse compatible TG2 `future_off` and accepted active checkpoints rather
than retraining them. Freeze only the missing clean-base, auxiliary-only,
conditioning-only, and parameter-matched-null arms at seeds 1000--1002.

Attribute pretraining only from `future_off - clean`, downstream shaping only
from `auxiliary_only - future_off`, and inference content only from
`conditioning_only - parameter_matched_null`. Each label needs its own
positive-lower-bound interval and task-safety check.

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
