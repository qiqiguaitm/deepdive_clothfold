# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-07 15:58 UTC

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
| TG2 seed independence | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_seed_independence_amendment_v1.json` | Post-training audit | Admitted; awaits nine data-order records |
| TG2 post-training pipeline | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_posttraining_pipeline_v1.json` | Runtime-only | Admitted; nine materializers, joint gate, and nine evals registered |
| TG2 post-training pipeline v2 | `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_posttraining_pipeline_v2.json` | Runtime-only | Passed; strict sidecar validation, North sidecar staging, East integrity worker, and exact marker contract |
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

At 16:24 UTC, six TG2 jobs are Running: fixed-endpoint seeds 1000--1001 on
East; raw-milestone seeds 1001--1002 and future-off seeds 1000--1001 on North.
All six passed the Qwen3 unequal-length batch smoke and sustained optimization
beyond the first step with frozen global batch 128. North raw seeds 1001--1002
reached steps 3258 and 3272 at 2.29--2.31 s/step; East fixed seeds reached
steps 3263 and 3257 at 2.27--2.29 s/step; and North future-off seeds reached
steps 3175 and 3150 at 2.04--2.07 s/step. DataLoader time remains about 0.04 s,
with running-job ETAs of roughly 9.5--10.7 hours. The remaining three v8 cells
are submitted in the North backup profile and are Queueing because that profile
currently lacks sufficient personal GPU quota. No final TG2 checkpoint exists
yet. Six detached backup-profile attempts remain visible but are not experiment
progress: two are stuck Deploying and four are Queueing. The scheduler refuses
to count them as current work and retries conservative cleanup, but the platform
denies `StopJob`; their exact states are reported separately in the resource
scheduler snapshot.
Mutable resource counts and platform states remain authoritative only in
`logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`.

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
- TG1B `future_off,E=36` produced 20/24 summaries and `local_wm,E=50` produced
  18/24. Missing cells exhausted the frozen three attempts for fixed scene
  seeds that remained invalid. A completed CPU diagnosis shows all six stopping
  seeds were accepted 45--65 times in other completed runs, and the four shared
  failures are identical across the two TG1B arms. This is stochastic setup
  validity, not a permanently invalid scene identity, but increasing the
  frozen retry cap is still a recipe change and requires an explicit protocol
  decision. Replacing scenes remains forbidden.

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

- [ ] **TG1A-E1 [READY-RUNTIME-V10; prior failed run: `t-20260807165006-b4pqr`]**
  Evaluate `normal`; 4 GPUs, 1,200 accepted episodes. Runtime input schema
  rejected the frozen intervention field; 0/24 summaries.
- [ ] **TG1A-E2 [READY-RUNTIME-V10; prior failed run: `t-20260807165010-gk4h7`]**
  Evaluate `null`; 4 GPUs, 1,200 accepted episodes. Runtime input schema
  rejected the frozen intervention field; 0/24 summaries.
- [ ] **TG1A-E3 [READY-RUNTIME-V10; prior failed run: `t-20260807171443-psgh6`]**
  Evaluate `persistence`; 4 GPUs, 1,200 accepted episodes. Runtime input schema
  rejected the frozen intervention field; 0/24 summaries.
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

- [ ] **TG2-T01 [RUNNING-V7: `t-20260807223242-bt4fv`, North]**
  Train `future_off`, seed 1000; 4 GPUs.
- [ ] **TG2-T02 [RUNNING-V7: `t-20260807223247-sjmmb`, North]**
  Train `future_off`, seed 1001; 4 GPUs.
- [ ] **TG2-T03 [QUEUEING-V8: `t-20260807223921-4rhkv`, backup]**
  Train `future_off`, seed 1002; 4 GPUs.
- [ ] **TG2-T04 [RUNNING-V7: `t-20260807221602-j6sww`, East]**
  Train `fixed_endpoint`, seed 1000; 4 GPUs.
- [ ] **TG2-T05 [RUNNING-V7: `t-20260807221607-bckk5`, East]**
  Train `fixed_endpoint`, seed 1001; 4 GPUs.
- [ ] **TG2-T06 [QUEUEING-V8: `t-20260807223916-rv9gd`, backup]**
  Train `fixed_endpoint`, seed 1002; 4 GPUs.
- [ ] **TG2-T07 [QUEUEING-V8: `t-20260807223926-56pwj`, backup]**
  Train `raw_milestone`, seed 1000; 4 GPUs.
- [ ] **TG2-T08 [RUNNING-V7: `t-20260807221612-kpqwj`, North]**
  Train `raw_milestone`, seed 1001; 4 GPUs.
- [ ] **TG2-T09 [RUNNING-V7: `t-20260807221617-7hcmw`, North]**
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
