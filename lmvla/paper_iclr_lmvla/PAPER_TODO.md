# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-11 11:38 UTC

This file contains only unfinished training/evaluation evidence and the gates
that control later GPU work. Completed evidence, rejected protocols, and
superseded execution history are archived in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`, Sections 41--46.

The resource-aware scheduler is the sole execution owner. A checkbox records
scientific completion; it does not authorize manual launch, stop, restart,
reprioritization, or replacement. Mutable job state is authoritative only in
`logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`.

## 1. Current evidence status

The paper asks:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

Three result boundaries remain binding:

- The temporal-contract audit establishes that the released LaWAM system is
  endpoint-aligned under the audited current loader, whereas historical local
  recurrence milestones are usually multi-chunk and lack time-to-go. This is
  descriptive timing evidence, not a control mechanism.
- The original TG2 matrix cannot answer the control question. All nine
  checkpoints completed, but exact per-rank data order differed across arms
  within every seed; its integrity gate rejected the comparison and all nine
  evaluations are retired.
- The deterministic recovery probe passed with `in_order=true`, but a loader
  reproducibility result is not policy evidence. No temporal-grounding utility,
  content-use, cadence, or target-horizon gate has passed.

## 2. Active TG2R recovery matrix

TG2R repeats `future_off`, `fixed_endpoint`, and `raw_milestone` at training
seeds 1000--1002. It preserves the original initialization, target routes,
data, global batch 128, four-GPU world size, eight workers per rank, 20,000
updates, `H=E=50`, and fixed-final-checkpoint selection. Its only scientific
training change is `datasets.vla_data.in_order: false -> true`.

Eight training cells and their location-aware materializers are complete and
archived in Section 44. Only the following execution chain remains:

- [x] **TG2R-T03 [COMPLETE: `t-20260811075607-jqpjx`]** Trained `future_off`,
  seed 1002 on four North GPUs under the operator-authorized primary-identity
  duplicate. The formal job completed at 11:28 UTC. The detached backup
  request and all of its output remain excluded from checkpoint, integrity,
  and evaluation provenance. The operator has separately authorized another
  primary-identity seed-1002 submission only if recovery is required; no
  additional training is needed while this completed artifact verifies.
- [ ] **TG2R-M03 [RUNNING locally]** Materialize the complete seed-1002
  full-state tree from North to East with source/destination SHA-256
  verification, including initialization and four rank-order sidecars. The v6
  atomic materializer started at 11:28 UTC with three parallel large-file
  streams; its incoming tree was 8.0 GiB at 11:38 UTC.
- [ ] **TG2R-I1 [BLOCKED by M03]** Run the joint nine-arm integrity gate in the
  admitted East root context. Require all nine successful terminal jobs,
  fixed step 20,000, complete optimizer state, exact initialization equality
  within seed, exact rank-order equality across arms within seed, distinct
  order across seeds, `in_order=true`, and eight workers in every full config.

No evaluation below is admissible before
`logs/resource_markers/temporal_grounding_tg2r_training_integrity.ok` exists.
Each evaluation uses four East GPUs, the unchanged six-task/four-evaluation-seed
scene manifest, and exactly 1,200 accepted episodes.

- [ ] **TG2R-E01 [BLOCKED by I1]** Evaluate `future_off`, seed 1000.
- [ ] **TG2R-E02 [BLOCKED by I1]** Evaluate `future_off`, seed 1001.
- [ ] **TG2R-E03 [BLOCKED by I1]** Evaluate `future_off`, seed 1002.
- [ ] **TG2R-E04 [BLOCKED by I1]** Evaluate `fixed_endpoint`, seed 1000.
- [ ] **TG2R-E05 [BLOCKED by I1]** Evaluate `fixed_endpoint`, seed 1001.
- [ ] **TG2R-E06 [BLOCKED by I1]** Evaluate `fixed_endpoint`, seed 1002.
- [ ] **TG2R-E07 [BLOCKED by I1]** Evaluate `raw_milestone`, seed 1000.
- [ ] **TG2R-E08 [BLOCKED by I1]** Evaluate `raw_milestone`, seed 1001.
- [ ] **TG2R-E09 [BLOCKED by I1]** Evaluate `raw_milestone`, seed 1002.

### TG2R analysis and primary gates

Analyze only after all nine evaluations pass exact scene pairing. Training seed
is the highest resampling unit; publish the full seed-by-task matrix.

1. **Fixed-endpoint utility:** the hierarchical 95% CI lower bound for
   `fixed_endpoint - future_off` must exceed zero.
2. **Raw-milestone utility:** test `raw_milestone - future_off` independently;
   do not infer it from fixed versus raw.
3. **Target-horizon effect:** the lower bound for
   `fixed_endpoint - raw_milestone` must exceed zero.
4. **Task safety:** no claimed winning arm may have a training-seed/task effect
   below -5 percentage points against its stated baseline.
5. **Reporting:** a macro mean cannot hide a task-level regression. Report
   uncertainty and all seed/task cells, including negative effects.

The expected canonical analysis output remains
`lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg2.json`. The rejected
parent matrix produced no such result and cannot contribute episodes.

## 3. Active TG1 common fixed-scene retry500 panel

The operator-authorized common amendment activated at 07:14:16 UTC. It changes
only `ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS: 3 -> 500` and applies to every TG1A and
TG1B condition. All five extant cap-3 result roots were recoverably renamed
with suffix `.pre_retry500_v1`; the three zero-cell roots were absent. Every
retry500 canonical result root was empty at activation. The root-owned legacy
normal feature capture remains in place as excluded provenance, while the new
normal run writes to the distinct, initially absent
`logs/tg1_retry500/predicted_endpoint_features` root. No old result or feature
cell is eligible for reuse.

Current scheduler-owned execution is:

- [ ] **TG1A-E1 [RUNNING: `t-20260811151421-rlt29`, 16/24]** Evaluate normal
  from an empty root under retry500 and produce a verified new feature
  capture.
- [ ] **TG1A-E2 [RUNNING: `t-20260811151424-z9hrr`, 8/24]** Evaluate null from
  an empty root under retry500.
- [ ] **TG1A-E3 [PENDING: East capacity]** Evaluate persistence from an empty
  root under retry500.
- [ ] **TG1A-E4 [BLOCKED by complete E1 capture]** Evaluate the frozen
  within-task different-episode shuffled mapping.
- [ ] **TG1B-E1 [PENDING: East capacity]** Evaluate future-off,
  `E=36`.
- [ ] **TG1B-E2 [PENDING: East capacity]** Evaluate future-off,
  `E=50`.
- [ ] **TG1B-E3 [PENDING: East capacity]** Evaluate local-WM,
  `E=36`.
- [ ] **TG1B-E4 [PENDING: East capacity]** Evaluate local-WM,
  `E=50`.

Partial retry500 cells remain operational telemetry only. No rate or contrast is
admissible until each required condition reaches exactly 24/24 summaries and
passes provenance verification.

### TG1 analysis gates

- Correct-content use requires TG1A `normal - shuffled` hierarchical paired
  95% CI lower bound `>0` and Holm-adjusted paired `p<0.05`.
- Route necessity requires the independent `normal - null` gate.
- Endpoint content beyond persistence requires the independent
  `normal - persistence` gate.
- TG1B's primary cadence contrast is
  `(local_wm_E50 - local_wm_E36) - (future_off_E50 - future_off_E36)`.
  A positive interval establishes checkpoint-specific cadence sensitivity
  only; it does not establish correct-content use.
- Every task must be reported. No partial panel or macro-only result may enter
  the manuscript.

## 4. Result-driven downstream branches

These branches are planning envelopes, not admitted jobs.

### TG3 — temporal-grounding mechanism

TG3 is eligible only if TG2R shows that fixed endpoint beats both future-off
and raw milestone and passes task safety. Freeze one matrix containing
`milestone_time`, `milestone_time_constant`, and `milestone_clipped` at seeds
1000--1002, followed by exact paired evaluation and fixed-checkpoint content-
and time-shuffle interventions.

A repair claim requires a positive 95% lower bound against raw milestone,
Holm-adjusted `p<0.05`, task safety, and recovery of at least half the TG2R
fixed-minus-raw mean gap. A temporal-use claim additionally requires correct
time-to-go to beat time-shuffled timing. If either gate fails, stop temporal
repair without adding post-hoc horizons, selectors, or gates.

### TG4 — source of utility and training protocol

TG4 is eligible only after TG1A content use or task-safe TG2R active-target
utility.

- **Source decomposition:** compare compatible clean-base, future-off,
  auxiliary-only, conditioning-only, parameter-matched-null, and accepted full
  checkpoints. Attribute pretraining, downstream shaping, and inference content
  only from their prespecified contrasts.
- **Joint versus staged training:** select one accepted target before outcomes;
  compare end-to-end joint optimization, predictor pretraining followed by a
  frozen predictor, predictor pretraining followed by joint unfreezing, and a
  compute/data-exposure-matched staged null at seeds 1000--1002. Each seed must
  receive its own predictor trajectory.
- **Fixed-plus-milestone mixture:** eligible only if fixed and raw each show
  task-safe benefit on prespecified non-overlapping task strata. Compare the
  mixture with the better single arm and a parameter-matched duplicate-target
  control. Do not mix two unsupported targets to search for a positive macro.

Every TG4 label needs a positive hierarchical lower bound, task safety, and the
corresponding fixed-checkpoint content intervention. If compute exposure cannot
be matched, report a systems tradeoff rather than a causal schedule advantage.

### TG5 — external replication

TG5 is eligible only after one task-safe positive TG2R or TG3 contrast. Freeze
exactly one contrast and replicate it on the complete four-suite LIBERO
protocol or a prespecified complete second RoboTwin panel. Cross-benchmark
wording requires the hierarchical interval to retain direction with every
suite/task reported; selected positive subsets are forbidden.

## 5. Stop and reporting rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, outcome weighting,
  oracle-transition, or failed helper jobs to search for a positive result.
- Do not evaluate or substitute an original rejected TG2 checkpoint.
- Partial rollouts, smoke tests, training losses, representation metrics,
  checkpoint existence, or unmatched evaluation seeds cannot pass a utility
  gate.
- Representation prediction does not establish control utility. Cadence
  sensitivity does not establish correct-content use. A public system score
  does not identify its causal component.
- Preserve the fixed -5-point task-safety threshold and every task-level
  regression. Do not promote a macro-only improvement.
- Do not tune target horizon, task groups, training seeds, checkpoint step,
  intervention mapping, retry recipe, or loss weight against closed-loop
  outcomes.
- Task_N remains outside this paper plan by operator instruction.

## 6. Canonical live sources

- Active scheduler summary: `logs/resource_scheduler_snapshot.md`
- Canonical mutable snapshot: `logs/resource_scheduler_snapshot.json`
- Canonical scheduler state: `logs/resource_scheduler_state.json`
- TG2R training contract:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_v1.json`
- TG2R post-training contract:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v1.json`
- TG1 retry500 amendment:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json`
- TG1 activation record:
  `logs/resource_markers/temporal_grounding_tg1_retry500_activation_v1.json`
- Completed evidence and protocol history:
  `lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`
