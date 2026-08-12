# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-12 04:07 UTC

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

## 2. Closed TG2R recovery matrix

TG2R repeats `future_off`, `fixed_endpoint`, and `raw_milestone` at training
seeds 1000--1002. It preserves the original initialization, target routes,
data, global batch 128, four-GPU world size, eight workers per rank, 20,000
updates, `H=E=50`, and fixed-final-checkpoint selection. Its only scientific
training change is `datasets.vla_data.in_order: false -> true`.

Eight training cells and their location-aware materializers are complete and
archived in Section 44. The integrity gate closed the remaining branch:

- [x] **TG2R-T03 [COMPLETE: `t-20260811075607-jqpjx`]** Trained `future_off`,
  seed 1002 on four North GPUs under the operator-authorized primary-identity
  duplicate. The formal job completed at 11:28 UTC. The detached backup
  request and all of its output remain excluded from checkpoint, integrity,
  and evaluation provenance. The operator has separately authorized another
  primary-identity seed-1002 submission only if recovery is required; no
  additional training is needed while this completed artifact verifies.
- [x] **TG2R-M03 [COMPLETE under v7]** Materialize the complete seed-1002
  full-state tree from North to East with source/destination SHA-256
  verification, including initialization and four rank-order sidecars. The v6
  attempt verified and atomically installed all 19 checkpoint-tree files at
  11:58 UTC, then exited before sidecar validation because the checkpoint-only
  parallel-transfer flag leaked into the nested four-file sidecar copy. The v7
  operational amendment disables that flag only for the nested copy. The
  scheduler retry reused the installed checkpoint, materialized and normalized
  all sidecars, and wrote the canonical marker at 12:06 UTC without retraining
  or retransmitting the large checkpoint tree.
- [x] **TG2R-I1 [COMPLETE; GATE REJECTED: `t-20260812114314-cnp6q`]** The
  admitted East root-context gate verified all nine terminal jobs, fixed step
  20,000, optimizer state, initialization equality within seed, rank-order
  equality across arms within seed, and the remaining frozen protocol fields.
  It then rejected the matrix because seeds 1000--1002 induce the same rank
  data orders under `in_order=true`, violating the preregistered independent-
  training-seed requirement. The immutable decision is
  `RESULTS_temporal_grounding_tg2r_integrity.json`; the failed `.ok` marker was
  not created and the same protocol must not be retried.

No evaluation below is admissible before
`logs/resource_markers/temporal_grounding_tg2r_training_integrity.ok` exists.
Each evaluation uses four East GPUs, the unchanged six-task/four-evaluation-seed
scene manifest, and exactly 1,200 accepted episodes.

- [x] **TG2R-E01--E09 [RETIRED by I1; 0 evaluations executed]** The nine
  checkpoint trees remain audit artifacts only. Running their closed-loop
  evaluations would not repair the missing independent training-seed unit and
  cannot contribute to a TG2 result.

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

- [x] **TG2R-A1 [RETIRED by I1]** No canonical TG2 result or gate marker is
  produced because no admissible evaluation matrix exists.

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

- [x] **TG1A-E1 [COMPLETE: `t-20260811151421-rlt29`, 24/24]** Evaluate normal
  from an empty root under retry500 and produce a verified new feature
  capture.
- [x] **TG1A-E2 [COMPLETE: `t-20260811151424-z9hrr`, 24/24]** Evaluate null from
  an empty root under retry500.
- [x] **TG1A-E3 [COMPLETE: `t-20260811220414-45x4t`, 24/24]** Evaluated
  persistence from an empty root under retry500 with fixed seeds verified.
- [ ] **TG1A-E4 [RUNNING: `t-20260812040634-7kwps` + tail
  `t-20260812120538-n7jgs`, East, 16/24]** Evaluate the frozen within-task
  different-episode shuffled mapping. The scheduler-owned v2 tail passed the
  same TG1A bundle, retry500, and runtime-v11 checks, then attached one locked
  worker per seed to claim only `stack_blocks_three`; the original four workers
  continue the disjoint `stack_blocks_two` cells. The scheduler records both
  the aggregate tail episode count and four per-seed counters, and treats 30
  minutes without progress as stale so that one stuck seed cannot be hidden by
  the other workers. The rejected v1 helper
  `t-20260812120240-wdzvz` failed before claiming a cell because it omitted the
  frozen runtime-v11 verifier context, and contributes no episode.
- [x] **TG1B-E1 [COMPLETE: `t-20260812073026-gt69q`, 24/24]** Evaluated
  future-off, `E=36`.
- [x] **TG1B-E2 [COMPLETE: `t-20260812073031-p9l4x`, 24/24]** Evaluated
  future-off, `E=50`.
- [x] **TG1B-E3 [COMPLETE: `t-20260812073036-54vnc`, 24/24]** Evaluated
  local-WM, `E=36`.
- [x] **TG1B-E4 [COMPLETE: `t-20260812073041-kf64m`, 24/24]** Evaluated
  local-WM, `E=50`.

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
- [ ] **TG1A-A1 [REGISTERED; BLOCKED by E1--E4]** Run the frozen TG1A
  analysis. The common statistical helper is included in the scheduler
  ready-hash closure.
- [x] **TG1B-A1 [COMPLETE; GATE REJECTED]** The frozen analysis completed after
  all four verified result roots were materialized locally. The cadence
  difference-in-differences is +1.42 percentage points with hierarchical 95%
  CI [-3.00, +5.92]; the interval crosses zero, so
  `local_wm_specific_cadence_sensitivity=false`. This is a diagnostic negative
  result and does not establish future-content use.

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
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_posttraining_v7.json`
- TG2R integrity rejection:
  `lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg2r_integrity.json`
- TG1 retry500 amendment:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json`
- TG1 activation record:
  `logs/resource_markers/temporal_grounding_tg1_retry500_activation_v1.json`
- Completed evidence and protocol history:
  `lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`
