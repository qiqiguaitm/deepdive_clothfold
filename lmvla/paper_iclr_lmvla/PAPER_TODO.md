# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-07 UTC

**Scope: GPU training and closed-loop evaluation only.** All immediately
available CPU/source/data checks have been run. Their completed results are in
`RESULTS_temporal_grounding_local_audit_v1.json` and Section 41 of
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`. This file contains no local-analysis,
writing, figure, build, cleanup, or publication task.

No item below is authorized for manual execution. The existing resource-aware
scheduler remains the sole execution owner. Before admitting any item, its job
bundle must freeze the exact source tree, data and checkpoint hashes, training
seeds, paired scene manifest, accepted-episode rules, intervention seeds,
report schema, analysis command, and stop rule. Do not change scheduler code or
configuration for this program.

## Evidence boundary fixed by the local audit

- The released RoboTwin LaWAM checkpoint uses 30 Hz over 1.2 s: `H=36`,
  `h_H=35`, and the evaluator executes `E=36` actions per query. Under the
  audited current two-frame loader rule, its model and execution endpoints
  coincide. The original training data/source are not local, so TG1A must
  preserve that provenance limitation.
- The historical local all6 matrix uses 50 Hz over 1.0 s: `H=50`, `h_H=49`,
  but the historical evaluator executes `E=36` actions. Its fixed training
  target lies beyond the executed prefix.
- Only 19.35% of frozen milestone pairs lie within the local 50-action
  training window, and only 12.50% lie within the 36-action executed prefix.
  Task means range from 53.64 to 130.44 frames.
- These are descriptive contract facts. They do not show that alignment causes
  utility, that misalignment causes regression, or that LaWAM uses correct
  future content.

## Execution order

| Order | ID | GPU work | Required status | Claim unlocked only if gate passes |
|---|---|---|---|---|
| 1 | TG1A | Released-checkpoint content interventions | Required | Correct future content is causally used by that checkpoint |
| 2 | TG1B | Historical 36-versus-50 execution-cadence panel | Required diagnostic | Historical local result is sensitive to the measured cadence mismatch |
| 3 | TG2 | Matched `future_off` / fixed endpoint / raw milestone training | Required | Replicated active-target utility and target-horizon effect |
| 4 | TG3 | Explicit time-to-go and chunk-clipped milestone training | Conditional on TG2 | Temporal grounding repairs raw milestones |
| 5 | TG4 | Clean-base and gradient-route factorization | Conditional on TG1A or TG2 | Component-specific causal attribution |
| 6 | TG5 | Prespecified external replication | Conditional on TG3 | Scoped cross-regime or cross-benchmark transfer |

TG1A, TG1B, and TG2 are the minimum new GPU evidence. TG3--TG5 are forbidden
unless their stated upstream gates pass.

## TG1A: released LaWAM fixed-checkpoint content panel

Use only the released RoboTwin checkpoint with SHA-256
`a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb`.
No training or checkpoint selection is allowed. Freeze one paired six-task
scene manifest and reuse the checkpoint, 30 Hz control contract, 36-action
execution cadence, observation, instruction, action bridge, and episode in
every condition.

- [ ] GPU evaluation: `normal`, the checkpoint's predicted endpoint feature.
- [ ] GPU evaluation: `shuffled`, a deterministic within-task,
  different-episode permutation with no self-match.
- [ ] GPU evaluation: `null`, the frozen zero/null route without removing
  parameters or changing action-expert execution.
- [ ] GPU evaluation: `persistence`, the current visual feature placed in the
  future slot with shape and scale matched to `normal`.
- [ ] GPU evaluation: `oracle`, only when an exact same-scene expert
  trajectory supplies the endpoint feature without action or success leakage.
  Otherwise run it as a separately labelled offline GPU action probe and
  exclude it from the closed-loop utility gate.

**TG1A gate.** Content use requires the lower bound of the hierarchical paired
95% interval for `normal - shuffled` to exceed zero and its Holm-adjusted paired
test to satisfy `p<0.05`. Route necessity requires `normal - null` separately.
Predictor limitation requires `oracle - normal` separately. Report all tasks;
no macro result overrides a task regression.

If `normal` does not beat `shuffled`, describe LaWAM as a system containing a
future-prediction route, not as evidence that correct future content causes its
control score.

## TG1B: historical execution-cadence sensitivity

This diagnostic quantifies the measured `H=50,E=36` mismatch without treating
cadence as future-content causality. Use the locally audited seed-2027
`local-WM` checkpoint
(`29ecbc3ee19585b5d9f3d3aa4bade8842e4bb016f88a3f0a3c97646164be321d`)
and `future_off` checkpoint
(`dfcf547f6d472a9540a71ea43f3da04925228cd7ccc17166290c68af04e6c538`).
Evaluate both at `E=36` and `E=50` on exactly paired scenes, producing a
two-checkpoint by two-cadence panel. Do not retrain or select a checkpoint from
these outcomes.

- [ ] GPU evaluation: `future_off`, executed at 36 and 50 actions per query.
- [ ] GPU evaluation: `local-WM`, executed at 36 and 50 actions per query.

**TG1B gate.** Report the paired difference-in-differences
`(local50-local36) - (off50-off36)` with a hierarchical 95% interval and every
task effect. A positive interval establishes cadence sensitivity specific to
the local-WM checkpoint; it does not establish content use or explain the
released LaWAM system. If both checkpoints change similarly, interpret the
result as a general replanning-cadence effect.

## TG2: execution-aligned matched training matrix

Freeze `E=H` before training so the new confirmatory matrix does not inherit
the historical 50-versus-36 mismatch. On the local 50 Hz all6 data this means
`H=E=50`, `h=49`. Hold the LaWAM backbone, initialization, action expert, data,
optimizer, training budget, target dimensionality, prediction capacity, loss
weight, injection route, and evaluator fixed. Use training seeds 1000--1002
and a frozen final step without evaluation-based selection.

- [ ] GPU train and paired evaluation: `future_off`, with downstream future
  prediction, distillation, and conditioning disabled by one audited route.
- [ ] GPU train and paired evaluation: `fixed_endpoint`, targeting the last
  valid and last executed index `z_{t+49}`.
- [ ] GPU train and paired evaluation: `raw_milestone`, targeting
  `z_{tau(t)}` without time-to-go.

The nine minimum training arms must pass parameter-tree, trainable-tree,
initialization-payload, dataset-order, normalization, optimizer-state, target-
coverage, checkpoint, and exact paired-scene audits before analysis.

**TG2 gates.**

1. Fixed-endpoint utility requires the hierarchical 95% interval lower bound
   for `fixed_endpoint - future_off` to exceed zero.
2. A target-horizon effect requires the lower bound for
   `fixed_endpoint - raw_milestone` to exceed zero.
3. Task safety requires no training-seed/task effect below -5 percentage
   points for a claimed winning arm against its stated baseline.
4. If both active targets fail against `future_off`, stop target engineering
   and write a bounded downstream-future-objective negative result.
5. If fixed and raw are statistically unresolved, do not run TG3. The
   descriptive horizon difference then remains unlinked to utility.

Resample training seeds first, then tasks and paired episodes. Publish every
seed-by-task effect; a macro average cannot hide a task regression.

## TG3: direct temporal-grounding interventions

Run only if TG2 establishes a fixed-versus-raw difference. Reuse the TG2
source, recipe, seeds, evaluator, `H=E`, and final-step rule.

- [ ] GPU train and paired evaluation: `milestone_time`, raw milestone plus
  normalized time-to-go, with a matched constant-time embedding control.
- [ ] GPU train and paired evaluation: `milestone_clipped`, targeting
  `z_{min(tau(t),t+49)}` and recording which branch each sample uses.
- [ ] If either arm passes, GPU fixed-checkpoint evaluation with normal,
  content-shuffled, and within-task time-shuffled conditions.

**TG3 gate.** A grounded arm must exceed `raw_milestone` with a positive-lower-
bound hierarchical 95% interval and Holm-adjusted `p<0.05`. A timing-use claim
also requires correct time-to-go to beat time-shuffled timing. A repair claim
requires recovery of at least half of the TG2 fixed-minus-raw mean gap and task
safety. If TG3 fails, reject temporal grounding as the demonstrated cause and
do not add post-hoc gates, horizons, or target selectors.

## TG4: source of any active future utility

Run only if TG1A or TG2 establishes active future utility.

- [ ] GPU matched training and evaluation: clean VLA from the same base
  initialization with no LaWAM pretraining or future modules.
- [ ] GPU matched training and evaluation: LaWAM-pretrained `future_off`.
- [ ] GPU matched training and evaluation: `auxiliary_only`, with no inference
  future content.
- [ ] GPU matched training and evaluation: `conditioning_only` and a
  parameter-matched null route.
- [ ] GPU matched training and evaluation: full accepted fixed/grounded arm.

**TG4 gate.** Attribute pretraining only from `future_off - clean`, downstream
shaping only from `auxiliary_only - future_off`, and inference content only
from `conditioning_only - parameter_matched_null`. Each label requires its own
positive-lower-bound interval and task-safety check.

## TG5: external validation

Run only for the single intervention selected by the TG3 gate. Freeze either
the complete four-suite LIBERO protocol or a second complete RoboTwin panel
before training; do not select a benchmark, task, or seed from candidate
outcomes.

- [ ] GPU matched training and paired evaluation of the accepted primary
  contrast and its content/time intervention on the frozen external panel.

**TG5 gate.** The primary contrast must retain direction with a hierarchical
95% interval excluding zero, with every suite/task reported. Cross-benchmark
wording is prohibited for a subset, saturated suite, or single training seed.

## Global stop rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, outcome weighting,
  oracle-transition, or failed helper jobs to search for a positive result.
- Partial rollouts, smoke tests, losses, representation metrics, checkpoint
  existence, or evaluation seeds without matched training seeds cannot pass a
  utility gate.
- A representation gain does not establish content use. A public system score
  does not identify its causal component. A cadence interaction does not prove
  future-content use.
- Preserve negative task effects. No positive macro can override the -5-point
  task-safety gate.
- If TG2 rejects both active targets, stop. If TG3 fails, stop temporal repair.
  If TG1A fails content use, retain that null even if a training package later
  improves utility.
