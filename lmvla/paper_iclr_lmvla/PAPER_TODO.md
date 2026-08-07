# Temporal-Grounding Evidence TODO

Updated: 2026-08-07 06:11 UTC

**Status: reopened as a planning-only evidence program.** No experiment in
this document is authorized for execution until its protocol, source tree,
data identities, checkpoints, training seeds, evaluation scenes, analysis
script, and stop rule are frozen and admitted by the existing resource-aware
scheduler. The scheduler currently reports zero pending and zero running jobs.
This TODO does not launch, reprioritize, restart, or configure any job.

The completed MINT-VLA, predictive-adapter P0--P5, R0--R4, efficiency, and
earlier LaWAM/LMWM diagnostic history are immutable prior evidence. Their
closure record is in `PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`; do not rerun them
to search for a positive result. The present program addresses a different
question:

> Does a predicted future representation constrain VLA action generation only
> when its temporal horizon is aligned with the action chunk or explicitly
> encoded?

The full manuscript replan is
`PAPER_REPLAN_TEMPORAL_GROUNDING_2026-08-07.md`.

## Current claim boundary

- The LaWAM implementation samples the current frame and the last valid frame
  of the same physical-time action chunk; its fixed-near-future target is
  therefore aligned with the chunk endpoint by construction.
- The frozen RoboTwin milestone artifact has variable target horizons with
  task means from 53.6 to 130.4 frames, whereas the LaWAM RoboTwin action
  horizon contains about 30 valid action steps. Recurrence milestones are
  typically multi-chunk targets and carry no explicit time-to-go.
- MINT-VLA predicts nontrivial recurrence-defined representations, but its
  completed matched three-seed control result rejects utility in the tested
  pi0.5 interface. Correct-content, privileged, and predictive-adapter
  interventions do not establish causal use of future content.
- These facts establish a temporal mismatch, not its causal role. Do not write
  that chunk alignment explains LaWAM or that variable horizon explains LMWM
  until the new matched interventions below pass.
- The public LaWAM system result does not isolate the contribution of future
  content from pretraining, backbone, action expert, data, or optimization.
  `Future-off` inherits LaWAM pretraining and is not a clean VLA baseline.

## Priority and decision order

| Priority | ID | Question | Required new evidence | Claim unlocked only if gate passes |
|---|---|---|---|---|
| Required | TG0 | Are target time, chunk time, and intervention identities audited? | CPU/source/data protocol freeze | Descriptive temporal-mismatch claim |
| Required | TG1 | Does a released or accepted LaWAM checkpoint use correct chunk-end future content? | Fixed-checkpoint normal/shuffled/null/persistence/oracle panel | Causal content-use claim for that checkpoint |
| Required | TG2 | Under matched training, does fixed chunk-end prediction improve over future-off and outperform raw milestones? | Three-seed future-off/fixed/raw matrix | Replicated target-horizon effect |
| Conditional | TG3 | If fixed and raw differ, does explicit temporal grounding repair raw milestones? | Three-seed time-to-go and chunk-clipped arms | Temporal-grounding mechanism claim |
| Conditional | TG4 | If active future utility exists, does it come from pretraining, downstream shaping, or inference content? | Clean-base and gradient-route factorization | Component-specific causal attribution |
| Conditional | TG5 | Does the accepted effect transfer across task regimes or a second benchmark? | Frozen external validation | Scoped generalization claim |

TG0--TG2 define the minimum publishable comparison. TG3 is the decisive
mechanism test. TG4 and TG5 are forbidden unless their upstream gates pass.

## TG0: temporal-contract and artifact audit

- [ ] Freeze notation: valid action count `H`, last sampled chunk offset
  `h=H-1`, raw milestone time `tau(t)`, normalized time-to-go
  `g_t=(tau(t)-t)/h`, and target representation `z`. Record any dataset whose
  observation/action timestamp convention requires a different endpoint.
- [ ] Audit the LaWAM dataloader and runtime contract for every proposed
  dataset: action frequency, `sec_chunk`, valid action count, sampled future
  frame index, padded action length, and number of executed actions.
- [ ] Audit every milestone pair for monotonicity, episode identity, current
  frame, target frame, horizon, task, terminal fallback, and source hash.
- [ ] Produce result-independent distributions of `g_t`, within-chunk versus
  beyond-chunk target rate, target persistence, and distinct targets per task.
- [ ] Freeze task strata before looking at new policy outcomes: ordered
  construction, reactive/contact, fine-grained geometry, and relational
  transfer. Record tasks that cannot be assigned without outcome knowledge.
- [ ] Freeze checkpoint hashes, evaluator scenes, accepted-episode rules,
  intervention seeds, report schemas, and one analysis command for TG1--TG3.

**TG0 gate:** all temporal and identity audits pass with no unexplained target
outside its source episode and no mismatch between declared and executed
action horizons. TG0 may support only a descriptive statement about temporal
alignment. It cannot support a control mechanism.

## TG1: LaWAM fixed-checkpoint content intervention

Use one frozen, fully audited LaWAM checkpoint. No training or checkpoint
selection is permitted in TG1. Every condition must reuse the same observation,
instruction, action bridge, scene, episode, and checkpoint.

- [ ] `normal`: the checkpoint's predicted chunk-end feature.
- [ ] `shuffled`: a deterministic within-task permutation of predicted future
  features with no episode self-match.
- [ ] `null`: the architecture's frozen null/zero route without removing
  parameters or changing action-expert execution.
- [ ] `persistence`: the current visual feature in the future-conditioning
  position, shape- and scale-matched to normal.
- [ ] `oracle`: a privileged chunk-end feature only where an exact same-scene
  expert trajectory supplies it. Verify scene and timestamp identity and that
  no action or success label enters the model. If an exact target is unavailable
  in closed loop, restrict oracle to a separately labelled offline action probe
  and exclude it from closed-loop utility gates.
- [ ] Evaluate all conditions on the same paired scenes, report every task,
  and run one hierarchical paired analysis with Holm correction over the
  preregistered normal--shuffled, normal--null, normal--persistence, and
  oracle--normal family.

**TG1 primary gate:** causal use of predicted content requires the lower bound
of the normal-minus-shuffled hierarchical 95% interval to exceed zero and the
Holm-adjusted paired test to be below 0.05. Route necessity requires the
normal-minus-null gate separately. Predictor limitation requires oracle to
exceed normal under the same criteria; oracle alone cannot establish utility.

If normal does not beat shuffled, describe the LaWAM checkpoint as a system
that contains a future-prediction route, not as evidence that correct future
content causes its control performance. Continue to TG2 only to compare
training objectives, not to rescue a content-use claim.

## TG2: matched target-horizon training matrix

TG2 holds the LaWAM backbone, initialization, action expert, data, optimizer,
training budget, adapter capacity, and evaluation protocol fixed. Training
seed is the top-level unit. Use at least seeds 1000--1002 and select a frozen
final step without evaluation-based checkpoint selection.

- [ ] `future_off`: LaWAM-pretrained initialization with downstream future
  prediction, distillation, and future conditioning disabled by a frozen,
  audited implementation.
- [ ] `fixed_endpoint`: the original last-valid-index chunk target `z_{t+h}`.
- [ ] `raw_milestone`: the recurrence-defined target `z_{tau(t)}` with the same
  target dimensionality, prediction capacity, loss weight, and injection route
  as `fixed_endpoint`.
- [ ] Audit parameter equality, trainable trees, initialization payloads,
  dataset order, normalization, optimizer state, target coverage, and final
  checkpoints for all nine minimum arms.
- [ ] Evaluate each matched seed on an exact paired scene manifest. Resample
  training seeds first, then tasks and paired episodes. Report seed-by-task
  effects; no macro average may hide a task regression.

**TG2 gates:**

1. Fixed-endpoint utility requires the `fixed_endpoint - future_off`
   hierarchical 95% interval lower bound to exceed zero.
2. A replicated horizon effect requires the `fixed_endpoint - raw_milestone`
   lower bound to exceed zero.
3. Task safety requires no seed/task effect below -5 percentage points for the
   claimed winning arm versus its stated baseline.
4. If both active targets fail against `future_off`, stop target engineering
   and write a bounded downstream-future-objective negative result.
5. If fixed and raw are statistically unresolved, do not run TG3; temporal
   mismatch exists descriptively but is not linked to utility.

## TG3: direct temporal-grounding interventions

Run TG3 only if TG2 establishes a fixed-versus-raw difference. Reuse the TG2
source, recipe, seeds, and evaluator; freeze both new arms before any training.

- [ ] `milestone_time`: raw milestone plus an explicit normalized time-to-go
  embedding `g_t`, with a matched constant-time embedding control.
- [ ] `milestone_clipped`: target the raw milestone if it occurs within the
  current chunk; otherwise target the demonstration frame at the last valid
  chunk index `z_{min(tau(t),t+h)}`. Record the fraction that reduces to each
  branch.
- [ ] If either arm passes, run fixed-checkpoint normal/shuffled/time-shuffled
  interventions to test whether the policy uses both content and timing.

**TG3 primary gate:** at least one grounded arm must exceed `raw_milestone`
with a positive-lower-bound hierarchical 95% interval and Holm-adjusted paired
`p<0.05`. A timing-use claim additionally requires normal time-to-go to beat a
within-task time-shuffled control. A repair claim requires recovery of at least
half of the TG2 fixed-minus-raw mean gap and task safety.

If TG3 fails, reject temporal grounding as the demonstrated cause. Do not add
learned gates, adaptive horizons, or new target selectors post hoc.

## TG4: source of apparent LaWAM utility

Run TG4 only if TG1 or TG2 establishes active future utility.

- [ ] Train a clean, budget-matched VLA from the same base initialization with
  no LaWAM pretraining or future modules.
- [ ] Compare clean VLA, LaWAM-pretrained `future_off`, and active
  `fixed_endpoint` at matched training seeds.
- [ ] Factor training shaping from inference content using `auxiliary_only`
  and `conditioning_only` arms with parameter-matched null modules and frozen
  gradient routes.

**TG4 gates:** pretraining utility is `future_off - clean`; downstream shaping
is `auxiliary_only - future_off`; inference content is
`conditioning_only - parameter_matched_null`. Each causal label requires its
own positive-lower-bound interval and task-safety check. Do not infer one
component from a comparison that changes another.

## TG5: conditional external validation

Run TG5 only for the single intervention selected by the frozen TG3 gate. Do
not select a benchmark, task, or seed after observing candidate outcomes.

- [ ] Choose either the complete four-suite LIBERO protocol or a frozen second
  RoboTwin task panel before training. Match the corresponding LaWAM baseline,
  action horizon, data budget, training seeds, and evaluation scenes.
- [ ] Replicate the accepted primary contrast and its content/time
  intervention. Report all suites/tasks, including regressions.
- [ ] If a second VLA is proposed, define it as a separate gate with its own
  clean baseline; architecture compatibility is not empirical transfer.

**TG5 gate:** the primary contrast retains direction with a hierarchical 95%
interval excluding zero. Cross-benchmark wording is prohibited if only a
subset, saturated suite, or one training seed is available.

## Publication gates

- [ ] Do not rewrite the title or abstract as a positive method claim until
  TG2/TG3 determine the paper branch. Keep all unfinished numbers as explicit
  placeholders.
- [ ] Main-paper tables must expose every training seed and every task. Fixed
  checkpoint interventions and training-seed comparisons must not share one
  unlabeled uncertainty estimate.
- [ ] Figure 1 must show the fixed action horizon `H`, the LaWAM chunk-end
  target, the variable milestone `tau(t)`, and the missing or explicit
  time-to-go. It must not depict temporal grounding as causal before TG3.
- [ ] Claim-bearing figures require a white canvas, 6--8 pt sans-serif text,
  strokes at least 0.5 pt, sentence-case labels, a colourblind-safe palette
  plus non-colour cues, and captions defining samples, seeds, intervals, and
  whether evidence is descriptive or causal.
- [ ] Keep the ICLR main text at nine pages. The main narrative receives at
  most four figures and two tables; full audits, per-seed/task matrices, and
  historical MINT evidence belong in the appendix.
- [ ] Preserve the completed MINT negative result even if a temporally grounded
  arm succeeds. A new positive arm does not retroactively validate old content
  or erase task-level regressions.
- [ ] Before manuscript promotion, run claim-to-artifact checks, LaTeX build,
  undefined-reference/overfull checks, figure preflight, and Web attachment
  synchronization.

## Stop and scope rules

- The resource-aware scheduler remains the sole execution owner. This TODO is
  not a queue, cron entry, daemon, or launch authorization.
- Do not reopen original MINT-VLA, predictive-adapter P0--P5, R0--R4, oracle
  transition, outcome weighting, or failed helper jobs.
- Do not tune target horizons, task groups, loss weights, intervention seeds,
  or checkpoint steps against closed-loop outcomes.
- Partial rollouts, smoke tests, losses, representation metrics, and evaluation
  seeds without matched training seeds cannot pass a utility gate.
- A positive macro does not override a negative task effect. A representation
  gain does not establish content use. A public system score does not identify
  the component that caused it.
- If TG2 rejects both active targets, stop. If TG3 fails, stop temporal repair.
  If TG1 fails content use, retain that null even if a training package later
  improves utility.
