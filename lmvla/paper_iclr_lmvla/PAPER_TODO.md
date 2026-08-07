# Predictive-Adapter Evidence-Strengthening TODO

Updated: 2026-08-07 02:35 UTC

This file contains only unfinished evidence and current publication gates.
Completed MINT-VLA, P0--P2, R0--R4, efficiency, and execution history remain in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`. Canonical JSON results take precedence
over this plan.

P3--P5 are registered in the resource-aware scheduler after frozen preflights.
The scheduler owns their training, evaluation, analysis, and retry lifecycle;
do not manually launch, stop, restart, or reprioritize a job.

## Execution snapshot

- P3 A0 seeds 1001 and 1002 completed from the official initialization on
  North 8xH20 each as `t-20260807015907-2jh5j` and
  `t-20260807015912-m47wd`. Both fixed step-49,999 checkpoints pass the frozen
  source, data, normalization, parameter-tree, optimizer-state, payload, and
  atomic-commit audits. Their 24-cell/1,200-episode evaluations are complete,
  exactly paired, and contain no failed task.
- P4 masked, shuffled, and zero-gate seeds 1001 and 1002 completed on East
  8xH20 as `t-20260806224622-vnfhj`, `t-20260807013535-jr5mk`, and
  `t-20260807042808-g8gsz`. All six 1,200-episode reports passed the frozen
  pairing verifier. The three-seed analysis is complete and all three mechanism
  gates fail.
- P5 exact-public paired evaluation completed on the local 2xA100 host. All 24
  cells and 1,200 episodes pass exact scene-key verification. The public
  checkpoint reaches 78.17%; the candidate mean difference is +2.44 points
  with hierarchical 95% CI `[-0.72,+5.39]`, so the mature-initialization gate
  fails.
- P3 evaluation used 16 North H20s at peak through non-overlapping formal and
  helper workers. The seed-1002 canonical report was finalized from all 24
  completed cells after the GPU wrapper found no unclaimed task; no episode was
  rerun or omitted. P3 analysis is complete. No claim-bearing GPU job remains,
  and no further training is authorized by this TODO.

## Current evidence boundary

- P2 establishes replicated predictive-adapter candidate utility relative to
  one fixed current-source A0 checkpoint: seed effects are +13.42, +9.08, and
  +12.33 points, with equal-seed mean +11.61 points and hierarchical paired 95%
  CI `[+8.31,+14.67]`.
- P2 does **not** include independently trained A0 seeds 1001 and 1002. Its
  interval captures candidate-seed and paired-episode variation, not baseline-
  training variation.
- P3 rejects independent matched-seed utility. Matched candidate-minus-A0
  effects are +13.42, -5.50, and -2.08 points at seeds 1000--1002. Their
  equal-seed mean is +1.94 points with hierarchical 95% CI
  `[-5.78,+12.75]`; the interval crosses zero and the task-safety gate also
  fails. P2 is therefore demoted to candidate-seed replication against one
  fixed A0 rather than an independently matched method effect.
- P4 does not identify the mechanism. Across three candidate seeds,
  normal-minus-shuffled is +0.53 points with hierarchical 95% CI
  `[-2.14,+3.08]` and Holm-adjusted `p=0.534`; normal-minus-zero-gate is +1.50
  points with CI `[-1.31,+4.39]` and adjusted `p=0.205`; normal-minus-masked is
  +1.33 points with CI `[-2.22,+4.86]` and adjusted `p=0.224`. Content-specific
  causality, route necessity, and action-conditioning use therefore all remain
  unidentified.
- P5 establishes an exact paired public reference. The public pi0.5 checkpoint
  reaches 78.17%, versus 82.42%, 78.08%, and 81.33% for candidate seeds
  1000--1002. Their effects are +4.25, -0.08, and +3.17 points; the equal-seed
  mean is +2.44 points with hierarchical 95% CI `[-0.72,+5.39]`. The interval
  crosses zero, so there is no established improvement over the mature public
  initialization.
- R1 and R4 are closed negative extensions. Do not reopen recurrence-aligned
  auxiliary training or outcome weighting to recover a positive result.

## Priority and decision order

| Priority | ID | Question | New work | Claim unlocked only if gate passes |
|---|---|---|---|---|
| Required | P3 | Does the P2 effect survive independent matched A0 training seeds? | Complete; gate rejected | No independently matched utility claim |
| Required | P4 | Does the policy use the correct predicted action content? | 6 fixed-checkpoint control evaluations | Content-, route-, or action-conditioning causality, comparison by comparison |
| High | P5 | Does the adapter improve the mature public initialization? | Audit/recover or rerun 1 public evaluation | Improvement beyond the pretrained checkpoint rather than only weak-finetuning recovery |
| Conditional | P6 | If content remains unresolved, which training component produces utility? | Closed without execution because P3 failed | None |
| Reviewer-triggered | P7 | Does the matched effect persist on unseen scene seeds? | Closed unless a reviewer explicitly reopens it | None |

P3--P5 answer distinct questions and may be prepared in parallel after their
protocols are frozen. P6 is blocked until P3--P5 finish. P7 is not part of the
default compute plan.

## P3: independent matched-baseline replication

The current P2 denominator is the single 69.00% A0 seed-1000 report. P3 closes
the largest reviewer-facing uncertainty by training the missing A0 seeds under
the exact current-source P1/P2 recipe.

- [x] Freeze a P3 protocol that pins the existing source audit, official pi0.5
  initialization, dataset, mean/std normalization, batch 16, 50,000 updates,
  optimizer, final step 49,999, and the existing 24-cell scene manifest.
- [x] Train A0 seeds 1001 and 1002 with `pi05_predictive_adapter_p1_a0_exact`.
  Intermediate checkpoints, loss curves, and smoke tests are health telemetry
  only; select the fixed final checkpoint without evaluation-based selection.
- [x] Audit both final checkpoints for source, data, normalization, parameter
  tree, optimizer state, payload, and atomic commit identity.
- [x] Evaluate both A0 checkpoints on 24 task-by-evaluation-seed cells and 1,200
  episodes each, with exact episode pairing to the corresponding existing P2
  candidate seed.
- [x] Analyze candidate-minus-A0 for matched training seeds 1000--1002 with
  training seed as the top-level resampling unit and paired episodes nested
  within each of the six equally weighted tasks. Use at least 20,000 frozen
  bootstrap draws and report every seed and task effect.

**P3 primary gate:** the hierarchical 95% interval lower endpoint for the
three matched candidate-minus-A0 pairs must exceed zero. A statement that the
gain is task-safe additionally requires no seed/task effect below -5 points.
If the interval crosses zero, demote P2 to replicated-candidate evidence against
one fixed A0 and do not describe it as independent matched-seed replication.

**P3 result:** rejected. The mean matched-seed effect is +1.94 points with
hierarchical 95% CI `[-5.78,+12.75]`. Seed effects are +13.42, -5.50, and
-2.08 points, and seed 1001 contains task regressions below -5 points. The
result is not independently replicated and is not task-safe.

## P4: three-seed inference intervention panel

P4 reuses the three accepted candidate checkpoints and their existing normal
reports. It adds no training. Every control must retain the same checkpoint,
scene, episode, observation, and action bridge as normal inference.

- [x] Freeze deterministic `zero_gate`, `shuffled`, and `masked` intervention
  identities, permutation seeds, checkpoint hashes, result names, and exact
  pairing checks before running a new rollout.
- [x] Evaluate zero-gate, shuffled-action, and masked-action conditions at
  candidate seeds 1001 and 1002: six new 24-cell/1,200-episode reports. Reuse
  seed-1000 reports only after their episode keys and hashes pass the new audit.
- [x] Run a three-seed hierarchical paired analysis for normal-minus-shuffled
  (correct action content), normal-minus-zero-gate (route use), and normal-
  minus-masked (action-conditioning use). Treat these as a frozen family of
  three comparisons and apply Holm correction to their pooled paired tests.
- [x] Report every condition, training seed, and task. Do not use a positive
  macro to claim uniform use when a task effect is negative.

**P4 claim gates:**

- Content-specific causality requires the normal-minus-shuffled hierarchical
  95% interval lower endpoint to exceed zero and its Holm-adjusted paired test
  to be below 0.05.
- Route necessity and action-conditioning use are separate claims and require
  the corresponding zero-gate or masked comparison to pass the same criteria.
- Passing zero-gate or masked while failing shuffled supports use of the route,
  not use of the correct predicted content.
- If normal-minus-shuffled remains unresolved, retain the paper's current
  conclusion: the adapter package has utility, but its predictive-content
  mechanism is unidentified.

## P5: mature-public-checkpoint reference

The candidate mean is only 2.44 points above the 78.17% exact-paired public
pi0.5 calibration, and the public task profile is heterogeneous. P5 determines
whether fine-tuning with the adapter improves the initialization itself or
mainly prevents the degradation observed in the current-source A0.

- [x] Audit whether the original public 24-cell run can recover all 1,200
  per-episode outcomes with exact keys and the frozen manifest SHA. Aggregate
  task summaries alone are insufficient for a paired claim.
- [x] If any episode outcome or identity is unavailable, reevaluate the exact
  public checkpoint once on the existing 24-cell/1,200-episode manifest. Do not
  train or tune the public checkpoint.
- [x] Compare every P2 candidate seed with the fixed public checkpoint using
  paired episodes, then average candidate training seeds equally. Report all
  six task effects; the existing descriptive profile suggests that a positive
  macro could coexist with Hammer and ranking regressions.

**P5 gate:** a claim that the adapter improves the mature initialization
requires the hierarchical 95% interval lower endpoint for candidate minus
public checkpoint to exceed zero. This comparison still has a fixed public
denominator and does not replace P3. If the interval crosses zero, state that
the adapter improves matched fine-tuning relative to A0 but does not establish
improvement over the pretrained checkpoint.

## P6: conditional training-component factorization

Run P6 only if P3 confirms matched utility and P4 fails to identify correct
content. Its purpose is to distinguish a useful predictive objective from P0
initialization and extra action-expert capacity. Freeze two parameter-matched
controls before training:

**Closed without execution:** P4 failed its content gate, but P3 did not
confirm matched utility. The conjunction required to authorize P6 is false;
using idle GPUs for these arms would be an unregistered post-hoc expansion.

1. `p0_init_no_aux`: the same adapter tree and P0 overlay as the accepted
   candidate, but predictive auxiliary-loss weight zero during policy training;
   the action objective may still train the route.
2. `random_init_no_aux`: the same adapter tree and route with per-seed frozen
   random initialization and predictive auxiliary-loss weight zero.

- [ ] Audit equality of inherited pi0.5 parameters, data order, optimizer,
  scheduler, batch, updates, parameter count, and action bridge across full
  candidate and both controls. Document the intentionally different adapter
  initialization and auxiliary-loss fields.
- [ ] Train both controls at seeds 1000--1002 to fixed step 49,999 and evaluate
  every checkpoint on the same 24 cells and 1,200 episodes.
- [ ] Analyze three preregistered contrasts with hierarchical intervals and
  Holm correction: full candidate minus `p0_init_no_aux` (continued auxiliary
  supervision), `p0_init_no_aux` minus `random_init_no_aux` (P0 predictive
  pretraining), and `random_init_no_aux` minus A0 (parameter/route capacity).

No component receives a causal label unless its own interval excludes zero.
If all contrasts are unresolved, report the positive result as utility of the
full training package and stop mechanism expansion.

## P7: reviewer-triggered scene robustness

Do not schedule P7 by default. If P3 passes and submission review risk justifies
the cost, freeze previously unused RoboTwin scene seeds before any rollout and
evaluate matched A0/candidate seeds 1000--1002 without retraining. Use the same
six tasks, four evaluation seeds, 50 accepted episodes per cell, and the P3
hierarchical analysis. This can support robustness to new scenes within
RoboTwin only; it cannot support new-task, real-robot, or second-VLA claims.
P3 failed, so P7 is closed under the current plan even before reviewer cost is
considered.

## Publication and reporting gates

- [x] Update the abstract's replicated-utility wording after P3 finishes.
  P3, not the already accepted P2 gate, determines whether “matched training
  seeds” is admissible.
- [x] Add a content-specific statement only if P4's normal-minus-shuffled gate
  passes. Otherwise preserve the current explicit null attribution.
- [x] Present P5 beside, not instead of, P3 so a strong or weak public reference
  cannot obscure the matched-training comparison.
- [x] Any new main-paper table must show all training seeds and six tasks. Any
  new claim-bearing figure must use a white canvas, 6--8 pt sans-serif text,
  strokes at least 0.5 pt, sentence-case labels, colourblind-safe colours plus
  marker/line-style cues, and captions defining samples and interval hierarchy.
- [ ] Keep the paper within the ICLR main-text limit. Prefer replacing the
  current P2 table with a consolidated P3/P4 table rather than adding pages.
  The P2 table has been replaced by the consolidated P3 table, but the compiled
  draft still places references after approximately 16 main-text pages (23
  pages including references and appendix). Structural compression remains a
  publication task; it does not authorize additional experiments.

## Stop and scope rules

- Do not reopen original one-token MINT-VLA, oracle-transition, R1, or R4 arms.
- Do not launch a second VLA, new-task training, or real-robot study from this
  TODO. Those are separate projects unless requested by a reviewer or a new
  explicit paper claim.
- Do not infer prediction-content causality from representation cosine,
  checkpoint loss, zero-gate alone, or a candidate-versus-A0 utility result.
- Do not infer robustness from simulator evaluation seeds when baseline
  training seeds are missing.
- Failed platform attempts before a valid episode are operational records, not
  evidence. Partial cells, smokes, and intermediate checkpoints never enter a
  claim-bearing table.
