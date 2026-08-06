# MINT-VLA Scientific-Writing Audit

Updated: 2026-08-06

Target venue: ICLR 2027. The review applies the installed `scientific-writing`
skill's ICLR conventions and borrows Nature's claim-first, accessible prose.
Nature layout rules are not applied when they conflict with ICLR.

## Pass

- **Abstract structure:** 200 source words; standard world-model terminology,
  integration gap, method, numerical findings, and a bounded interpretation
  are present.
- **Introduction CARS moves:** territory, specific niche, and "Here we present"
  response appear in that order.
- **Contribution statement:** three concrete contributions now end the
  introduction; none claims SOTA or universal world-model benefit.
- **Figure-led explanation:** the architecture figure appears in the
  introduction and is explicitly identified as the $\pi_{0.5}$ instantiation,
  rather than the architecture-independent method definition.
- **Method scope:** Section 4 defines milestone discovery, native-space targets,
  prediction, and policy adaptation before presenting $\pi_{0.5}$ as one
  implementation. A second-VLA instantiation remains an explicit T6 evidence gate.
- **Nature-style figures:** figures use a white canvas, sans-serif type at final
  size, strokes above 0.5 pt, square schematic boxes, and blue-white-red signed
  heatmaps. Numeric signs provide a second cue beyond colour.
- **Claim-first Results:** section headings and opening sentences state the
  finding or unresolved claim before procedure.
- **Hedging:** measured values are stated directly; causal and generalization
  interpretations are bounded.
- **Terminology:** the paper defines prediction as estimating a future
  observation or latent visual state. `Residual-parameterized` describes the
  predictor, while `absolute milestone feature` describes the injected value.
  Ambiguous shorthand such as `future prediction`, `live/stale target`,
  `channel utility`, and `interface package` has been removed.
- **References:** 25 cited entries, with no `TBD` or `VERIFY` fields.
- **AI-pattern scan:** no blocklisted promotional vocabulary remains.
- **Reproducibility prose:** configuration hashes, artifact coverage, software
  versions, seeds, parameter counts, and AI-use disclosure are present.
- **Content causality:** the primary $\pi_{0.5}$ intervention uses a common
  518-scene intersection for all A2/A3 conditions. A separate supporting LaWAM
  analysis reports nine six-task method--control comparisons with 1,200 paired
  episodes each. Both use exact McNemar tests and Holm correction; their
  architectures and evidence roles remain explicit.
- **Spatial-interface boundary:** the privileged 4x4 probe reports its frozen
  two-task, 1,000-update scope, paired episode interval, metric direction, and
  offline rather than closed-loop status.
- **Efficiency boundary:** measured memory and latency are stated relative to
  A0 and are not presented as a speed ranking against other WAMs.
- **Oracle-transition boundary:** the direct three-seed closed-loop follow-up
  reports its crossing-zero interval, the negative seed-1002 effect, and all
  six task differences. Positive multistage means are not promoted over the
  three task regressions.
- **Semantic-interface boundary:** the fixed-checkpoint five-arm screen reports
  no-subtask and within-task shuffled controls, paired intervals, and all six
  task effects. The rejected prompt gate is not generalized to learned latent
  or semantic-prediction interfaces.
- **Outcome-diagnostic boundary:** the 120-rollout recurrence panel reports
  outcome separation together with 38.4--99.0% success false-positive rates.
  It is described as diagnostic information, not reward, value, failure
  detection, or control utility.
- **Method-identity boundary:** the completed negative MINT-VLA matrix and the
  policy-preserving predictive adapter are described as different method
  hypotheses. The former supports an interface-failure result; the latter has
  a positive seed-1000 closed-loop screen whose replication remains open.
- **P1 causal boundary:** P1 reports all six task effects and the matched A0,
  zero-route, shuffled-action, and masked-action controls. Its +13.42-point A0
  contrast is not converted into a content-specific claim because the
  normal--shuffled contrast is only +1.25 points (Holm-adjusted p=1.0).
- **R1 scope boundary:** the 62.92% predictive-plus-CRAVE result, its four
  task-level regressions, and its intervals against A0 and CRAVE-only are
  reported as rejection of the auxiliary recurrence extension, not of P1.
- **R4 scope boundary:** the seed-1000 table reports all tasks, the Stack-2
  regressions, and the crossing-zero interval against ordinary weighting. The
  completed three-seed table reports seed heterogeneity and both crossing-zero
  intervals. Demonstration weights are not described as value, reward, or
  planning.

## Closed evidence verdict

1. **The primary result is complete.** All nine corrected A0/A2-Abs/A3 rows
   pass the frozen 24-cell audit. Across three training seeds, A0 is 79.39%,
   A2-Abs 75.44%, and A3 70.81%.
2. **Optimization uncertainty is reported.** The paired hierarchical
   differences versus A0 are -3.94 pp for A2-Abs (95% CI [-6.58,-0.97]) and
   -8.58 pp for A3 ([-11.47,-5.75]).
3. **A positive method claim is rejected.** Content causality is null, the
   privileged spatial gate is adverse, and both conditioned arms regress in
   the corrected matrix. The manuscript is framed as a negative integration
   result.
4. **Scope expansions are closed.** The task-selector, mature-initialization,
   and second-VLA extensions required a positive primary utility gate. They are
   not used to recover a post-hoc positive claim.
5. **Legacy limitations remain explicit.** The pilot artifact omits Stack-3
   milestone supervision and is retained only as exploratory evidence.
6. **The oracle-transition follow-up is closed.** Its +0.53-point difference
   from A0 has 95% CI [-2.06,+3.00]; MT1 fails, and MT3--MT6 remain unlaunched.
7. **The privileged semantic screen is closed.** Correct next-subtask prompts
   score 77.08% versus 80.83% without a subtask and 77.50% with shuffled
   semantics. The primary interval and task-safety gates fail.
8. **The P1 screen is accepted but not replicated.** Seed-1000 predictive
   inference reaches 82.42% versus 69.00% for matched A0, with no task-level
   regression. P2 seeds 1001--1002 remain mandatory before a method-level
   utility claim.
9. **The R1 extension is closed.** Predictive-plus-CRAVE reaches 62.92% and
   violates four task-safety guards. This verdict is not generalized to the
   parent policy-preserving adapter.
10. **The R4 screen is not replicated.** Terminal outcome reaches 77.58% at
    seed 1000, but its three-seed effects are +2.81 points versus ordinary
    (95% CI [-0.94,+6.58]) and +3.81 versus outcome-free CRAVE
    ([-0.25,+7.97]). Both utility checks fail; the task-safety guard passes.

## Warning: manuscript closure

- The abstract is at the upper edge of the skill's 150--200-word ICLR range;
  any further result sentence requires a compensating cut.
- The method describes recurrence mining as numbered prose. Convert it to an
  algorithm environment once the final six-task artifact fixes the definitive
  implementation.
- Figure 2 remains explicitly labelled as the one-seed legacy pilot; the
  claim-bearing table reports all three corrected training seeds.
- The current build is 17 pages through references and 23 pages with the
  appendix. Final submission length must be checked against the ICLR 2027
  main-text limit before submission.
- The observation-only predictor cannot resolve instruction-dependent futures.
  The limitation is now explicit, but an instruction-aware control is needed
  before claiming general milestone selection.
- P2 is complete and accepted relative to its fixed matched A0. The result is
  reported as a full task-by-seed table rather than a new claim-bearing figure;
  its text explicitly withholds content-specific causality and independent
  baseline-seed replication. R4 remains a completed negative replication.

## Venue distinction

The title acronym, standalone Related Work, central Methods section, and
three-item contribution list fit ICLR. They would violate Nature-family
conventions, which prefer a short acronym-free title, literature woven into the
introduction, Methods at the end, and no contribution list. The paper should
retain the ICLR structure while using Nature's accessible and claim-first prose.
