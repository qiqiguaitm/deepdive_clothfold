# MINT-VLA Scientific-Writing Audit

Updated: 2026-08-01

Target venue: ICLR 2027. The review applies the installed `scientific-writing`
skill's ICLR conventions and borrows Nature's claim-first, accessible prose.
Nature layout rules are not applied when they conflict with ICLR.

## Pass

- **Abstract structure:** 198 words; standard world-model terminology, integration gap, method, numerical
  findings, and a bounded interpretation are all present.
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

## Fail: requires new evidence

1. **The primary result is absent.** The corrected A0/A2-Abs/A3 table is still
   pending. A submission abstract cannot rely on the legacy weak-baseline pilot.
2. **Optimization uncertainty is absent.** Each primary arm needs at least three
   independent training seeds, seed-level values, and intervals.
3. **The contribution of predicted future content is unresolved.** A2 correct is tied with current
   in aggregate; within-task instance-shuffled and cross-task controls and the full A3 intervention
   panel are incomplete. Holm-adjusted paired tests are also absent.
4. **The mechanism is not factorized.** Target computation, predictor form, forward
   prefix, auxiliary loss, gradient route, and capacity change together between
   the main pilot arms.
5. **The task-specific claim is post hoc.** Block and bowl stacking plus reactive
   and geometric controls have not been tested under a predeclared protocol.
6. **A3 milestone supervision is incomplete.** The audited pilot artifact omits
   Stack-3, so its A3 result cannot support a benefit from recurrence-based milestones.
7. **Use with a second VLA is untested.** The adapter is separated from
   the $\pi_{0.5}$ prefix in the method, but T6 must test a second VLA before the
   abstract can claim empirical transfer across architectures.

## Warning: manuscript closure

- The confirmatory and intervention tables contain `pending` cells. Keep them
  in an internal draft only; remove incomplete tables from a submitted PDF.
- The abstract is at the upper edge of the skill's 150--200-word ICLR range.
  Replace pilot values with the final matrix rather than adding more sentences.
- The method describes recurrence mining as numbered prose. Convert it to an
  algorithm environment once the final six-task artifact fixes the definitive
  implementation.
- Figure 2 reports point estimates from one trained checkpoint. Its final form
  needs training-seed uncertainty rather than only rollout counts.
- Peak memory, paired inference latency, and action-server throughput remain
  unmeasured. Do not use `lightweight` as an empirical speed claim before those
  measurements are available.
- The observation-only predictor cannot resolve instruction-dependent futures.
  The limitation is now explicit, but an instruction-aware control is needed
  before claiming general milestone selection.

## Venue distinction

The title acronym, standalone Related Work, central Methods section, and
three-item contribution list fit ICLR. They would violate Nature-family
conventions, which prefer a short acronym-free title, literature woven into the
introduction, Methods at the end, and no contribution list. The paper should
retain the ICLR structure while using Nature's accessible and claim-first prose.
