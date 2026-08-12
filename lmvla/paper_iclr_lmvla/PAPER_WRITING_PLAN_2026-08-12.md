# Paper Writing Plan: Actionable Predicted Futures

Date: 2026-08-12

Target: ICLR 2027, nine-page main text plus references and appendix.

This document fixes the manuscript logic while TG4 remains unfinished. It does
not change an experiment protocol or authorize scheduler work. Canonical JSON
results override prose and tables.

## Central question and thesis

The paper asks:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

The current evidence supports a bounded thesis. Future-state predictability is
not sufficient for control utility. A released LaWAM checkpoint does use its
specific predicted endpoint content in closed loop, whereas MINT-VLA and the
PredictiveActionAdapter do not establish replicated control benefit. The
temporal relation between target and action chunk is a measured design
distinction and a plausible explanation, but TG1B does not identify cadence as
the cause. TG4 must finish before utility can be attributed to pretraining,
auxiliary shaping, inference conditioning, or their interaction.

## Contribution order

1. **Temporal contract and evidence ladder.** Define model horizon `H`, executed
   prefix `E`, target time `tau(t)`, and four distinct questions:
   predictability, matched package utility, fixed-checkpoint content use, and
   component attribution.
2. **Controlled predictive interfaces.** Present MINT-VLA and the
   PredictiveActionAdapter as architectures that expose different conditioning
   locations. The adapter is a policy-preserving, proposal-conditioned future-
   grid route into the action expert; it is an architectural contribution, not
   an established improvement claim.
3. **Causal and replicated evidence.** Show that the released LaWAM checkpoint
   depends strongly on correct endpoint content, while matched MINT-VLA and
   adapter tests establish important negative boundaries. Report every task
   regression and never replace training-seed variation with evaluation seeds.

## Claim--evidence matrix

| Claim | Canonical evidence | Status in paper |
|---|---|---|
| Released LaWAM uses its specific predicted content | TG1A normal--shuffled +53.67 pp, 95% CI [+36.08,+68.58], Holm p=6.75e-180; all six tasks positive | Established for one frozen checkpoint |
| LaWAM route is necessary and beats persistence | TG1A normal--null +58.83 pp; normal--persistence +69.58 pp; both gates pass | Established for one frozen checkpoint |
| Released LaWAM target is endpoint-aligned | CPU contract audit: H=E=36, target offset 35 | Descriptive contract evidence |
| Historical milestones are usually multi-chunk | 420,238 pairs; only 12.50% within the executed 36-action prefix | Descriptive target evidence |
| Cadence mismatch causes the LMWM result | TG1B difference-in-differences +1.42 pp, 95% CI [-3.00,+5.92] | Not established |
| MINT-VLA prediction implies control utility | Prediction cosine 0.8134 vs 0.7479 persistence, but control -8.58 pp, CI [-11.47,-5.75] | Rejected |
| PredictiveActionAdapter improves control across matched seeds | +13.42/-5.50/-2.08 pp; mean +1.94, CI [-5.78,+12.75] | Not established |
| PredictiveActionAdapter is task-safe | Regressions of -13, -9, and -5.5 pp in prespecified seed--task cells | Rejected |
| PredictiveActionAdapter uses correct proposal content | normal--shuffled +0.53 pp, CI [-2.14,+3.08], Holm p=0.534 | Not established |
| A specific LaWAM component causes the TG1A utility | TG4 | Placeholder until complete and audited |

## Main-paper structure

1. **Introduction.** Prediction quality is not a control claim; future
   representations encode both content and time. State the evidence gap and the
   bounded findings.
2. **Related work.** Organize around fixed-horizon prediction, long-horizon
   subgoals, integration interfaces, and evaluation rather than paper chronology.
3. **Temporal contracts and evidence requirements.** Define notation and the
   evidence ladder. Make clear that no rung implies the next.
4. **Predictive interfaces.** Describe released LaWAM, recurrence milestones and
   MINT-VLA, then the PredictiveActionAdapter. Distinguish prefix conditioning,
   full-grid conditioning, and action-expert residual routing.
5. **Experimental design.** Specify frozen scenes, training/evaluation seeds,
   paired interventions, hierarchical intervals, Holm correction, task-safety,
   and stop rules.
6. **Results.** Claim-first order: temporal contracts; LaWAM content use;
   cadence non-result; MINT and adapter boundaries; TG4 placeholder.
7. **Discussion and conclusion.** Interpret why a distant milestone may need a
   local conversion without claiming that timing caused the observed failures.

## Display plan

- **Figure 1:** temporal contract, three conditioning locations, and evidence
  ladder. It replaces the MINT-first hero figure.
- **Figure 2:** released-checkpoint TG1A condition rates and all six task-level
  normal--shuffled effects.
- **Table 1:** compact evidence ledger with estimand, interval, task-safety, and
  allowed conclusion.
- **Table 2:** PredictiveActionAdapter matched seed-by-task matrix, because its
  heterogeneity is central and cannot be hidden by a macro average.
- **Appendix:** full historical MINT, oracle, semantic, spatial, adapter,
  outcome-weighting, protocol-integrity, and TG4 audit details.

All figures use a white canvas, 6--8 pt sans-serif text at final width, strokes
of at least 0.5 pt, colourblind-safe colours with a second visual cue, and
self-contained captions defining samples and uncertainty.

## Conditional TG4 insertion

TG4 is the only unfinished claim-bearing experiment. Until its 18 training
cells, integrity gate, evaluations, and hierarchical analysis complete, the
manuscript must state that source attribution is unresolved. Once complete:

- add only prespecified accepted contrasts;
- include all seed and task effects, including regressions;
- distinguish a total package effect from pretraining, auxiliary, conditioning,
  and interaction effects;
- revise the title or abstract only if the audited result materially changes
  the central thesis.
