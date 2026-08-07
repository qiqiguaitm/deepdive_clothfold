# MINT-VLA -> ICLR 2027 Paper

## Current Story

- Method: MINT-VLA stores mined milestone frames, re-encodes targets with the current VLA visual encoder during training, predicts an absolute milestone through a residual parameterization, and connects it to the VLA's native conditioning input. The current pi0.5 instantiation uses one prefix token.
- Primary VLA/benchmark: pi0.5 on RoboTwin 2.0. LIBERO and LaWAM are supporting diagnostics.
- Structure: problem and controlled comparisons -> MINT-VLA -> completed
  three-seed negative result -> interface diagnosis -> policy-preserving repair
  screen -> rejected independent matched-seed replication.
- Questions: Q1 matched control performance; Q2 predicted content; Q3
  factorized integration mechanism; Q4 training-seed robustness, task scope,
  and efficiency. Q5 second-VLA instantiation is closed because the draft makes
  no multi-architecture claim.

## Evidence Boundary

- The completed A0--A3 matrix is a legacy matched pilot: joint-delta actions, quantile normalization, batch 64, 20k updates, one training seed per arm.
- Pilot macros: A0 35.50, A2-Abs 48.75, A3 49.58. A3-A2-Abs is only +0.83 pp.
- The formal public pi0.5 same-bridge calibration is 78.42% (941/1,200), showing the pilot baseline is weak.
- The corrected absolute-action/mean-std/batch-16/50k matrix is complete across training seeds 1000--1002: A0 79.39%, A2-Abs 75.44%, and A3 70.81%.
- Paired hierarchical differences versus A0 are -3.94 pp for A2-Abs (95% CI [-6.58,-0.97]) and -8.58 pp for A3 ([-11.47,-5.75]); the paper therefore makes a negative integration result, not a method-benefit claim.
- A3 is observation-conditioned and not fully gradient-isolated. LaWAM Future-off is not a clean no-WM VLA baseline.
- The policy-preserving predictive adapter is a separate method hypothesis. Its
  seed-1000 P1 screen is direct positive closed-loop evidence: 82.42% versus
  69.00% for matched current-source A0, with no task regression. P2 replicated
  candidate training against that one fixed A0, but P3 independently trained
  the missing A0 seeds and rejected matched-seed utility: effects are +13.42,
  -5.50, and -2.08 pp, with mean +1.94 pp and hierarchical 95% CI
  [-5.78,+12.75]. The result is neither independently replicated nor task-safe.
- P4's three-seed normal-minus-shuffled effect is +0.53 pp (95% CI
  [-2.14,+3.08], Holm-adjusted p=0.534); route and masked-action controls also
  cross zero. Content, route necessity, and action-conditioning use remain
  unidentified.
- P5's exact-paired public pi0.5 reference is 78.17%. Candidate-minus-public is
  +2.44 pp with 95% CI [-0.72,+5.39], so there is no established improvement
  over the mature initialization.
- Corrected R1 predictive-plus-CRAVE reaches 62.92%, below A0 by 6.08 pp and
  CRAVE-only by 5.75 pp, with four task regressions above five points. This
  rejects the recurrence-aligned auxiliary extension, not the parent P1
  adapter.
- R4 seed 1000 is complete and audited: terminal-outcome weighting reaches
  77.58%, versus 74.25% ordinary and 71.08% outcome-free CRAVE. Its 95% interval
  against ordinary crosses zero, and Stack-2 regresses by 3.5 pp, so this is an
  accepted directional screen rather than a replicated policy effect.
- The frozen R4 three-seed replication is complete and rejects a replicated
  utility claim. Terminal-outcome weighting averages 74.94%, versus 72.14%
  ordinary and 71.14% outcome-free CRAVE, but both hierarchical 95% intervals
  cross zero. These arms test demonstration weighting, not Q-values, action
  advantages, rewards, critics, or model-predictive control.
- `PAPER_TODO.md` is complete under the frozen evidence plan. P6 and P7 were
  closed without execution by their preregistered stop conditions; no new
  experiment is authorized.

## Files

- `main.tex`: paper entry point.
- `sections/`: manuscript sections.
- `numbers.tex`: centralized numeric macros.
- `PAPER_TODO.md`: dependency-ordered experiment and writing plan.
- `ANALYSIS_pi05_preserving_wm_integration_2026-08-04.md`: architecture
  screening and the selected pi0.5-preserving world-model route.
- `PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`: completed evidence and prior TODO
  history removed from the active plan.
- `../lmwm/docs/PROGRESS_pi05_vla_baseline_2026-08-01.md`: latest baseline/causal audit.

Build with `make`. The draft uses the ICLR 2026 style until the ICLR 2027 template is released.

## Repository Hygiene

- Keep scientific source and provenance under version control: analysis/build/audit
  scripts and their tests in `../lmwm/scripts/`, experiment launch and evaluation
  scripts in `../../train_scripts/kai/`, lightweight canonical JSON/Markdown
  evidence in `../lmwm/docs/`, and frozen protocols, launch manifests, and source
  snapshots in `manifests/` and `frozen_sources/`.
- Keep runtime state local: checkpoints, feature arrays, rollout directories,
  scheduler queues, logs, Python caches, browser snapshots, and LaTeX intermediate
  files are ignored because they are large or reproducible.  Promote a lightweight
  result from `logs/` into `docs/`, `manifests/`, or the evidence archive before
  relying on it outside the local workspace.
- The Web report keeps only files linked from its `index.html` plus the current
  manuscript PDF.  Its TODO and evidence-archive copies must be byte-identical to
  the canonical paper files.
- Do not store repository-resident timers, recursive Codex runners, or review
  daemons here.  Paper review scheduling is external; experiment execution remains
  owned by the resource-aware scheduler.
