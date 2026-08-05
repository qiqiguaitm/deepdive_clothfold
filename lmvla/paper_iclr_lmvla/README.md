# MINT-VLA -> ICLR 2027 Paper

## Current Story

- Method: MINT-VLA stores mined milestone frames, re-encodes targets with the current VLA visual encoder during training, predicts an absolute milestone through a residual parameterization, and connects it to the VLA's native conditioning input. The current pi0.5 instantiation uses one prefix token.
- Primary VLA/benchmark: pi0.5 on RoboTwin 2.0. LIBERO and LaWAM are supporting diagnostics.
- Structure: problem and controlled comparisons -> MINT-VLA -> confirmatory design -> completed three-seed negative result and interface diagnosis.
- Questions: Q1 matched control performance; Q2 predicted token content; Q3 factorized integration mechanism; Q4 robustness, task scope, and efficiency. Q5 second-VLA instantiation is explicitly closed because the draft makes no multi-architecture claim.

## Evidence Boundary

- The completed A0--A3 matrix is a legacy matched pilot: joint-delta actions, quantile normalization, batch 64, 20k updates, one training seed per arm.
- Pilot macros: A0 35.50, A2-Abs 48.75, A3 49.58. A3-A2-Abs is only +0.83 pp.
- The formal public pi0.5 same-bridge calibration is 78.42% (941/1,200), showing the pilot baseline is weak.
- The corrected absolute-action/mean-std/batch-16/50k matrix is complete across training seeds 1000--1002: A0 79.39%, A2-Abs 75.44%, and A3 70.81%.
- Paired hierarchical differences versus A0 are -3.94 pp for A2-Abs (95% CI [-6.58,-0.97]) and -8.58 pp for A3 ([-11.47,-5.75]); the paper therefore makes a negative integration result, not a method-benefit claim.
- A3 is observation-conditioned and not fully gradient-isolated. LaWAM Future-off is not a clean no-WM VLA baseline.

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
