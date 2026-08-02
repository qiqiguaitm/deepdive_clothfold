# MINT-VLA -> ICLR 2027 Paper

## Current Story

- Method: MINT-VLA stores mined milestone frames, re-encodes targets with the current VLA visual encoder during training, predicts an absolute milestone through a residual parameterization, and connects it to the VLA's native conditioning input. The current pi0.5 instantiation uses one prefix token.
- Primary VLA/benchmark: pi0.5 on RoboTwin 2.0. LIBERO and LaWAM are supporting diagnostics.
- Structure: problem and controlled comparisons -> MINT-VLA -> confirmatory design -> completed pilot and open gates.
- Questions: Q1 matched control performance; Q2 predicted token content; Q3 factorized integration mechanism; Q4 robustness, task scope, and efficiency; Q5 instantiation on a second VLA architecture.

## Evidence Boundary

- The completed A0--A3 matrix is a legacy matched pilot: joint-delta actions, quantile normalization, batch 64, 20k updates, one training seed per arm.
- Pilot macros: A0 35.50, A2-Abs 48.75, A3 49.58. A3-A2-Abs is only +0.83 pp.
- The formal public pi0.5 same-bridge calibration is 78.42% (941/1,200), showing the pilot baseline is weak.
- Final claims depend on corrected absolute-action/mean-std/batch-16/50k A0/A2-Abs/A3 runs, content interventions, factorized ablations, and independent training seeds.
- A3 is observation-conditioned and not fully gradient-isolated. LaWAM Future-off is not a clean no-WM VLA baseline.

## Files

- `main.tex`: paper entry point.
- `sections/`: manuscript sections.
- `numbers.tex`: centralized numeric macros.
- `PAPER_TODO.md`: dependency-ordered experiment and writing plan.
- `../lmwm/docs/PROGRESS_pi05_vla_baseline_2026-08-01.md`: latest baseline/causal audit.

Build with `make`. The draft uses the ICLR 2026 style until the ICLR 2027 template is released.
