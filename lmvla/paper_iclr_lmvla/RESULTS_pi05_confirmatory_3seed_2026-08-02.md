# pi0.5 RoboTwin Confirmatory Three-Seed Result

Status: complete and protocol-audited on 2026-08-02.

## Protocol

- Arms: A0 no hint, A2-Abs offline absolute hint, A3 live current-encoder residual.
- Training seeds: 1000, 1001, 1002.
- Evaluation: six RoboTwin tasks, four evaluator seeds, 50 episodes per cell.
- Evidence volume: 24 cells and 1,200 episodes per trained policy; 10,800 total episodes.
- Scene manifest SHA256: `08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`.
- All nine reports pass exact cell coverage, 50-episode coverage, scene-key matching,
  and launch/source/data provenance audits.

## Macro Result

| Arm | Seed 1000 | Seed 1001 | Seed 1002 | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| A0 | 75.50 | 80.00 | 82.67 | 79.39 | 2.96 |
| A2-Abs | 70.00 | 74.67 | 81.67 | 75.44 | 4.79 |
| A3 | 66.92 | 68.92 | 76.58 | 70.81 | 4.17 |

Paired hierarchical contrasts, resampling training seeds and then paired scenes
within task:

| Contrast | Delta (pp) | 95% interval (pp) | Paired episodes | Unmatched keys |
|---|---:|---:|---:|---:|
| A2-Abs - A0 | -3.94 | [-6.58, -0.97] | 3,600 | 0 |
| A3 - A0 | -8.58 | [-11.47, -5.75] | 3,600 | 0 |

Per-training-seed deltas are A2-Abs: -5.50/-5.33/-1.00 pp and A3:
-8.58/-11.08/-6.08 pp. Every paired training-seed delta is negative.

## Task Means

| Task | A0 | A2-Abs | A2 delta | A3 | A3 delta |
|---|---:|---:|---:|---:|---:|
| Beat block hammer | 89.50 | 86.33 | -3.17 | 80.83 | -8.67 |
| Blocks ranking RGB | 95.00 | 89.17 | -5.83 | 88.50 | -6.50 |
| Blocks ranking size | 61.83 | 55.83 | -6.00 | 48.83 | -13.00 |
| Handover block | 59.83 | 54.33 | -5.50 | 51.00 | -8.83 |
| Stack blocks three | 77.33 | 73.67 | -3.67 | 65.67 | -11.67 |
| Stack blocks two | 92.83 | 93.33 | +0.50 | 90.00 | -2.83 |

## Interpretation

The corrected official-aligned pi0.5 baseline removes the apparent milestone
utility. A2-Abs has one small positive task mean (stack-two) but a significantly
negative macro contrast; A3 regresses every task. Existing predictor-quality
and representation results therefore cannot be interpreted as closed-loop
control utility. T4--T6 are closed by the preregistered positive-utility gate.

Authoritative machine-readable result:
`logs/eval_reports/pi05_confirmatory_training_seed_matrix.json`.
