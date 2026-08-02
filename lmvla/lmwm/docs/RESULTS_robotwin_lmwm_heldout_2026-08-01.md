# RoboTwin LMWM Held-Out Predictor Results

Date: 2026-08-01

## Protocol

Two task-stratified episode-heldout folds were trained independently. Each
fold uses 960 training episodes and holds out 240 episodes across the same six
RoboTwin tasks. Metrics are computed on 256 samples per task and then averaged
over tasks. Persistence predicts the current visual feature as the future
feature and is the required no-change baseline.

Canonical machine-readable result:

- `logs/eval_reports/robotwin_lmwm_heldout_twofold.json`

## Aggregate Results

| Metric | Two-fold mean |
|---|---:|
| Future-feature cosine | 0.8134 |
| Persistence cosine | 0.7479 |
| Predictor lift over persistence | +0.0655 |
| Smooth L1 | 0.0250 |
| Retrieval top-1 | 46.9% |
| Retrieval top-5 | 80.8% |
| Retrieval from the same episode | 76.5% |

The two folds are consistent: predictor lift is +0.0624 and +0.0686, while
top-1 retrieval is 45.0% and 48.9%.

## Task Breakdown

| Task | Predictor cosine | Persistence | Lift | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 0.8133 | 0.7966 | +0.0167 | 57.2% | 91.8% |
| blocks_ranking_rgb | 0.7779 | 0.7316 | +0.0463 | 32.2% | 71.5% |
| blocks_ranking_size | 0.7925 | 0.7251 | +0.0675 | 37.7% | 72.5% |
| handover_block | 0.8311 | 0.7574 | +0.0737 | 70.1% | 88.7% |
| stack_blocks_three | 0.8315 | 0.7399 | +0.0916 | 43.4% | 75.2% |
| stack_blocks_two | 0.8344 | 0.7370 | +0.0974 | 41.0% | 85.0% |

## Interpretation

The LMWM predictor learns episode-general future-feature structure rather than
only memorizing training episodes: it beats persistence on both held-out folds
and retrieves the correct held-out future among task-level candidates well
above chance. The largest predictive lift occurs on stacking, while the lift
on `beat_block_hammer` is small because current features already provide a
strong persistence baseline.

This result establishes predictability, not policy utility. It does not show
that the policy consumes the predicted content or that a more accurate future
feature improves closed-loop success. Those claims remain gated by the strict
fixed-scene hint interventions and the state-dependent demo-retrieval upper
bound. Completed causal controls are currently null, so the paper must retain
the distinction between representation learning and inference-time guidance.
