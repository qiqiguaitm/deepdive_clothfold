# R4 Seed-1000 Fixed-Checkpoint Evidence

Decision: **accepted** under the preregistered macro and task-safety gate.

| Task | Ordinary | Outcome-free CRAVE | Terminal outcome | Terminal - ordinary | Terminal - CRAVE |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 86.0 | 83.5 | 92.0 | +6.0 | +8.5 |
| blocks_ranking_rgb | 93.5 | 91.0 | 94.0 | +0.5 | +3.0 |
| blocks_ranking_size | 57.5 | 61.0 | 65.5 | +8.0 | +4.5 |
| handover_block | 49.0 | 42.0 | 57.5 | +8.5 | +15.5 |
| stack_blocks_three | 67.0 | 58.0 | 67.5 | +0.5 | +9.5 |
| stack_blocks_two | 92.5 | 91.0 | 89.0 | -3.5 | -2.0 |
| **Macro** | **74.2** | **71.1** | **77.6** | **+3.3** | **+6.5** |

Paired hierarchical 95% intervals:
- Terminal outcome minus ordinary: [-2.25, +8.92] points; task safety=True.
- Terminal outcome minus outcome_free_crave: [+0.33, +12.92] points; task safety=True.

Claim boundary: Terminal-outcome and outcome-free CRAVE are sample-weighting signals over expert demonstrations; this evidence does not estimate Q-values, action advantages, a reward model, a world critic, or model-predictive control.
