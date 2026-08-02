# Public pi0.5 RoboTwin checkpoint on the internal evaluation bridge

## Purpose

This run separates checkpoint quality from evaluator/protocol effects by evaluating the public
`SidneyXie/pi05_robotwin` checkpoint with the same six RoboTwin tasks and action bridge used for
the internal pi0.5 A0--A3 matrix.

## Protocol

- Checkpoint: `/vePFS/tim/hf_models/SidneyXie_pi05_robotwin`.
- Job: `pi05-public-samebridge-v2-bj4g` (`t-20260801063017-wt7mw`).
- Tasks: `beat_block_hammer`, `blocks_ranking_size`, `blocks_ranking_rgb`,
  `handover_block`, `stack_blocks_two`, and `stack_blocks_three`.
- `demo_clean`, unseen instructions, one simulator slot, replan interval 50, no action ensemble.
- One evaluation seed and 20 accepted episodes per task (120 episodes total).
- Completion: 6/6 task summaries on North vePFS.

This is a same-bridge diagnostic, not a replacement for the internal matrix's four simulator
seeds and 50 episodes per task/seed.

## Results

| Task | Successes | Public SR | Internal A0 SR | Difference |
|---|---:|---:|---:|---:|
| beat block hammer | 18/20 | 90.0% | 70.0% | +20.0 pp |
| blocks ranking size | 10/20 | 50.0% | 24.0% | +26.0 pp |
| blocks ranking RGB | 19/20 | 95.0% | 50.5% | +44.5 pp |
| handover block | 6/20 | 30.0% | 9.0% | +21.0 pp |
| stack blocks two | 19/20 | 95.0% | 42.0% | +53.0 pp |
| stack blocks three | 18/20 | 90.0% | 17.5% | +72.5 pp |
| **Six-task macro** | **90/120** | **75.0%** | **35.5%** | **+39.5 pp** |

## Interpretation

The internal evaluator is capable of producing high success rates: the public checkpoint reaches
90--95% on hammer, ranking-RGB, and both stacking tasks without changing the bridge. The low
internal A0 score therefore cannot be attributed to a globally broken renderer, action conversion,
or success detector.

The public checkpoint is not uniformly near 90%. Ranking-size is 50% and handover is 30%, so a
single externally reported score near 90% can depend strongly on the selected task panel and
aggregation protocol. Direct leaderboard comparison still requires matching the exact official
task set, episode count, randomization, and aggregation.

The `+39.5 pp` same-bridge gap points primarily to training/checkpoint differences. The largest
gaps are stack-three, stack-two, and ranking-RGB. These tasks should be used first when comparing
the internal data mixture, observation/action normalization, and training schedule against the
public recipe.

Because the public run has only 20 episodes per task from one evaluation seed, task-level
differences should be treated as diagnostic effect sizes rather than final confidence intervals.
The internal A0--A3 ranking remains defined by the larger 1,200-episode-per-arm protocol in
`RESULTS_pi05_robotwin_official_matrix_2026-07-31.md`.

## Four-seed confirmation

The matched `4 seeds x 6 tasks x 50 episodes` run completed as
`pi05-public-samebridge-4seed-v3-bj4g` (`t-20260801071323-vxtwt`), with 24/24
task-seed summaries and 1,200 accepted episodes:

| Task | Seed SR (%) | Mean SR (%) | Episodes |
|---|---:|---:|---:|
| beat block hammer | 90, 92, 96, 94 | **93.0** | 200 |
| blocks ranking size | 56, 66, 64, 70 | **64.0** | 200 |
| blocks ranking RGB | 98, 98, 96, 94 | **96.5** | 200 |
| handover block | 58, 58, 40, 58 | **53.5** | 200 |
| stack blocks two | 90, 94, 96, 94 | **93.5** | 200 |
| stack blocks three | 72, 68, 64, 76 | **70.0** | 200 |
| **Six-task macro/micro** | - | **78.42** | **1,200** |

The run produced 941 successes. Hammer wall time was 637--698 seconds per seed;
ranking-size took about 36--41 minutes per seed, and stack-three about 34--37
minutes. The larger run confirms the diagnostic conclusion while correcting its
optimistic stack-three estimate from 90% to 70%.

The public checkpoint is near 90% only on hammer, ranking-RGB, and stack-two.
Its six-task macro is 78.42%, with ranking-size, handover, and stack-three at
64.0%, 53.5%, and 70.0%. Claims that pi0.5 is "around 90% on RoboTwin" therefore
depend on task selection and aggregation; they are not reproduced as an
all-six-task macro under this protocol.

## Systems observations

The first diagnostic request spent several minutes compiling the policy and exceeded the default
websocket keepalive timeout. Disabling websocket ping timeouts and preserving the TorchInductor
cache made the formal run stable. The earlier 20-episode diagnostic task wall times ranged from
about 5.8 minutes (hammer) to 17.9 minutes (ranking-size). Future multi-GPU runs should shard
episodes rather than assigning one
whole task per worker, because the six-task/four-GPU layout leaves tail GPUs idle.
