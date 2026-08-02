# pi0.5 RoboTwin official-aligned matrix (2026-07-31)

## Baseline role

`A0` is the primary VLA baseline for the RoboTwin claim. It starts from the
official `pi05_base` checkpoint and contains no LMWM hint, future objective, or
WM-specific parameter path. A1/A2/A3 are interpreted only as matched pi0.5
extensions relative to this A0 row. The separate LaWAM `no-WM` row is not a pure
VLA baseline because it inherits WM pretraining; it is reported as
`LaWAM-init / Future-off` in mechanism analyses.

## Protocol

- Tasks: `beat_block_hammer`, `blocks_ranking_size`, `blocks_ranking_rgb`,
  `handover_block`, `stack_blocks_two`, `stack_blocks_three`.
- `demo_clean`, unseen instructions, one simulator slot, replan interval 50.
- Four simulator seeds, 50 episodes per task and seed.
- Training follows the project pi0.5 official-aligned 20k-step configuration.

These numbers are directly comparable within this matrix. They are not yet evidence of direct
comparability with externally reported RoboTwin scores, whose task set, camera/randomization,
checkpoint and evaluation bridge must be matched separately.

## A0 no-hint baseline

- Training: `pi05-robotwin-a0-official-bj`
  (`t-20260730092737-597xp`).
- Evaluation: `pi05-robotwin-a0-official-eval-v2-x4`
  (`t-20260731230433-npm45`).
- Result root: `pi05_rt_a0_official_v2` on North vePFS.
- Completion: 24/24 summaries, 1200 episodes total.

| Task | Seed SR (%) | Mean SR (%) |
|---|---:|---:|
| beat_block_hammer | 70, 76, 66, 68 | 70.0 |
| blocks_ranking_size | 26, 28, 30, 12 | 24.0 |
| blocks_ranking_rgb | 54, 54, 50, 44 | 50.5 |
| handover_block | 10, 8, 10, 8 | 9.0 |
| stack_blocks_two | 40, 42, 46, 40 | 42.0 |
| stack_blocks_three | 14, 18, 18, 20 | 17.5 |
| **Six-task macro** | | **35.50** |

This closes the internal no-hint baseline but not the external-protocol question. The score is far
below public RoboTwin pi0.5 reports, so those reports must not be cited as a direct baseline until
the exact public checkpoint, task assets, camera/randomization settings, action conversion and
evaluation bridge are reproduced together.

## A1 absolute prefix

- Training: `pi05-robotwin-a1-prefix-official-bj`
  (`t-20260730083819-x7l5s`).
- Result root: `pi05_rt_a1_prefix_official_v2` on North vePFS.
- Completion: 24/24 summaries, 1200 episodes total.

| Task | Seed SR (%) | Mean SR (%) |
|---|---:|---:|
| beat_block_hammer | 84, 76, 78, 80 | 79.5 |
| blocks_ranking_size | 30, 32, 20, 16 | 24.5 |
| blocks_ranking_rgb | 50, 54, 52, 54 | 52.5 |
| handover_block | 6, 6, 4, 6 | 5.5 |
| stack_blocks_two | 70, 62, 68, 52 | 63.0 |
| stack_blocks_three | 20, 22, 26, 20 | 22.0 |
| **Six-task macro** | | **41.17** |

A1 exceeds A0 by `+5.67 pp`, driven mainly by hammer and stack-two, but remains
`-7.58 pp` below A2 absolute and `-8.41 pp` below A3. Its handover result is
lower than A0 and stack-three improves only modestly. A simple shallow absolute
prefix therefore helps some tasks but does not explain the stronger A2/A3 gains.

## A2 absolute prefix

- Training: `pi05-robotwin-a2-prefix-official-bj`
  (`t-20260730151045-4znrf`).
- Evaluation: `pi05-robotwin-a2-absolute-official-eval-v2-x4`
  (`t-20260731230549-tvdn5`).
- Result root: `pi05_rt_a2_prefix_official_v2` on North vePFS.
- Completion: 24/24 summaries, 1200 episodes total.

| Task | Seed SR (%) | Mean SR (%) |
|---|---:|---:|
| beat_block_hammer | 82, 82, 74, 74 | 78.0 |
| blocks_ranking_size | 38, 18, 34, 32 | 30.5 |
| blocks_ranking_rgb | 68, 60, 54, 78 | 65.0 |
| handover_block | 20, 24, 14, 20 | 19.5 |
| stack_blocks_two | 64, 66, 78, 68 | 69.0 |
| stack_blocks_three | 28, 40, 22, 32 | 30.5 |
| **Six-task macro** | | **48.75** |

A2 absolute exceeds A2 residual by `+4.92 pp` macro. Residual improves ranking-size by `+4.0 pp`
but loses `12.5 pp` on ranking-RGB and `12.5 pp` on stack-two. Under the current implementation,
forming a residual target in policy-prefix space is therefore not an improvement over the absolute
prefix. A2 absolute exceeds A0 by `+13.25 pp` under this identical evaluator.

## A2 residual prefix

- Training: `pi05-robotwin-a2-res-prefix-official-bj`
  (`t-20260730225631-65trh`).
- Evaluation: `pi05-robotwin-a2-residual-official-eval-v4-x4`
  (`t-20260731221831-4wgst`).
- Result root: `pi05_rt_a2_residual_prefix_official_v4` on North vePFS.
- Completion: 24/24 summaries, 1200 episodes total.

| Task | Seed SR (%) | Mean SR (%) |
|---|---:|---:|
| beat_block_hammer | 78, 76, 76, 76 | 76.5 |
| blocks_ranking_size | 36, 36, 30, 36 | 34.5 |
| blocks_ranking_rgb | 58, 50, 46, 56 | 52.5 |
| handover_block | 12, 30, 20, 10 | 18.0 |
| stack_blocks_two | 64, 54, 58, 50 | 56.5 |
| stack_blocks_three | 22, 38, 18, 22 | 25.0 |
| **Six-task macro** | | **43.83** |

The residual arm is therefore not sufficient to close the pi0.5 RoboTwin gap. The largest
failures are `handover_block` and `stack_blocks_three`; it is also `-4.92 pp` below A2 absolute
under the identical evaluator.

## A3 live visual-space residual

- Training: `pi05-robotwin-a3-live-resid-official-east`
  (`t-20260730084422-4lb46`).
- Evaluation: gf1, eight parallel one-slot simulator workers.
- Result root: `pi05_rt_a3_live_residual_official_gf1_8g` on shared vePFS.
- Completion: 24/24 summaries, 1200 episodes total.

| Task | Seed SR (%) | Mean SR (%) |
|---|---:|---:|
| beat_block_hammer | 64, 66, 72, 62 | 66.0 |
| blocks_ranking_size | 22, 14, 40, 26 | 25.5 |
| blocks_ranking_rgb | 76, 70, 72, 64 | 70.5 |
| handover_block | 24, 38, 20, 22 | 26.0 |
| stack_blocks_two | 68, 76, 74, 78 | 74.0 |
| stack_blocks_three | 26, 46, 40, 30 | 35.5 |
| **Six-task macro** | | **49.58** |

A3 exceeds A0 by `+14.08 pp`, A2 residual by `+5.75 pp`, and A2 absolute by `+0.83 pp`. Relative to A2
absolute, the improvement is concentrated in
`blocks_ranking_rgb`, handover and both stacking tasks; hammer and ranking-size regress. This is
evidence that recomputing the milestone target in the current visual-encoder space is more useful
than the current A2 residual-prefix implementation. The small macro margin over A2 absolute is not
yet enough to establish a robust overall improvement; independent training seeds are still
required.

## Closed task-level comparisons while full rows run

The following cells already contain all four simulator seeds and are valid task-level comparisons;
they are not substituted for the pending six-task macro.

| Arm | Hammer | Ranking size | Ranking RGB | Handover |
|---|---:|---:|---:|---:|
| A0 | 70.0 | 24.0 | 50.5 | 9.0 |
| A1 absolute | 79.5 | 24.5 | 52.5 | 5.5 |
| A2 absolute | 78.0 | 30.5 | 65.0 | 19.5 |
| A2 residual | 76.5 | 34.5 | 52.5 | 18.0 |
| A3 live residual | 66.0 | 25.5 | 70.5 | 26.0 |

The low A0 result is therefore not explained by a globally broken evaluator: within the identical
bridge, A2 absolute improves handover by `+10.5 pp` and ranking-RGB by `+14.5 pp`, while A3 reaches
`+17.0 pp` and `+20.0 pp` respectively. A3 changes by `-4.0 pp` on hammer and only `+1.5 pp` on
ranking-size relative to A0, confirming a task-dependent tradeoff rather than uniform scaling.

The completed `stack_blocks_two` cells are A0 `42.0`, A1 `63.0`, A2 absolute `69.0`, A2
residual `56.5`, and A3 `74.0` percent. This task accounts for a substantial part of the gains over
A0, while handover remains low for every arm.

## Matrix status at 20:08 UTC

| Arm | Completed summaries | State |
|---|---:|---|
| A0 official | 24/24 | complete, 35.50% macro |
| A1 absolute prefix | 24/24 | complete, 41.17% macro |
| A2 absolute prefix | 24/24 | complete, 48.75% macro |
| A2 residual prefix | 24/24 | complete |
| A3 live residual | 24/24 | complete, 49.58% macro |

All five internal rows are complete. Their ranking is A3 `49.58`, A2 absolute
`48.75`, A2 residual `43.83`, A1 `41.17`, and A0 `35.50` percent macro. This
ranking is valid only inside the shared evaluation bridge; the public-checkpoint
same-bridge run is tracked in `RESULTS_pi05_public_samebridge_2026-07-31.md`. It reaches
`75.0%` macro on the identical six-task bridge and narrows the external gap diagnosis to
training/checkpoint differences rather than a globally broken evaluator.
