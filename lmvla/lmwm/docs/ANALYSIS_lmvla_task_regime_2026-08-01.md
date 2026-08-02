# LMVLA task-regime analysis (2026-08-01)

## Comparison that the current matrix supports

The LaWAM testbed contains three distinct questions that must not be merged:

1. **Active LaWAM to LMWM:** `local-WM` retains the fixed-near-future LaWAM
   objective. Absolute, residual, isolation, and combo retain that path and add
   milestone conditioning. These are the direct incremental LMWM comparisons.
2. **Active future training to Future-off:** `Future-off` disables downstream
   future prediction and distillation, but inherits the same LaWAM-pretrained
   initialization. This comparison tests whether an active future objective
   should remain during downstream fine-tuning.
3. **Pure VLA to LM-informed VLA:** this is not identified by the completed
   matrix because Future-off already contains LaWAM pretraining. It requires a
   clean, budget-matched VLA baseline.

The completed training-seed means are:

| Arm | Seeds | Macro (%) | Delta from local-WM (pp) | Delta from Future-off (pp) |
|---|---:|---:|---:|---:|
| LaWAM-init / Future-off | 3 | **90.83** | +4.17 | -- |
| local-WM | 2 | 86.67 | -- | -4.17 |
| absolute milestone | 2 | 89.21 | +2.54 | -1.63 |
| residual milestone | 2 | 87.50 | +0.83 | -3.33 |
| residual + isolation | 2 | 89.04 | +2.38 | -1.79 |
| isolation | 1 | 89.42 | +2.75 | -1.42 |

Thus LMWM is not behaviorally inert: absolute and combo recover `+2.54` and
`+2.38 pp` over active local-WM. However, no replicated milestone arm exceeds
Future-off. The defensible statement is that milestone conditioning improves
the active LaWAM interface on selected tasks, while current evidence does not
show that keeping a future objective active during downstream fine-tuning is
better than turning it off. The broader claim that latent-model information is
or is not useful to a VLA remains unresolved without the clean VLA baseline.

## Task-level decomposition

Two-seed task means show where LMWM recovers the active local-WM deficit:

| Task | Future-off | local-WM | Absolute - local | Residual - local | Combo - local | Combo - Future-off |
|---|---:|---:|---:|---:|---:|---:|
| beat block with hammer | 95.17 | 92.25 | -1.50 | -0.50 | 0.00 | -2.92 |
| stack two blocks | 100.00 | 100.00 | -0.25 | -0.50 | -0.50 | -0.50 |
| stack three blocks | 89.50 | 89.50 | +6.50 | **+8.75** | +7.00 | **+7.00** |
| rank blocks by RGB | 99.17 | 94.50 | +0.25 | -10.00 | +2.50 | -2.17 |
| rank blocks by size | 82.67 | 74.00 | +6.25 | +6.00 | +4.25 | -4.42 |
| handover block | 78.50 | 69.75 | +4.00 | +1.25 | +1.00 | -7.75 |

The low-success tasks do not obey one simple rule. LMWM recovers part of the
local-WM loss on ranking-size and handover, but still remains below Future-off;
only stack-three converts that recovery into a net improvement over
Future-off. Headroom is therefore relevant but insufficient.

Using `100 - Future-off success` as a headroom proxy gives an exploratory
Spearman correlation of `rho=0.486` (`p=0.329`) with combo-minus-local
recovery. Its correlation with combo-minus-Future-off net benefit is
`rho=-0.657` (`p=0.156`). Neither is significant across six tasks. Lower
baseline success therefore does not by itself predict a net LMWM gain.

## Completed duration and milestone analysis

The frozen 1,200-episode milestone artifact permits a result-independent task
attribute audit. Pair counts equal the usable frame counts in the selected
demonstrations.

| Task | Mean frames | Mean target horizon | Mean distinct targets | Combo - local (pp) |
|---|---:|---:|---:|---:|
| hammer | 114.1 | 53.6 | 1.30 | 0.00 |
| handover | 284.3 | 111.7 | 1.97 | +1.00 |
| stack two | 313.2 | 114.8 | 2.82 | -0.50 |
| ranking RGB | 459.2 | 120.0 | 4.15 | +2.50 |
| ranking size | 459.7 | 122.8 | 4.00 | +4.25 |
| stack three | 470.7 | 130.4 | 3.82 | +7.00 |

Across only six post-hoc task points, mean duration correlates with the
absolute-minus-local and combo-minus-local deltas (Spearman `rho=0.829`,
unadjusted `p=0.042` for both). This is hypothesis-generating, not
confirmatory: duration, horizon, and stage count are correlated; task outcomes
are heterogeneous; and no task-regime hypothesis was preregistered before the
matrix was inspected. In particular, the long ranking-RGB task loses
`10.00 pp` under residual-only, while stack-two is saturated and cannot reveal
content utility. Duration alone cannot explain the effect.

## Insufficient hint or harmful hint?

The completed fixed-checkpoint interventions already separate part of this
question:

- For absolute milestones, correct, zero, shuffled, and cross-task hints are
  statistically indistinguishable on hammer, ranking-RGB, and stack-two. This
  is consistent with weak use or insufficient decision information, not proof
  that the hint is wrong.
- For residual-only, shuffled and cross-task hints outperform the correct hint
  on the pooled matched cohort (`+2.00 pp`, `p=0.0347`; `+2.18 pp`,
  `p=0.0227`). This is direct evidence of harmful conditioning in that route.
- With gradient isolation, incorrect hints become numerically worse than the
  correct hint; ranking-RGB shows a local correct-versus-cross-task advantage
  of `+3.57 pp` (`p=0.0156`). The reversal implicates the integration and
  gradient route, rather than task duration alone.

Current evidence therefore supports a mixed diagnosis: some checkpoints do
not use the predicted content detectably, while residual-only can turn that
content into an actively harmful condition. It does not support the blanket
claim that short tasks fail because their milestone predictions are wrong.
The existing intervention panel covers only three tasks, so a matched six-task
panel is still required to test a duration-by-condition interaction.

## Predeclared validation panel

The task groups are frozen before new confirmatory results:

| Group | Tasks | Role |
|---|---|---|
| ordered construction | `stack_blocks_two`, `stack_blocks_three`, `stack_bowls_two`, `stack_bowls_three` | target regime |
| reactive/contact | `beat_block_hammer`, `click_bell`, `stamp_seal` | negative control |
| fine-grained geometry | `blocks_ranking_size`, `place_object_scale` | discriminability control |
| relational transfer | `handover_block`, `handover_mic` | neighboring regime |

A scoped claim requires independent training seeds, replication on both block
and bowl stacking, an advantage over fixed horizon, and a group-by-method
interaction. Until then, task duration and ordered construction remain
exploratory explanations rather than paper claims.
