# RoboTwin all6 v2 method matrix (updated 2026-08-01)

## Protocol

- Three training seeds for Future-off; two for local-WM, absolute, residual,
  and combo; one for isolation.
- Six balanced RoboTwin tasks, `demo_clean`, unseen instructions.
- Four simulator seeds per checkpoint and 50 valid episodes per task/seed.
- Every cell below averages 200 closed-loop episodes. `Macro` averages the six task rates.

## Training-seed means

| Method | beat hammer | ranking RGB | ranking size | handover | stack three | stack two | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| LaWAM-init / Future-off | **95.17** | **99.17** | **82.67** | **78.50** | 89.50 | **100.00** | **90.83** |
| local-WM | 92.25 | 94.50 | 74.00 | 69.75 | 89.50 | **100.00** | 86.67 |
| absolute milestone | 90.75 | 94.75 | 80.25 | 73.75 | 96.00 | 99.75 | 89.21 |
| residual milestone | 91.75 | 84.50 | 80.00 | 71.00 | **98.25** | 99.50 | 87.50 |
| gradient isolation (one seed) | 94.00 | 93.00 | 79.00 | 78.00 | 92.50 | **100.00** | 89.42 |
| residual + isolation | 92.25 | 97.00 | 78.25 | 70.75 | 96.50 | 99.50 | 89.04 |

Macro deltas relative to Future-off are: local-WM `-4.17 pp`, absolute
`-1.63 pp`, residual `-3.33 pp`, isolation `-1.42 pp` (one seed), and combo
`-1.79 pp`. Relative to active local-WM, absolute and combo recover `+2.54 pp`
and `+2.38 pp`.

## Current verdict

Absolute is nearly identical across seeds 2026 and 2027 (`89.17%` and
`89.25%`), and combo and local-WM are similarly stable. Future-off scores
`91.08%`, `90.92%`, and `90.50%` across three seeds. The macro ordering is
therefore unlikely to be an isolated initialization effect.

LMWM is not inert: compared with active local-WM, combo improves stack-three
by `+7.00 pp`, ranking-size by `+4.25 pp`, ranking-RGB by `+2.50 pp`, and
handover by `+1.00 pp`. Only stack-three also exceeds Future-off. The matrix
supports selective recovery of the active LaWAM interface, but it does not
show a replicated net gain over disabling downstream future training.

Future-off inherits LaWAM pretraining, so this result cannot determine whether
latent-model pretraining benefits a clean VLA. Isolation is the only supporting
arm still missing a second training seed.

## Causal intervention results

The absolute-checkpoint zero-hint run completed all three selected tasks and all
four evaluation seeds on 2026-07-31 (`12/12` summaries). RoboTwin can reject
unstable initial scenes, so inference interventions do not always retain the
same accepted simulator seeds. The table therefore uses only seed IDs common to
the predicted-milestone and zero-milestone runs.

| Task | Common episodes | Correct SR | Zero SR | Zero - correct | Correct-only / zero-only | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| beat_block_hammer | 174 | 94.25% | 90.80% | -3.45 pp | 10 / 4 | 0.180 |
| blocks_ranking_rgb | 198 | 93.94% | 92.93% | -1.01 pp | 10 / 8 | 0.815 |
| stack_blocks_two | 199 | 99.50% | 99.50% | 0.00 pp | 1 / 1 | 1.000 |
| **Pooled** | **571** | **95.97%** | **94.57%** | **-1.40 pp** | **21 / 13** | **0.229** |

Zeroing the predicted milestone therefore causes no statistically significant
closed-loop degradation in this panel. The hammer direction is compatible with
a small task-specific contribution, but the pooled evidence does not support
the milestone content being the main inference-time decision source. The
completed distribution-preserving controls below provide the stronger test.

The shuffled-hint and other-task runs also completed all three selected tasks
and all four evaluation seeds (`12/12` summaries each). The other-task hint is
drawn from `handover` episode 4400. This avoids the previous ambiguous
`wrong_*` artifact, which came from `stack_blocks_two` and was therefore not
cross-task for one of the selected tasks.

The table below restricts every row to episode seeds common to all four
conditions, making the controls directly paired.

| Task | Common episodes | Correct | Zero | Shuffled | Other-task |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 143 | 95.80% | 93.71% | 97.20% | 96.50% |
| blocks_ranking_rgb | 196 | 93.88% | 92.86% | 95.41% | 94.39% |
| stack_blocks_two | 199 | 99.50% | 99.50% | 99.50% | 98.99% |
| **Pooled** | **538** | **96.47%** | **95.54%** | **97.40%** | **96.65%** |

Relative to correct milestones, pooled deltas are zero `-0.93 pp`, shuffled
`+0.93 pp`, and other-task `+0.19 pp`. Correct-only/control-only discordant
counts are `18/13`, `9/14`, and `14/15`; exact paired McNemar p-values are
`0.473`, `0.405`, and `1.000`, respectively. None provides evidence that the
correct predicted content has decision-time value. The distribution-preserving
foreign controls are numerically no worse, and stack-two is effectively
saturated in all conditions. The completed pre-registered panel therefore
favors training-time representation shaping over inference-time milestone
guidance for this checkpoint.

The paired analysis is reproducible with
`lmvla/lmwm/scripts/rt_causal_intervention_analysis.py`. With multiple controls,
the script defaults to the episode cohort common to every condition; `--pairwise`
is available only for two-condition diagnostics.

### Residual-checkpoint causal intervention result

All zero, shuffled, and other-task controls completed the pre-registered
three-task panel (`12/12` summaries per control). The table uses the same `550`
episodes available in every condition, rather than a different pairwise cohort
for each comparison.

| Task | Common episodes | Correct | Zero | Shuffled | Other-task |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 157 | 89.81% | 91.72% | 94.27% | 93.63% |
| blocks_ranking_rgb | 194 | 85.05% | 86.60% | 86.60% | 87.63% |
| stack_blocks_two | 199 | 99.50% | 100.00% | 100.00% | 100.00% |
| **Pooled** | **550** | **91.64%** | **92.91%** | **93.64%** | **93.82%** |

Relative to correct residual milestones, pooled deltas are zero `+1.27 pp`,
shuffled `+2.00 pp`, and other-task `+2.18 pp`. Correct-only/control-only
discordant counts are `8/15`, `6/17`, and `6/18`; exact paired McNemar p-values
are `0.210`, `0.0347`, and `0.0227`, respectively.

For this residual checkpoint, correct predicted milestone content therefore
does not provide inference-time decision value. Both distribution-preserving
incorrect controls are significantly better on the pooled shared cohort. This
supports a harmful-conditioning diagnosis for the current residual injection,
while leaving open whether the checkpoint's training-time auxiliary objective
improves representation learning.

### Residual-plus-isolation combo causal intervention result

The combo checkpoint's zero, shuffled, and other-task controls also completed
the full panel (`12/12` summaries per control). Restricting all four conditions
to one shared cohort gives:

| Task | Common episodes | Correct | Zero | Shuffled | Other-task |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 144 | 97.22% | 95.14% | 97.22% | 97.22% |
| blocks_ranking_rgb | 196 | 97.96% | 95.41% | 94.90% | 94.39% |
| stack_blocks_two | 198 | 99.49% | 99.49% | 100.00% | 100.00% |
| **Pooled** | **538** | **98.33%** | **96.84%** | **97.40%** | **97.21%** |

Relative to correct combo milestones, pooled deltas are zero `-1.49 pp`,
shuffled `-0.93 pp`, and other-task `-1.12 pp`. Correct-only/control-only
discordant counts are `13/5`, `9/4`, and `8/2`; exact paired McNemar p-values
are `0.0963`, `0.267`, and `0.109`, respectively. The pooled comparisons do not
cross the 0.05 threshold. On ranking-RGB, however, correct beats the other-task
control by `3.57 pp` with `7/0` discordant episodes (`p=0.0156`).

Gradient isolation therefore reverses the harmful-conditioning pattern seen in
the residual-only checkpoint: both distribution-preserving incorrect controls
change from significantly better than correct to numerically worse. This is
evidence that isolation protects the policy from harmful residual conditioning.
It is not yet a general pooled significance result for inference-time guidance;
the positive decision-content claim is currently task-specific to ranking-RGB.
