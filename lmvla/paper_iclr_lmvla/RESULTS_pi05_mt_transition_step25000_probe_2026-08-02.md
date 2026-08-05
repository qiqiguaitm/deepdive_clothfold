# pi0.5 Milestone-Transition Step-25k Probe

Date: 2026-08-02

This is a five-scene trajectory diagnostic on `beat_block_hammer`. It is not a
confirmatory result and does not replace the frozen 24-cell evaluation.

## Matched outcomes

| Frozen scene seed | MT1 correct | MT1 null | A0 step 25k |
|---:|---:|---:|---:|
| 100000 | 0 | 0 | 0 |
| 100002 | 1 | 1 | 0 |
| 100003 | 0 | 1 | 1 |
| 100007 | 1 | 1 | 0 |
| 100008 | 1 | 0 | 1 |
| **Success** | **3/5** | **3/5** | **2/5** |

MT1 correct versus MT1 null has one discordant win in each direction, hence
zero paired net gain. Correct and null each gain two scenes and lose one scene
relative to the checkpoint-age-matched A0 control, for a net gain of one scene.

## Interpretation

At step 25,000, the transition-conditioned training path may recover faster
than A0 on this small diagnostic, but transition content has no demonstrated
utility: correct and null conditioning are tied in aggregate and in paired net
outcome. The scene-level swaps also show why five scenes are inadequate for a
branch decision. No hyperparameter or protocol change follows from this probe.
The predeclared final correct, null-input, within-task permutation,
cross-task, and parameter-matched null-trained evaluations remain the gate.
