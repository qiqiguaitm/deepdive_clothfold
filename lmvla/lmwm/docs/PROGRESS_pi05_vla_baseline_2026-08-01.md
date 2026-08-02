# pi0.5-centered RoboTwin evidence plan (2026-08-01)

## Evidence hierarchy

1. **External performance anchor: published pi0.5 RoboTwin.** This checkpoint is
   the direct pi0.5 reference under its published 50k-step, absolute-action,
   mean/std recipe.
2. **Primary matched VLA baseline: pi0.5 A0-public-recipe.** It starts from
   `pi05_base`, has no LMWM path, and reproduces the published recipe in the
   internal OpenPI stack. This is the baseline for all new controlled claims.
3. **Controlled WM matrix: matched pi0.5 A1/A2/A3.** These arms must share the
   new A0's data,
   initialization, action horizon, batch size, training steps, and evaluator. They
   test increasingly integrated milestone conditioning while keeping the VLA
   architecture fixed.
4. **Mechanism diagnostics: LaWAM matrix.** The former `no-WM` row is named
   **LaWAM-init / Future-off** because it inherits WM pretraining. It can diagnose
   downstream objectives, residualization, isolation, and hint interventions, but
   it is not evidence for WM versus a pure VLA.

## Closed results

The completed **legacy internal pilot** contains four simulator seeds, 50 episodes
per task and seed, and six tasks (1,200 episodes per arm):

| Arm | Six-task macro | Delta from A0 |
|---|---:|---:|
| pi0.5 A0, no hint | 35.50% | - |
| pi0.5 A1, absolute prefix | 41.17% | +5.67 pp |
| pi0.5 A2, absolute prefix | 48.75% | +13.25 pp |
| pi0.5 A2, residual prefix | 43.83% | +8.33 pp |
| pi0.5 A3, live visual-space residual | 49.58% | +14.08 pp |

The pilot indicates that milestone conditioning can improve a matched pi0.5
checkpoint, and A3 is best overall. The A3 versus A2
absolute margin is only `+0.83 pp`, so it is not yet a robust claim about live
residual targets. Offline residualization is worse than absolute conditioning by
`-4.92 pp` under the current implementation.

This table is no longer called official-aligned. Audit found that all five arms
used joint-delta actions, quantile normalization, batch 64, and 20k updates. The
published checkpoint uses absolute actions, mean/std normalization, batch 16,
and 50k updates. The pilot remains a valid internally matched ablation, but its
35.50% A0 is not the final pi0.5 VLA baseline.

The local and published state statistics match to within `6.6e-7`, confirming
that this is the same 6,075,103-frame dataset distribution. In contrast, action
means differ by as much as `1.08` and action standard deviations by `0.51`,
which directly exposes the delta-versus-absolute action mismatch.

The public pi0.5 checkpoint reaches 75.0% in the initial same-bridge 20-episode
diagnostic. Its formal four-seed run is complete at 24/24 task-seed summaries
and 941/1,200 successes: hammer 93.0%, ranking-size 64.0%, ranking-RGB 96.5%,
handover 53.5%, stack-two 93.5%, stack-three 70.0%, and a 78.42% six-task macro.
This shows that the same bridge can reproduce high RoboTwin success and that
the internal A0's 35.50% macro is mainly a training/checkpoint problem rather
than a globally broken evaluator. It also shows that an overall "around 90%"
description depends on task selection: ranking-size, handover, and stack-three
remain genuine weaknesses even for the public checkpoint.

The internal A0 native-protocol diagnostic is complete at 600 episodes: hammer
68%, ranking-size 24%, ranking-RGB 45%, handover 8%, stack-two 35%, and
stack-three 11%, for a 31.83% six-task macro. The same checkpoint's four-seed
same-bridge means are 70.0%, 24.0%, 50.5%, 9.0%, 42.0%, and 17.5%, for a 35.50%
macro. Native is 3.67 percentage points lower overall, but both protocols show
the same task-level failure pattern and remain far below the public checkpoint's
78.42%. The old internal A0's low performance is therefore primarily a
checkpoint/training-recipe problem, not an artifact of the internal bridge.

## A2 causal evaluation incident

The first A2 `current` and `zero` causal jobs failed before producing summaries.
All RoboTwin workers exited with `SIGFPE` on the first So400m forward pass after
the simulator and checkpoint had loaded successfully. The failure came from
running SAPIEN and the PyTorch/So400m encoder in the same simulator process; it
was not evidence against the A2 checkpoint or the intervention.

So400m hint computation now runs in the pi0.5 policy-server process. The zero
control injects a 1152-dimensional float32 zero feature without loading So400m.
A one-episode full-chain smoke, including rendering, online encoding, hint
injection, and policy inference, completed successfully in job
`t-20260801092955-92bwt`. The formal current-hint run restarted as
`t-20260801093423-7d7bx` and has passed the former crash point. No aggregate
causal score is claimed until its 12 summaries are complete.

The first completed current-hint cell is hammer: 74%, 76%, 72%, and 70% over
the four simulator seeds, for a 73.0% mean across 200 episodes. The same A2
checkpoint with its correct predicted milestone scored 82%, 82%, 74%, and 74%
(78.0% mean). Current-state replacement is lower on every seed, with a mean
paired difference of -5.0 percentage points. This is preliminary evidence that
future content contributes beyond supplying an extra visual token. It is not a
closed causal claim until zero and shuffled controls complete; a distribution
shift in the replacement token remains an alternative explanation. The native
evaluator's invalid-scene filtering also produced only 177/200 overlapping
scene IDs between the two conditions. On those matched scenes, 36 succeed only
with the predicted milestone and 24 only with the current feature (two-sided
exact McNemar p=0.155). The direction is consistent but the hammer cell alone
is not statistically conclusive.

The hammer zero-hint control has also completed: 78%, 66%, 60%, and 74%, for
a 69.5% mean. The three conditions are ordered as correct predicted milestone
78.0% > current-state feature 73.0% > zero 69.5%. Correct therefore exceeds
zero by 8.5 percentage points. Across 179 overlapping scene IDs, 38 succeed only
with the predicted milestone and 22 only with zero (two-sided exact McNemar
p=0.0519). This is stronger evidence that the milestone carries useful content,
but remains just above the conventional 0.05 threshold. Current versus zero is
weaker (35 versus 30 discordant matched scenes across 176 overlaps). The final
claim still requires shuffled control and the ranking-RGB and stack-two cells.

The current-state ranking-RGB cell has now completed at 70%, 66%, 64%, and 74%
over the four simulator seeds (68.5% mean, 200 episodes). The matching correct
predicted-milestone run scored 70.5% under the earlier full A2 evaluation. The
0.5-point difference is negligible at this stage, unlike hammer's ordered
correct/current/zero pattern. Zero, shuffled, and matched episode-level tests
must complete before interpreting this task.

The full current-state control is now complete: hammer 73.0%, ranking-RGB
68.5%, and stack-two 71.5%, for a 71.0% macro/micro mean over 600 episodes
(426 successes). Its aggregate report is
`logs/eval_reports/pi05_rt_a2_current_hint_causal.json`. Completion released
four Beijing GPUs only after the platform task exited; the scheduler then
started shuffled-hint job `t-20260801105042-9vmlj` and Beijing returned to
20/20 GPUs.

The zero control is now also complete at 12/12 cells: hammer 69.5%,
ranking-RGB 62.0%, and stack-two 70.0%, for 403/600 successes and a 67.17%
macro/micro mean. The current-state token is therefore `+3.83 pp` above zero
in aggregate, with positive task deltas of `+3.5`, `+6.5`, and `+1.5 pp`.
This consistent ordering supports a contribution from visual content rather
than token presence alone. However, the matching correct-prediction cells from
the completed A2 evaluation are hammer 78.0%, ranking-RGB 65.0%, and stack-two
69.0%, giving a 70.67% three-task macro. That is effectively tied with the
71.0% current-state control (`-0.33 pp` for correct versus current), despite
the task-specific redistribution. The present aggregate evidence therefore
does **not** show that future milestone content is better than a current visual
feature. The shuffled control is still required to determine whether structured
visual content matters beyond a nonzero token.

## Remaining work by priority

### P0: close baseline validity

- The public pi0.5 same-bridge `4 seeds x 6 tasks x 50 episodes` run is complete
  at 78.42% macro.
- Native RoboTwin evaluation of the internal A0 checkpoint is complete at a
  31.83% six-task macro versus 35.50% through the internal bridge.
- Train `pi05_robotwin_a0_public_recipe_bj` with absolute actions, mean/std,
  batch 16, 50k updates, seed 1000, and the published optimizer schedule.
- Evaluate that checkpoint with the same six-task, four-seed bridge. Treat it as
  the primary internal VLA baseline only after it closes the public-checkpoint gap.

### P1: establish WM causality in pi0.5

- Evaluate the same A2/A3 checkpoint with correct, zero, shuffled, and
  other-task milestone inputs. This separates milestone information from extra
  parameters and auxiliary representation shaping.
- Port A2 absolute and A3 onto the corrected A0 recipe only after the corrected
  A0 checkpoint passes the evaluation gate. Do not replicate the legacy 20k
  recipe with more training seeds.
- Add independent training seeds for corrected A0, A2 absolute, and A3. Evaluation seeds
  quantify simulator variance, not training variance.
- Report hierarchical uncertainty with training seed as the upper-level unit.

Prepared in the resource-aware queue:

- A2 and A3 `current`, `zero`, and feature-shuffled causal controls on hammer,
  ranking-RGB, and stack-two (`4 seeds x 50 episodes`).
- Corrected A0 seed-1000 training and its full six-task, four-evaluation-seed
  protocol. Legacy seed-2027 A0/A2/A3 training entries are disabled.
- The Shanghai `robot-task` queue requires at least 12 nominally free GPUs before
  another four-GPU probe because an eight-free-GPU snapshot still failed to
  allocate a contiguous four-GPU job.

### P2: retain mechanism diagnostics

- Finish already-running LaWAM seed replications because their compute is already
  committed and they test stability.
- Do not launch new expensive LaWAM pure/reset-flow baselines while pi0.5 is the
  declared VLA baseline. Their scheduler entries remain disabled.
- Rename every reported LaWAM `no-WM` result to `LaWAM-init / Future-off`.

## Current resource allocation (03:37 UTC)

- Beijing: 20/20 project GPUs: corrected pi0.5 A0 training (4), A2
  shuffled-hint evaluation (4), A3 current-hint evaluation (4), and the
  LaWAM-init/Future-off seed-2028 replication (8). The verified A3 checkpoint
  copy completed at 03:34 UTC; job `t-20260801113452-2dl8p` is running all four
  simulator seeds. The same template is queued for A3 zero and shuffle controls.
- Shanghai robot-task: nominally 8 GPUs free, but both a prior 8-GPU allocation
  and the 4-GPU A2 shuffled-hint allocation `t-20260801102728-46dh7` remained
  queued. A later 2-GPU A3 current-control allocation
  `t-20260801105741-czthv` also remained queued and was reclaimed after 180
  seconds. The reported eight GPUs therefore cannot satisfy even the available
  2-GPU preset; the dispatch threshold remains 12 free GPUs.
- Shanghai East: 8/8 occupied by another job.
- gf1: 8/8 used by an in-flight residual seed replication.
- Local: 2/2 used by native internal-A0 ranking-size and stack-three evaluation.

The scheduler monitors all resources and will not submit the disabled LaWAM
strict-baseline tasks. New capacity should first be assigned to pi0.5 causal
interventions or training-seed replication.
