# Experiment handoff (2026-07-30 05:55 UTC)

## Current objective

Make RoboTwin the primary evidence for LMWAM by closing three gaps:

1. exact balanced six-task data and a full method matrix;
2. independent training seeds and causal milestone interventions;
3. cross-stack replication with official-aligned pi05 A0/A1/A2/A3.

LIBERO remains a diagnostic benchmark rather than the main evidence.

## Established results

- RoboTwin balanced-gate dual2q evaluation: macro success `88.42%`.
- The earlier pi05 A0 replan probe is invalid as a policy baseline: macro
  success `6.67%`, with nearly all success from `beat_block_hammer`.
- LIBERO residual no-temporal-schedule replications:
  `95.85%` for `n2_seed2` and `94.73%` for `resid_noTsS4`.
- One-H20 all6 throughput:
  batch16/accum2 `2.279 s/update`; batch32/accum1 `2.142 s/update`
  (`6.0%` faster at equal global batch).
- Fused AdamW task `t-20260730133037-4cg54` completed at
  `2.140 s/update`; this is not a stable gain over non-fused and remains off.

## Running production tasks

Status at 2026-07-30 05:55 UTC:

| Arm | Task ID | Progress | Throughput / ETA |
|---|---|---:|---|
| pi05 A0 official | `t-20260730092737-597xp` | 16.0k/20k | 2.6 s/step, ~2.9 h |
| pi05 A1 absolute prefix | `t-20260730083819-x7l5s` | 13.9k/20k | 2.6 s/step, ~4.4 h |
| pi05 A3 live residual | `t-20260730084422-4lb46` | 12.8k/20k | 2.6 s/step, ~5.3 h |
| all6 no-WM seed2026 | `t-20260729213821-2wb77` | 9.4k/20k | ~2.0 s/step |
| all6 local seed2026 | `t-20260729213825-8fmp2` | 4.0k/20k | ~2.4 s/step |

The remaining Beijing all6 arms are queued: absolute, residual, isolation,
local seed2027 and combo seed2027. Shanghai combo seed2026 is also queued.

## A2 preparation

`t-20260730073515-6wktk` completed all 27.5k episode shards and wrote the
6,075,103-frame absolute `hint.npz` (about 12 GB). It then failed in residual
conversion because the first episode used an unpooled local grid before reading
the pooled cache. The assignment is fixed and covered by a synthetic
first/subsequent-frame regression test. Recovery task
`t-20260730151342-6tvjv` reuses the completed absolute artifact and only builds
the residual file.

The monitor automatically submits:

- A2 absolute: `pi05_robotwin_a2_prefix_official_bj.yaml`;
- A2 residual: `pi05_robotwin_a2_residual_prefix_official_bj.yaml`.
- four-seed A2/A2-residual RoboTwin evaluations after each training task
  completes.

Both use official-aligned 20k-step/global-batch-64 training and persistent
HF/JAX caches.

A2 absolute training was automatically submitted as
`t-20260730151045-4znrf` and is running. A2 residual training remains gated on
the recovery output.

The online RoboTwin hint path now supports So400m and residual hints. Initial
validation exposed H20 bf16 batch-kernel drift: batch1/2 had cached-grid cosine
`0.98799/0.98487`, while batch4 and above reproduced the cache exactly. The
online encoder now pads small batches to four and trims the output. Final task
`t-20260730141040-v7m8p` passed with grid MAE `0`, cosine `0.99999988`, and
absolute/residual identity error `0`.

## Pending framework probes

| Probe | Task ID | State |
|---|---|---|
| all6 8-H20 batch32/accum1 | `t-20260730131155-62dqf` | Beijing queue |
| A3 sparse-diagnostics control | `t-20260730131618-cgntr` | Shanghai queue |
| A3 fused multi-view encoding | `t-20260730131623-gwv4n` | Shanghai queue |

Do not change the all6 production batch default until the 8-H20 probe confirms
the one-GPU result. The fused-vision implementation has already passed a real
SigLIP equivalence check: forward max difference `0`, parameter-gradient max
difference `4.47e-8`.

## Framework changes completed

- StarVLA full Accelerate save/resume: model, optimizer, scheduler,
  dataloader, RNG and optimizer step; legacy inference checkpoint retained.
- StarVLA sparse CUDA-event timing, sparse metric scalarization,
  `zero_grad(set_to_none=True)`, and corrected processor kwargs.
- OpenPI gradient/parameter norms only at log boundaries.
- OpenPI opt-in multi-view/A3-target vision batch fusion.
- Persistent Hugging Face and JAX compilation caches in A0/A1/A2/A3 launchers.

Validation completed:

- StarVLA CPU checkpoint save/mutate/load round trip;
- OpenPI pi0 tests: 4 passed;
- fused-vision forward and gradient equivalence;
- local and Beijing pycompile plus `git diff --check`.

## Automation

The `volc-auto-progress` tmux session runs
`train_scripts/kai/volc/auto_progress_monitor.py` every 180 seconds.
State is stored in `logs/volc_auto_progress_state.json`.

It currently:

- records throughput probe summaries;
- submits A2/A2-residual when their hint files appear;
- submits A0/A1/A2/A2-residual evaluation after training completes;
- tracks the all6 matrix and existing RoboTwin/LIBERO summaries.

Next experimental priority after checkpoints appear:

1. A0/A1/A2/A3 official RoboTwin evaluation;
2. all6 no-WM/local/absolute/residual/isolation/combo comparison;
3. correct/zero/shuffled/other-task milestone interventions;
4. second training seed and hierarchical bootstrap;
5. expand from six tasks to a preregistered 10-12 task panel.

A3 is trained on Shanghai vePFS while the validated RoboTwin simulator is
mounted in Beijing. Its evaluation still requires a verified resumable
checkpoint transfer or a validated Shanghai RoboTwin runtime; do not treat it
as automatically closed.
