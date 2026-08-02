# RoboTwin training throughput optimization (2026-07-30)

## Controlled probes

All A3 probes used the official-aligned global batch size 64 and identical
model/data settings. Production training steps were not reduced.

| Configuration | Stable throughput | Decision |
|---|---:|---|
| `num_workers=2`, `fsdp_devices=1` | 7.99 s/step | reject |
| `num_workers=8`, `fsdp_devices=1` | 2.65 s/step | production default |
| `num_workers=12`, `fsdp_devices=1` | 2.65 s/step | no gain over 8 |
| `num_workers=16`, `fsdp_devices=1` | stalled after loader creation | reject |
| `num_workers=8`, `fsdp_devices=4` | 2.50 s/step | only 4-6% faster; keep official mesh |

The workers=8 setting is about 3.8x faster than the previous production
throughput of 9.8-10.4 s/step. FSDP4 changes the official `fsdp_devices=1`
configuration and adds checkpoint resharding risk for a small incremental
gain, so it was not adopted.

The all6 no-cache and 308 GB hot-cache probes both measured about
2.31-2.32 s/optimizer-step with 0-1 ms data time. Keep
`enable_video_frame_cache=false`; the cache does not improve throughput.

## Framework audit and implementation

The July 30 framework audit found that data loading is no longer the steady
state bottleneck. The following semantics-preserving changes are implemented:

- StarVLA training metrics remain as detached GPU tensors and are reduced and
  converted to Python scalars only at `logging_frequency`; the previous loop
  called `.item()` for every component on every micro-step.
- Progress timing uses sparse CUDA events every 20 optimizer steps. Plain
  `perf_counter` is not a valid GPU timer after removing the per-step
  synchronization.
- `optimizer.zero_grad(set_to_none=True)` is enabled. Fused AdamW is available
  as an opt-in probe flag and remains disabled by default.
- Transformers processor arguments now use `processor_kwargs`, removing the
  warning emitted once per rank and batch.
- StarVLA checkpoints now save model, optimizer, scheduler, dataloader and RNG
  state through Accelerate, plus the completed optimizer step. The legacy
  `steps_N_pytorch_model.pt` inference path is retained as a symlink to the
  saved model state. A CPU save/mutate/load round-trip restored model
  parameters, optimizer LR and step exactly.
- OpenPI computes full-model `grad_norm` and `param_norm` only at log
  boundaries. Windowed losses retain their previous averaging semantics.
- The A0/A1/A2/A2-residual/A3 Beijing launchers now put Hugging Face datasets
  and the JAX compilation cache on persistent vePFS paths, so retries and
  resumes reuse prepared artifacts.
- An opt-in OpenPI vision-batch fusion path encodes the three camera views and
  optional A3 target frame in one encoder call when shapes match. A reduced
  SigLIP test measured zero forward difference and maximum parameter-gradient
  difference `4.47e-8`. It remains disabled pending a paired H20 probe.
- RoboTwin online So400m evaluation pads encoder batches smaller than four.
  H20 bf16 batch1/2 otherwise selected a numerically different kernel
  (cached-grid cosine `0.98799/0.98487`); batch4 reproduced the offline cache
  exactly. Final validation `t-20260730141040-v7m8p` measured MAE `0`, cosine
  `0.99999988`, and zero residual-identity error.

### Pending controlled probes

| Task | ID | Queue | Purpose |
|---|---|---|---|
| all6 batch32/accum1 | `t-20260730131155-62dqf` | Beijing, queued | Preserve global batch 256 while replacing two micro-steps with one |
| A3 sparse diagnostics control | `t-20260730131618-cgntr` | Shanghai, queued | New logging path with per-view vision encoding |
| A3 fused vision | `t-20260730131623-gwv4n` | Shanghai, queued | Paired treatment for camera/target batch fusion |

The one-GPU paired all6 checks completed successfully:

| Configuration | Task | Stable CUDA time | Result |
|---|---|---:|---|
| batch16, accumulation2 | `t-20260730132135-fnpps` | 2.279 s/update | control |
| batch32, accumulation1 | `t-20260730131747-vtfvh` | 2.142 s/update | no OOM; 6.0% faster |
| batch32, accumulation1, fused AdamW | `t-20260730133037-4cg54` | 2.140 s/update | no stable gain; keep disabled |

Both configurations used global batch 32 and identical optimizer-step counts.
The result isolates micro-batch fusion from the logging changes. Adopt it for
the 8-GPU jobs only if the queued global-batch-256 probe confirms the gain
under DDP; the existing production defaults remain batch16/accumulation2.
The fused-AdamW probe matched the non-fused first timing window
(`2.140` versus `2.141` s/update); its faster final window was a single short
sample. This is insufficient evidence to change the optimizer default.

The first Shanghai 8-GPU all6 probe, `t-20260730130941-nb4w8`, was stopped
without running because the queue reported insufficient whole-node quota. It
was resubmitted to Beijing as listed above. None of these pending probes is
used as evidence of a speedup yet.

## Production resumes

- A1: `t-20260730083819-x7l5s`, resumed from complete step 7000 with
  `num_workers=8`; verified at 2.6 s/step.
- A3: `t-20260730084422-4lb46`, resumed from complete step 6000 with
  `num_workers=8`; verified at 2.6 s/step.
- A0: `t-20260730092737-597xp`, resumed from the complete step-10000
  checkpoint with `num_workers=8`; verified at 2.6 s/step. Switching at
  10.9k instead of waiting for step 15000 is expected to save about
  17-18 hours despite the roughly 900-step rollback.
- Shanghai all6 combo: `t-20260730084503-gv5st`, non-preemptible and queued
  behind A3.

The monitor state now tracks the resumed task IDs, and the
`volc-auto-progress` tmux monitor has been restarted.

## Additional status

- A2 So400m export `t-20260730073515-6wktk` completed all 27.5k shards and
  the 6,075,103-frame absolute hint, then exposed a first-episode pooling bug
  in residual conversion. The fix passed a synthetic regression test;
  residual-only recovery is `t-20260730151342-6tvjv`. A2 absolute training
  `t-20260730151045-4znrf` was submitted automatically.
- At 2026-07-30 05:32 UTC, A0 was at 15.5k/20k (about 3.2 h remaining), A1
  at 13.5k/20k (about 4.7 h remaining), and A3 at 12.3k/20k (about 5.6 h
  remaining), all near 2.6 s/step.
- Beijing all6 `nowm` seed 2026 was at 8.8k/20k and `local` seed 2026 at
  3.4k/20k. The remaining Beijing all6 matrix arms are queued.
- The incomplete cross-region A3 checkpoint copy is quarantined as
  `6000.incomplete_cross_region_20260730T0045Z` and cannot be mistaken for a
  resumable checkpoint.
