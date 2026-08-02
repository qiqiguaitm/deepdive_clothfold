# pi0.5 RoboTwin simulator-slot speed probe

## Setup

- Checkpoint: official-aligned A2 prefix, server-side So400m hint.
- Task: `beat_block_hammer`, `demo_clean`.
- Four eval seeds, 10 valid episodes per seed.
- One policy server/GPU; compare `ROBOTWIN_NUM_SLOTS=1` and `2`.

## Timing

| Slots | Per-seed elapsed seconds | Mean | Relative throughput |
|---|---|---:|---:|
| 1 | 180.6, 155.0, 183.7, 176.5 | 174.0 | 1.00x |
| 2 | 139.0, 115.0, 133.6, 136.2 | 131.0 | 1.33x |

Two simulator slots reduce mean elapsed time by 24.7%, substantially less than an ideal 2x speedup.

## Protocol warning

This is not a drop-in replacement for the main evaluation protocol. The two runs shared only 35 of their 40 accepted scene seeds because invalid-scene replacement proceeded differently. Among those 35 common seeds, 13 success labels changed. Aggregate success was 80% for slots=1 and 70% for slots=2, but the probe is too small and the trajectories are not paired consistently enough to interpret that difference as a policy effect.

Use slots=2 only for non-decisive smoke tests or throughput diagnostics. Keep `ROBOTWIN_NUM_SLOTS=1` for paper-facing evaluations and causal interventions.
