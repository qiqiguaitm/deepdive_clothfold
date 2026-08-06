# pi0.5-Preserving Predictive and Outcome-Calibrated Control TODO

Updated: 2026-08-06 01:17 UTC

This file contains only unfinished training/evaluation evidence and current
gates. Completed P0/P1/R0/R1/R2/R3 evidence, completed R4 prerequisites, and
superseded execution history are preserved in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`. Canonical JSON artifacts take
precedence over status prose.

## Evidence boundary at this cutoff

- P1 is complete and accepted at training seed 1000. The predictive adapter
  reaches 82.42% versus 69.00% for its matched current-source A0, with no
  task-level regression. This is direct single-seed closed-loop evidence.
- P1 does not yet establish a replicated method effect. Its margin over
  action-shuffled conditioning is only +1.25 percentage points
  (Holm-adjusted `p=1.0`), so a strong content-specific causal claim is also
  premature.
- R1 is complete and rejected. The predictive-plus-CRAVE extension reaches
  62.92%, below A0 by 6.08 points and CRAVE-only by 5.75 points, with four
  task regressions larger than five points. This rejects the recurrence-aligned
  extension, not the parent predictive adapter.
- R4 data, runtime, smoke, three step-5,000 training checkpoints, and checkpoint
  integrity are complete. They authorize the fixed-checkpoint evaluation but
  are not control evidence.

## P2: predictive-adapter replication

At the 01:17 UTC scheduler cutoff, both frozen seed replications are healthy on
Robot-East-H20. Seed 1001 is near step 23,100 and seed 1002 near step 23,000 of
50,000; both have complete committed step-20,000 Orbax checkpoints. Training
loss and intermediate checkpoints are health telemetry only.

- [ ] Finish seed-1001 and seed-1002 training at the frozen final step 49,999
  and pass the source, dataset, normalization, parameter-tree, optimizer-state,
  and checkpoint audits.
- [ ] Evaluate both final candidates on all 24 frozen task-by-evaluation-seed
  cells (1,200 episodes per training seed). Do not evaluate an intermediate
  checkpoint.
- [ ] Run the preregistered hierarchical paired analysis over training seeds,
  tasks, and episode keys. A replicated utility claim requires the 95% interval
  for candidate minus matched A0 to exclude zero. Report every training-seed
  effect and all six task effects.
- [ ] Only if the P2 utility gate passes, finalize matched parameters, FLOPs,
  peak training memory, direct and WebSocket latency, and throughput against
  A0 on the frozen hardware protocol.

## R4: outcome-calibrated fixed-checkpoint screen

At the 01:17 UTC scheduler cutoff, ordinary fine-tuning has completed 22/24
cells with no failed cell. The first ABI-repaired North attempts for
outcome-free CRAVE and terminal outcome failed at their first compiled policy
query because the transferred Triton `ptxas` bytes had lost executable mode;
all eight failures occurred before a valid episode. A scheduler-reload race
briefly started two more jobs under the superseded readiness marker; both were
stopped during initialization. These zero-episode attempts are excluded. A
second immutable repair verifies the SHA-256 of `cuobjdump`, `nvdisasm`,
`ptxas`, and `ptxas-blackwell`, restores mode `0755`, and executes a version
preflight for each binary. Fresh recommendation audits then selected North for
CRAVE job `t-20260806085323-85xzv` and terminal-outcome job
`t-20260806085327-wdxr6`, four H20 GPUs each. The cold H20 `max-autotune`
compile completed without another runtime error. CRAVE has completed three of
four first-task cells and terminal outcome all four, each containing 50 valid
episodes; the remaining CRAVE seed is active and every seed has an empty failed
set. These seven completed cells prove the repaired execution path, but partial
panels must not be summarized as a method result.

- [ ] Complete all 24 cells and 1,200 episodes for ordinary, outcome-free
  CRAVE-weighted, and terminal-outcome-weighted seed-1000 checkpoints.
- [ ] Materialize both North reports locally, verify the frozen scene manifest,
  episode identities, checkpoint hashes, action bridge, and 24-cell/1,200-
  episode counts, then run the preregistered three-arm gate.
- [ ] Report every task. Do not promote a macro improvement if any task violates
  the five-point regression guard.
- [ ] Replicate R4 only if the seed-1000 fixed-checkpoint gate passes. Otherwise
  close the branch without launching new training seeds.
- [ ] Do not make a Q-value, advantage, world-critic, reward, or
  model-predictive-control claim. The terminal-outcome and CRAVE sidecars are
  sample-weighting signals over expert demonstrations, not action-diverse
  consequence estimates.

## Manuscript and figure gates

- [ ] Replace the P2 manuscript placeholder only after both final evaluations
  and the hierarchical gate are complete.
- [ ] Add an R4 claim-bearing table or figure only after all three 24-cell
  reports pass audit. Until then, retain a placeholder and no partial macro.
- [ ] Any new claim-bearing figure must use a pure-white canvas; 6--8 pt
  sans-serif text at final size; strokes of at least 0.5 pt; sentence-case
  labels; a non-colour cue for every series; and a caption defining all
  conditions, episode counts, training seeds, and interval type.

## Current scheduler gates

Canonical snapshot at 01:17 UTC: 467 tasks total, 333 completed, 120 disabled,
9 pending, and 5 running. All eight East H20 GPUs, eight North H20 GPUs, and
both local A100 GPUs are occupied by the five active P2/R4 tasks. Seven tasks
wait only for final checkpoints, reports, or an accepted gate; none is ready
but resource-starved. gf1 remains permanently retired and robot-task new
submissions remain disabled.

The snapshot queue inventory reports 333 completed and 120 disabled, whereas a
direct aggregation of historical state objects reports 346 completed and 107
disabled. The 13-task difference is audited: one retired local-assist task and
12 superseded L2 control/attach tasks retain historical `completed` state but
are administratively disabled in the current queue policy. The snapshot uses
the current effective status; the state file preserves execution history.
Running, pending, and total counts agree, and this dual accounting does not
change the five active task identities or any scientific result.

## Stop rules

- P2 failure retains the current negative-integration paper and archives the
  predictive-adapter screen as bounded single-seed evidence.
- R4 failure closes outcome-calibrated replication. CRAVE progress and terminal
  outcomes must not be relabelled as reward, action advantage, or control value.
- AHEAD-style interception and world-critic RL require new benchmarks or
  action-diverse reward data and remain outside this graph.
- LeWM/DINO visual replacement, from-PaliGemma initialization, A2/A3 variants,
  R1 replication, and closed MT3--MT6 tasks must not be scheduled from this
  TODO.
