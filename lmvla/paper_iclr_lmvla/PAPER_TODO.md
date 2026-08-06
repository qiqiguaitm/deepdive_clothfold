# pi0.5-Preserving Predictive and Outcome-Calibrated Control TODO

Updated: 2026-08-06 10:10 UTC

This file contains only unfinished training/evaluation evidence and current
gates. Completed P0/P1/R0/R1/R2/R3/R4 evidence, completed P2 training, and
superseded execution history are preserved in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`. Canonical JSON artifacts take
precedence over status prose.

## Evidence boundary at this cutoff

- P1 remains a positive seed-1000 predictive-adapter screen: 82.42% versus
  69.00% for matched current-source A0. Its +1.25-point margin over shuffled
  action conditioning has Holm-adjusted `p=1.0`; P2 is required for both a
  replicated utility claim and a stronger content-specific interpretation.
- R1 remains a rejected recurrence-aligned auxiliary extension. Its 62.92%
  result does not negate the parent P1 adapter.
- R4 is complete and rejected at its three-seed replication gate. Terminal-
  outcome weighting averages 74.94%, versus 72.14% ordinary and 71.14%
  outcome-free CRAVE. Mean effects are +2.81 points (95% CI
  `[-0.94,+6.58]`) and +3.81 points (`[-0.25,+7.97]`); both utility checks
  fail while the five-point task-safety guards pass.
- P2 final training and all six R4 replication training arms are complete and
  audited. Checkpoint completion and training telemetry are not control
  evidence.

## P2: predictive-adapter replication

Both seed-1001 and seed-1002 final step-49,999 checkpoints passed independent
v2 identity and integrity audits. The first East evaluations failed before any
valid episode because they inherited an sm80 Curobo build on H20; those
zero-summary attempts are archived and excluded. The frozen sm90 ABI smoke
passed, and fresh four-H20 jobs `t-20260806155107-lf65m` and
`t-20260806155110-9sb67` are running. The latest audited per-cell cutoff at
09:30 UTC was 12/24 and 13/24, respectively, with zero failed cells; the 09:48
  scheduler state still reports both evaluations running. At 10:10 UTC both
  have reached `20/24` cells with zero failed cells.

- [ ] Complete both final-checkpoint evaluations on all 24 frozen
  task-by-evaluation-seed cells and 1,200 episodes per training seed.
- [ ] Materialize and audit both reports, including scene and episode
  identities, checkpoint hashes, source amendment, and zero invalid cells.
- [ ] Run the preregistered hierarchical paired analysis over training seed,
  task, evaluation seed, and paired episode key. Report both seed effects and
  all six task effects; a replicated utility claim requires the 95% interval
  for predictive adapter minus matched A0 to exclude zero.
- [ ] Only if the P2 utility gate passes, finalize matched parameter, FLOP,
  peak-memory, direct-latency, WebSocket-latency, and throughput comparisons.

## Manuscript and figure gates

- [ ] Replace the P2 placeholder only after both complete reports and the
  hierarchical gate exist.
- [ ] Any new claim-bearing figure must use a pure-white canvas; 6--8 pt
  sans-serif text at final size; strokes of at least 0.5 pt; sentence-case
  labels; a non-colour cue for every series; and a caption defining conditions,
  episode counts, training seeds, and interval type.

## Current scheduler gates

Canonical snapshot at 10:10 UTC: 497 tasks total, 370 completed, 120 disabled,
5 pending, and 2 running. East is occupied by the two P2 evaluations at 20/24
cells each. All R4 parent, materialization, and gate nodes are complete; North
has no managed active or queued GPU. The remaining pending nodes are downstream
P2 audit/gate/conditional-efficiency work plus the exhausted local attach
helper. gf1 remains permanently retired and robot-task submissions remain
disabled.

The snapshot inventory reports 370 completed and 120 disabled; direct task-
object aggregation reports 383 completed and 107 disabled. The 13-task
difference is understood: one retired local-assist node and 12 superseded L2
control/attach nodes retain historical completion state but are disabled by
current queue policy. Running, pending, and total counts agree.

## Stop rules

- P2 failure retains the predictive adapter as bounded seed-1000 evidence and
  leaves the replicated method claim rejected.
- R4 replication failure retains the accepted seed-1000 screen but rejects a
  replicated outcome-calibrated utility claim.
- AHEAD-style interception and world-critic RL require new benchmarks or
  action-diverse reward data and remain outside this graph.
- LeWM/DINO replacement, from-PaliGemma initialization, A2/A3 variants, R1
  replication, and closed MT3--MT6 tasks must not be scheduled from this TODO.
