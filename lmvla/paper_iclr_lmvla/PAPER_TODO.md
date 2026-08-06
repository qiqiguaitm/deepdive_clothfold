# pi0.5-Preserving Predictive and Outcome-Calibrated Control TODO

Updated: 2026-08-06 05:36 UTC

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
- R4 seed-1000 fixed-checkpoint evidence is complete and accepted by its
  preregistered screen: terminal outcome reaches 77.58%, versus 74.25% for
  ordinary sampling and 71.08% for outcome-free CRAVE. This authorizes the
  frozen two-seed replication; it is not yet a replicated method claim.

## P2: predictive-adapter replication

At the 05:36 UTC scheduler cutoff, both frozen seed replications are healthy on
Robot-East-H20. Seeds 1001 and 1002 have passed approximately step 43,400 and
43,500 of 50,000; both have complete committed step-40,000 Orbax checkpoints.
Training loss and intermediate checkpoints are health telemetry only.

The final-checkpoint gate is now explicit rather than inferred from
`params/_METADATA`. Integrity amendment
`pi05_predictive_adapter_p2_integrity_amendment_v1` pins the clarified
scene-seed verifier and evaluation launcher, records exact training metadata,
target-pair, frame-cache index, and normalization identities, and requires an
atomic step-49,999 checkpoint with a parameter tree matching the step-25,000
reference and complete first- and second-moment optimizer state. The scheduler
contains one zero-GPU audit node per replication seed. Each formal evaluation
requires that seed's independent audit marker and reruns the audit at launch.
The frozen postprocessing overlay also restores the exact preregistered source
versions while allowing canonical reports to be written outside that immutable
tree. Its hashes, platform launchers, and protocol verification are pinned in
the same amendment. The amendment changes no training or evaluation condition.

- [ ] Finish seed-1001 and seed-1002 training at the frozen final step 49,999
  and pass the source, dataset, normalization, parameter-tree, optimizer-state,
  and checkpoint audits.
- [ ] Evaluate both final candidates on all 24 frozen task-by-evaluation-seed
  cells (1,200 episodes per training seed). Do not evaluate an intermediate
  checkpoint.
- [ ] Run the preregistered hierarchical paired analysis over training seeds
  and paired episode keys within each of the six fixed tasks, with task effects
  averaged equally. A replicated utility claim requires the 95% interval for
  candidate minus matched A0 to exclude zero. Report every training-seed effect
  and all six task effects.
- [ ] Only if the P2 utility gate passes, finalize matched parameters, FLOPs,
  peak training memory, direct and WebSocket latency, and throughput against
  A0 on the frozen hardware protocol.

## R4: outcome-calibrated fixed-checkpoint screen

All three seed-1000 arms have completed and passed the frozen-manifest audit:
24/24 cells, 1,200 episodes, six tasks, and no invalid cell per arm. Canonical
reports are `lmvla/lmwm/docs/pi05_r4_ordinary_seed1000.json`,
`lmvla/lmwm/docs/pi05_r4_outcome_free_crave_seed1000.json`, and
`lmvla/lmwm/docs/pi05_r4_terminal_outcome_seed1000.json`. Two-slot execution
exposed a verifier defect: asynchronous completion permuted otherwise identical
unique scene-seed sets. The clarified verifier still rejects wrong, missing,
extra, or duplicate seeds but treats completion order as non-identifying. The
amendment and regression tests passed; no rollout was changed or rerun.

The first ABI-repaired North attempts failed before a valid episode because the
transferred Triton `ptxas` bytes had lost executable mode. The immutable repair
verified tool hashes, restored mode `0755`, and passed binary preflights. Fresh
recommendation audits selected North for CRAVE job
`t-20260806085323-85xzv` and terminal-outcome job
`t-20260806085327-wdxr6`, four H20 GPUs each. Both wrote all 24 cells. A
platform teardown race omitted their final markers and reported terminal as
failed after its last summary was durable. The frozen verifier and summarizer
were rerun in place, both trees passed, and verified file-by-file reverse sync
materialized the reports. A scheduler fallback that began a duplicate local
evaluation was stopped before producing a summary.

The seed-1000 gate accepted terminal outcome: 77.58% versus 74.25% ordinary
and 71.08% outcome-free CRAVE, for macro deltas of +3.33 and +6.50 percentage
points. No task regressed by more than five points against either control. The
descriptive paired hierarchical intervals are [-2.25, +8.92] points against
ordinary and [+0.33, +12.92] against CRAVE; the first crosses zero, so this is
a replication trigger rather than a final significance claim. Complete
task-level evidence is frozen in `RESULTS_pi05_r4_seed1000_complete.{json,md}`.

The conditional replication branch was preregistered before either pending
seed-1000 panel completed. Protocol `pi05_r4_three_seed_replication_v1`
freezes seeds 1001/1002, the same three arms and 5,000-step recipe, the same
24-cell/1,200-episode evaluation per arm and seed, and a paired hierarchical
gate over training seed, task, evaluation seed, and scene. Acceptance requires
the 95% interval lower bound to exceed zero against both controls and forbids
any seed/task regression larger than five points. The scheduler contains six
training, six evaluation, and one final gate node. The accepted marker now
makes training eligible subject to a fresh resource recommendation and a
verified environment; evaluations and the final gate remain dependency-blocked.

- [x] Complete all 24 cells and 1,200 episodes for ordinary, outcome-free
  CRAVE-weighted, and terminal-outcome-weighted seed-1000 checkpoints.
- [x] Materialize both North reports locally, verify the frozen scene manifest,
  episode identities, checkpoint hashes, action bridge, and 24-cell/1,200-
  episode counts, then run the preregistered three-arm gate.
- [x] Report every task and apply the five-point regression guard.
- [ ] Complete the authorized seed-1001/1002 three-arm replication and its
  frozen hierarchical gate. Do not promote the accepted seed-1000 screen to a
  replicated claim.
- [ ] Do not make a Q-value, advantage, world-critic, reward, or
  model-predictive-control claim. The terminal-outcome and CRAVE sidecars are
  sample-weighting signals over expert demonstrations, not action-diverse
  consequence estimates.

## Manuscript and figure gates

- [ ] Replace the P2 manuscript placeholder only after both final evaluations
  and the hierarchical gate are complete.
- [ ] Add the audited R4 seed-1000 table with an explicit screen-only label;
  replace it with a claim-bearing replicated table only after seeds 1001/1002
  and the final hierarchical gate complete.
- [ ] Any new claim-bearing figure must use a pure-white canvas; 6--8 pt
  sans-serif text at final size; strokes of at least 0.5 pt; sentence-case
  labels; a non-colour cue for every series; and a caption defining all
  conditions, episode counts, training seeds, and interval type.

## Current scheduler gates

Canonical snapshot at 05:36 UTC: the two P2 training tasks occupy all eight
East H20 GPUs and have passed approximately step 43,400/49,999 and
43,500/49,999 with stable finite losses. Their exact final-checkpoint audits
remain dependency-blocked until step 49,999 exists.

R4 replication training has moved to a separately frozen North operational
amendment without changing the scientific protocol. The staged runtime passed
the 17-file source audit, exact four-rank/effective-batch-16 config validation,
and binding check. The East and North public-model directories matched over all
29 files (9,354,105,778 bytes; aggregate manifest SHA-256
`1ed525630ca0b88ac8bad1f7e57732153b1639cfee724ca7096c38dd67947770`). A
single-task container smoke then passed step 100 before concurrency expanded.
Fresh recommendation audits selected Robot-North-H20 for all six jobs:
ordinary seed 1001 `t-20260806131842-nbqkd`, ordinary seed 1002
`t-20260806133116-vb7lj`, outcome-free CRAVE seeds 1001/1002
`t-20260806133121-7qxml` and `t-20260806133125-t49f6`, and terminal-outcome
seeds 1001/1002 `t-20260806133130-zcvnz` and
`t-20260806133135-qqwz7`. All six are running and occupy 24/25 North H20 GPUs;
the first smoke arm has passed step 500 while the five later jobs complete
their first compilation. Evaluations and the final gate remain correctly
dependency-blocked. gf1 remains permanently retired and robot-task new
submissions remain disabled.

The snapshot queue inventory reports 335 completed and 120 disabled, whereas a
direct aggregation of historical state objects reports 348 completed and 107
disabled. The 13-task difference is audited: one retired local-assist task and
12 superseded L2 control/attach tasks retain historical `completed` state but
are administratively disabled in the current queue policy. The snapshot uses
the current effective status; the state file preserves execution history.
Running, pending, and total counts agree, and this dual accounting does not
change the four running task identities or any scientific result.

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
