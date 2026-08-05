# pi0.5-Preserving Predictive and Recurrence-Aligned Control TODO

Updated: 2026-08-05 08:20 UTC

This file contains only unfinished training/evaluation work and current gates.
Completed or superseded evidence is preserved in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`; canonical JSON artifacts take
precedence over status prose. Completed P0, R0, R2, and R3 evidence is archived;
none of those results establishes control utility.

## Current execution and gates

- P1 current-source A0 and candidate seed 1000 retain matched, source-frozen
  10,000-step recovery checkpoints. The operator permanently shut down and
  retired gf1 on 2026-08-04; its last heartbeats near steps 13,500 and 12,400
  are historical telemetry, not recoverable running work. At 14:28 UTC the
  scheduler terminalized both original attempts and returned P1 to pending.
  gf1 now has zero schedulable capacity: runtime queue construction removes
  every gf1 candidate, orphaned gf1-only nodes are disabled, SSH probing is
  suppressed, and the launcher has a hard rejection backstop. Historical gf1
  provenance remains unchanged. All future P1 training and evaluation must use
  Robot-East-H20 or the audited Robot-North-H20 recovery path. Both formal P1
  launchers detect complete numeric checkpoints and pass `--resume`, so a
  resubmitted attempt restores optimizer state instead of overwriting the
  10,000-step evidence. A checksum-frozen North stage copied
  approximately 187.4 GB of base/P0/P1 state, frame cache, pairs, dataset,
  tokenizer cache, normalization assets, and source files under
  `.staging/pi05_p1_failover_20260804T1034Z`. The original sync stopped after
  12.44 GB because loop-internal SSH consumed the artifact manifest from stdin;
  the corrected retryable transfer subsequently completed all bytes. A
  concurrent change to `kai0/src/openpi/training/config.py` at 14:16 UTC made
  the current source tree diverge from the 10:34 inventory. The exact 75-file
  frozen `openpi` tree was recovered from the isolated R1 overlay; its file
  count, byte count, and inventory hash exactly match the unchanged P1
  manifest. After staging that source without overwriting the shared checkout,
  the North artifact audit, source-freeze check, both CPU dry runs, and both
  optimizer-resume checks pass. The operator-confirmed gf1 retirement is bound
  to the original PIDs and last-known steps in an authorization whose local and
  North audits both report `launch_authorized=true`. The first 8-GPU paired
  resume (`t-20260805090937-zgccb`) failed before the launcher and exposed a
  scheduler defect: a task without an output specification treated any
  platform terminal state as completion. The parent now requires both remote
  step-49999 metadata files, so a failed or stopped platform job cannot unlock
  materialization or evaluation. Two subsequent infrastructure probes exposed
  and closed container-runtime mismatches: the shared editable install
  shadowed frozen `openpi`, and the staged `videos` symlink still targeted the
  East filesystem. The runtime amendment pins a stage-local `jq`,
  `PYTHONPATH`, loader identity, the North video target, matching
  episodes/tasks/info hashes, 82,500 visible videos, and a decoded 139-frame
  LeRobot episode-0 sample. North materializers are now blocked until their
  parent is actually complete. After a fresh recommendation audit, the repaired
  paired resume (`t-20260805110010-5mw68`) started immediately on
  Robot-North-H20 at 03:00 UTC. Both arms passed source verification, loader
  construction, and the 27,500-file data index, then restored their complete
  step-10000 Orbax train states in 24.18 and 23.35 seconds. At step 10100 both
  run near 1.3 iterations/s on disjoint four-GPU sets. At 05:50 UTC A0 reached
  step 22,800 and the candidate reached step 22,900, with approximately
  5.7 hours remaining. At 06:25 UTC the same repaired task remained healthy:
  A0 reached step 25,500 and the candidate reached step 25,700. A0 reports loss 0.0061--0.0066; the candidate reports
  main loss 0.0080--0.0092 and nonzero predictive loss 0.0032--0.0033, with no
  NaN, NCCL, or data error. The recovery
  atomically exposes final checkpoints only after complete 49999 metadata and
  train state arrive. A separate isolated P1 evaluation overlay and runtime
  preflight are hash-pinned for the five seed-1000 gate conditions, but
  closed-loop evaluation remains blocked on both final checkpoints.
- Corrected R1 CRAVE-only (`t-20260804154532-xrlk4`) and combined
  (`t-20260804154529-k9ss2`) training completed on Robot-East-H20. Both final
  step-49999 checkpoints contain root checkpoint metadata, parameter metadata,
  train state, and normalization assets. Nonzero recurrence losses established
  objective delivery during training, but closed-loop utility is still unknown.
  Four attempted seed-1000 evaluations (combined, shuffled, zero-route, and
  CRAVE-only) failed immediately and produced no evidence because the frozen R1
  verifier detected source drift before simulator startup. No artifact or loss
  from the superseded sparse-label launches may enter analysis. Exact frozen R1
  sources were recovered into an isolated overlay without replacing concurrent
  shared-tree changes. The unchanged v1 verifier and a CPU preflight now accept
  all 25 frozen source identities, 359,823 dense target rows, all four official
  pi0.5 batch-16/50k configs, and both final checkpoints. The only authorized
  runtime amendments select that overlay and write protocol audits to a
  caller-writable directory; their hashes are pinned in
  `pi05_r1_frozen_overlay_amendment_v1.json`.
- Canonical scheduler state at 05:49 UTC: 421 tasks total, 293 completed,
  90 disabled, 35 pending, and 3 scheduler-managed running tasks. Combined
  runs on local 2 GPU (PID 94791) and had completed 14/24 task-seed cells;
  CRAVE-only and shuffled are now complete at 24/24 cells with frozen-manifest
  macro success rates of 68.67% and 64.67%, respectively. These two arms alone
  cannot trigger the R1 gate. Earlier shuffled
  (`t-20260805082722-mltdq`), CRAVE-only (`t-20260805083104-zm26d`), and
  zero-route (`t-20260805082725-rkqtc`) East attempts failed before rollout.
  They passed the frozen verifier and loaded their checkpoints, but four
  simulator seeds concurrently ran `apt-get`, contending on the package lock
  and shared apt temporary directory. After serializing package installation,
  the next failure exposed cuRobo being JIT-compiled for A100 compute
  capability 8.0 on H20, producing `no kernel image is available`; later
  `left_planner` errors were consequences of the failed first construction.
  These are infrastructure failures, not control evidence. Renderer setup now
  uses `flock` and rechecks dependencies under the lock. East candidates use
  `TORCH_CUDA_ARCH_LIST=9.0`, the previously validated shared H20 extension
  cache at `/vePFS/tim/runtime/torch_extensions/h20_sm90_py310`, and
  `ROBOTWIN_ATTACH_REQUEUE_FAILED=1`, preserving completed cells while
  requeueing only failed cells. The two cache-less attempts were stopped after
  producing no cells. Corrected CRAVE-only (`t-20260805085511-2l2z9`) and
  shuffled (`t-20260805085627-q9svv`) were submitted only after fresh target
  recommendation audits and completed on East without repeating the H20
  kernel-image error. Corrected zero-route
  (`t-20260805114418-5twhg`) runs on East and had reached 16/24 cells with zero
  failures. Because the shared per-seed schedulers use `fcntl` locking,
  heartbeat leases, and atomic task claims, a fresh 4-GPU routing audit selected
  the four free East GPUs for a combined attach helper. The helper
  (`t-20260805135315-t9fmq`) started at 05:53 UTC, passed the frozen verifier,
  and attached without duplicate claims. After the distinct helper claims were
  visible, the original two local masters were interrupted cleanly, released
  their claims, and exited before report finalization. Canonical scheduler state
  now monitors the East helper directly, preventing an unintended local restart
  or concurrent report write. East is now 8/8 occupied, local is 0/2 occupied,
  and Beijing primary is 24/25 with the repaired P1 pair. At 06:25 UTC the
  East runs remained healthy; combined had reached 15/24 cells and zero-route
  20/24, with active scheduler heartbeats and no new failed cell. Source and amendment hashes are enforced
  before recommendation and submission; the overlay authorization is limited
  to these four seed-1000 R1 evaluations and cannot authorize R1 training or
  later seeds. P2 remains blocked until the P1 causal gate is complete and
  accepted. A separate readiness audit found that the two P2 replication
  training nodes incorrectly declared the 97 GB, 36,979-file frame cache as a
  regular file. The scheduler now has an explicit `ready_dirs` contract and
  both nodes use it, preventing a permanent false-not-ready state after the P1
  gate. The shared source still differs from the frozen P1/R1 identity. A new
  gate-conditioned frozen-training amendment now separates the immutable
  verification and Python source root from the canonical data and checkpoint
  output root. Its independently materialized overlay passes the unchanged P1,
  P2, and R1 verifiers over 25 R1 source identities and 12 P2 protocol files.
  The amendment authorizes only the listed seed-1001/1002 training nodes and
  cannot create either prerequisite gate. After scheduler reload, those training
  nodes have zero source-hash failures and remain unready solely because the
  P1/R1 gate files do not exist; no hash check has been relaxed. The two
  conditional P2 evaluation nodes now also have a separate hash-pinned
  amendment: verification and Python imports use the immutable overlay while
  checkpoints, datasets, and reports remain in the canonical repository. Both
  local and East launch paths have zero source-hash failures, but still require
  the original P1 gate and their seed checkpoint before dispatch. The complete
  scheduler/router and overlay suite passes 119 tests. At 08:30 UTC the repaired
  P1 pair remained healthy at A0 step 35,300 and candidate step 35,600, both at
  approximately 1.3 steps/s. The scheduler now discards only materializer
  failures accumulated before a North parent completes, so the final P1
  checkpoints can be synchronized without inheriting the obsolete exhausted
  retry budget. At 08:42 UTC the R1 correct-combined evaluation completed all
  24 cells and 1,200 episodes. Its macro success rate is 62.92%, versus 68.67%
  for CRAVE-only: the paired difference is -5.75 points with 95% CI
  [-9.00, -2.50]. Correct-combined is also statistically indistinguishable
  from zero-route (+0.75 points, CI [-2.42, 3.92]) and shuffled-action
  (-1.75 points, CI [-4.92, 1.42]). The predeclared R1 gate therefore already
  fails the necessary CRAVE comparison and seed-1001/1002 R1 replication must
  not launch. The formal four-arm gate report still waits for the matched P1 A0
  and predictive reports so that the complete rejected result is archived.
  Current owned allocation is North 8 GPUs, East 4 GPUs, and local 0/2 GPUs;
  there is no gate-ready pending work for the idle devices. New robot-task
  submissions are disabled by operator policy and that resource remains the
  final, ineligible fallback.
  The six post-readout R2 execution nodes are now explicitly disabled with the
  reason `R2 causal readout gate rejected`; they can no longer be mistaken for
  unfinished or future-runnable work. gf1 retirement is enforced at queue,
  capacity, monitor, and launcher layers.

## P1: seed-1000 closed-loop causal gate

- [ ] Resume the pending current-source A0 and full candidate from
  official pi0.5 at seed 1000 with matched data, normalization, batch size,
  update count, and source snapshot. The accepted historical A0 is not
  reusable under the completed identity audit.
- [ ] Evaluate candidate normal, zero-gate, action-shuffled, and action-masked
  conditions on the frozen 24-cell scene manifest. Normal must exceed A0 and
  all three controls, with no task regression larger than 5 points.

## P2: conditional replication and efficiency

- [ ] If and only if P1 passes, train candidate seeds 1001 and 1002 under the
  frozen recipe and complete both 24-cell evaluations.
- [ ] Run the predeclared hierarchical paired analysis over training seeds and
  episodes. A utility claim requires the 95% interval for candidate minus A0
  to exclude zero; report every task and seed effect.
- [ ] Only if P2 passes, measure parameters, FLOPs, peak training memory,
  direct and WebSocket latency, and throughput against A0 on matched hardware.

## R1: recurrence-aligned predictive adapter

- [x] Finish corrected CRAVE-only and combined seed-1000 training on the frozen
  dense-target protocol. Treat losses and intermediate checkpoints as health
  telemetry only.
- [ ] Complete the seed-1000 four-arm closed-loop comparison: current-source
  A0, predictive adapter, CRAVE-target auxiliary only, and predictive plus
  CRAVE targets. Evaluate zero-route and shuffled-action interventions on the
  same frozen 24-cell scene manifest and report every task.
- [ ] Replicate seeds 1001/1002 only if the combined arm exceeds A0 and both
  single-component arms, survives zero-route and shuffled-action interventions,
  and has no task regression larger than 5 points.

## R4: deferred outcome-calibrated improvement

The R4 input audit is implemented in
`lmvla/lmwm/scripts/audit_pi05_r4_outcome_dataset.py` and has three passing
tests. The current R0 artifacts contain true terminal outcomes and visual
videos, but not aligned per-step actions and states; they support diagnosis,
not AWBC/AWR training. R4 additionally requires a six-task, scene-disjoint
train/eval manifest with behavior-policy identity, hashed videos, and finite,
aligned action/state/frame trajectories with success and failure support for
every training task. No R4 training node is scheduled.

- [ ] Do not launch until success and failure rollouts with true outcome labels
  exist. Compare CRAVE-AWBC/AWR against outcome-free CRAVE labels and ordinary
  pi0.5 fine-tuning.
- [ ] Do not make a Q-value, advantage, world-critic, or model-predictive-control
  claim without action-diverse rollout consequences. Expert demonstrations and
  action-shuffled latent probes are insufficient.

## Current stop rules

- P1 failure closes predictive-adapter replication.
- P2 failure retains the current negative-integration paper and archives the
  predictive-adapter branch as another bounded interface result.
- R4 remains deferred until true rollout outcomes exist. CRAVE progress alone
  must never be reported as reward, action advantage, or control value.
- AHEAD-style interception and world-critic RL require new benchmarks or reward
  data and remain outside this graph.
- LeWM/DINO visual replacement, from-PaliGemma initialization, A2/A3 variants,
  and closed MT3--MT6 tasks must not be scheduled from this TODO.
