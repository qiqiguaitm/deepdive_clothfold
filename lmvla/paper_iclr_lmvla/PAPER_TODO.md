# pi0.5-Preserving Predictive and Recurrence-Aligned Control TODO

Updated: 2026-08-05 14:12 UTC

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
  At 10:20 UTC the repaired P1 pair remained healthy: A0 reached step 44,000
  and the candidate reached step 44,400 at approximately 1.3 steps/s, with
  estimated completion in 1 hour 10 minutes to 1 hour 18 minutes. Current
  owned allocation is North 8 GPUs and East 4/8 GPUs; local 2/2 GPUs became
  free after the predeclared R4 support collection completed. New robot-task
  submissions are disabled by operator policy and that resource remains the
  final, ineligible fallback.
  The six post-readout R2 execution nodes are now explicitly disabled with the
  reason `R2 causal readout gate rejected`; they can no longer be mistaken for
  unfinished or future-runnable work. gf1 retirement is enforced at queue,
  capacity, monitor, and launcher layers.
  At 11:37 UTC the candidate had written a complete step-49999 checkpoint, but
  A0's final Orbax save failed with `EDQUOT` after reaching step 49999. The
  remote gate correctly remained locked at one of two complete metadata files.
  An audited cleanup retained candidate steps 10000/49999 and A0 steps
  10000/45000, removed only redundant intermediate checkpoints and the failed
  temporary save, and released 445 GB. A fresh recommendation selected North
  for retry `t-20260805194337-htmjv`; the candidate arm is skipped because its
  final metadata is complete, while A0 restored step 45000 and reached step
  45400 at approximately 1.3 steps/s by 11:54 UTC, with about 59 minutes
  remaining. The currently running platform reservation is eight GPUs although
  only the four-GPU A0 arm is active; it will not be interrupted because that
  would consume another platform retry. Future recovery selection now prefers
  the minimum immediately runnable four-GPU shape.

At 12:55 UTC the audited North recovery completed both seed-1000 arms. The
candidate final checkpoint was already complete; A0 restored step 45000,
reached step 49999, and completed its final Orbax save without another quota
error. The first local materialization attempt then exposed a false completion
contract: the training report had arrived before either checkpoint copy, so a
failed copy was incorrectly classified as complete. Materialization now
requires an independent marker written only after both complete checkpoint
trees pass metadata checks. The legacy report-only state was reopened, and
root-owned canonical directories are preserved under dated archive names
before writable destinations are created. Six-way resumable OCDBT shard
transfer increased measured throughput from approximately 3.5 MB/s to 16.9
MiB/s. The A0 final checkpoint is currently being materialized; all five P1
closed-loop conditions remain blocked until both local checkpoint trees exist.

## P1: seed-1000 closed-loop causal gate

- [x] Resume the current-source A0 and full candidate from
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
- [x] Close seed-1001/1002 replication without launching it: combined is
  significantly worse than the CRAVE-only arm, so the preregistered necessary
  condition is false regardless of the still-pending A0/predictive comparisons.

## R4: deferred outcome-calibrated improvement

The R4 input audit is implemented in
`lmvla/lmwm/scripts/audit_pi05_r4_outcome_dataset.py` and has three passing
tests. The current R0 artifacts contain true terminal outcomes and visual
videos, but not aligned per-step actions and states; they support diagnosis,
not AWBC/AWR training. R4 additionally requires a six-task, scene-disjoint
train/eval manifest with behavior-policy identity, hashed videos, and finite,
aligned action/state/frame trajectories with success and failure support for
every training task. The historical public-policy collector produced videos but
no trajectories and was correctly rejected. A new collector is isolated from
the P1/P2 frozen runtime in an overlay materialized from lawam commit
`865e0b6` plus a hash-pinned patch. It atomically records aligned action, state,
and frame-index arrays without changing shared evaluation sources. The frozen
scene protocol contains 24 cells and 240 episodes: eval seeds 0/1 form train,
2/3 form eval, with scene identities disjoint before collection. A one-task,
  two-episode local smoke completed in 6 minutes 25 seconds after a fresh
  recommendation selected local GPU 0. It produced one success and one failure,
  two videos, and two finite aligned trajectories; the 400-step failure has
  action/state shape `(400, 14)` and frame indices 0--399. The first formal East
  submission (`t-20260805171632-rd8m2`) failed before collection because the
  container lacked system ffmpeg. The launcher now pins a tested ffmpeg binary
  on the shared vePFS and emits explicit prerequisite errors. After a new
  recommendation audit, `t-20260805172352-gjqqn` ran on four East H20 GPUs and
  completed all 24 cells and 240 episodes at 10:18 UTC. Every sampled trajectory
  has finite aligned 14-dimensional actions and states with strictly increasing
  frame indices. The first train-split task exposed a support gap:
  both train seeds were 10/10 on `beat_block_hammer`, while the one observed
  failure belongs to the eval split and cannot be moved into training. A
  separately frozen amendment therefore predeclares all 40 remaining source
  scenes for each train seed, retaining all 80 outcomes rather than stopping
  after a failure. A fresh recommendation selected the two local GPUs and the
  supplemental collector started at 09:51 UTC and completed all 80 retained
  scenes at 10:10 UTC. Seed 0 produced 37 successes and 3 failures; seed 1
  produced 38 successes and 2 failures, so hammer now has both outcomes in the
  train split without moving eval data or stopping after the first failure.
  Base collection completion is separated from
  training authorization: the immutable base manifest prevents redundant
  collection retries, while a second hash-pinned node merges all base and
  supplemental records, rejects policy or scene-identity mismatches, rebases
  artifact paths, and runs the unchanged R4 input audit. That first combined
  audit retained all 320 records and validated every trajectory, policy hash,
  and split identity, but correctly rejected training: `stack_blocks_two` was
  20/20 successful in the train split and had no failure support. Rather than
  add only another imbalanced task, a second frozen amendment predeclares all
  400 unused train scenes for the remaining five tasks. Two disjoint East
  two-GPU shards will produce exactly 100 retained train trajectories per task
  when combined with the base and hammer supplement; no shard may stop or
  filter based on observed outcome.

  A trainability audit then found a second, distinct requirement that the v1
  outcome audit intentionally did not prove. The public recipe consumes
  `cam_high`, `cam_left_wrist`, and `cam_right_wrist` at every real 50-step
  policy query, whereas v1 retains only the head-camera evaluation video. That
  video is action-aligned but repeats the last queried head image within each
  open-loop chunk; it cannot reconstruct the two wrist observations. A frozen
  query-observation amendment now captures the exact three images, 14-D query
  state, instruction, and frame index without modifying the v1 collector. Its
  audit requires all 600 predeclared train records to match the accepted outcome
  manifest exactly, verifies every 50-step query/action alignment, and preserves
  all success and failure scenes. A 2-episode local smoke is followed, only
  after the combined outcome gate, by a four-GPU East base collection split
  across two disjoint three-task shards and a parallel two-GPU local hammer
  supplement plus a four-GPU East balanced-support query collection. The first
  query smoke completed both rollouts but emitted no query files because the
  shared RoboTwin dependency shim's `sitecustomize.py` shadowed the capture
  hook; it was rejected with rc=14 and created no marker. A dedicated
  hash-pinned RoboTwin Python wrapper now installs the hook before running the
  bridge and clears an inherited `OUTPUT_ROOT`; `lawam/results` is itself the
  intentional symlink to `lawam_local/results`, so that path is not a data
  split. A corrected smoke reran both episodes but still emitted no query files,
  exposing the remaining import path: RoboTwin's `class_decorator` loads task
  modules through `importlib.import_module`, bypassing a hook that only wrapped
  `builtins.__import__`. The hook now covers both standard import paths, restores
  them immediately after patching `Base_Task`, and has a direct dynamic-import
  regression test. A v2 smoke confirmed that the bridge parent used the
  dedicated wrapper, but its spawned simulator slot still loaded the shared
  dependency `sitecustomize` first and emitted no query files. The query hook
  directory is now first on `PYTHONPATH`; its own `sitecustomize` preserves the
  required Warp compatibility shim and installs the capture hook in every
  fresh spawn interpreter. A subprocess regression test verifies this exact
  startup path. The corrected v2 retry then produced both three-camera files.
  One 400-step episode had exact query frames `[0,50,...,350]`; a successful
  112-step episode also captured the terminal success refresh at frame 112.
  That refresh is not a policy query and the strict alignment check rejected
  it. The v3 hook now retains only frames divisible by the frozen 50-step
  replan horizon, with a terminal-refresh regression test. A separate one-cell
  smoke scene manifest fixes finalization incorrectly expecting all 24 formal
  cells. The clean v3 smoke produced two records with exact frames
  `[0,50,100]`; all three uint8 cameras and 14-D states align with the retained
  action trajectories, and the smoke marker was created at 11:21 UTC. Earlier
  rejected attempts remain quarantined and produced no marker. The focused
  spawn-hook/query/scheduler suite passes 109 tests. R4 policy training
  remains unscheduled and unauthorized until both the combined outcome audit
  and the three-camera query audit are accepted.

  At 11:55 UTC the balanced B shard had completed all 160 retained episodes and
  emitted its marker. The A shard had completed both 40-episode ranking tasks
  for both train seeds and was collecting the final `handover_block` cells at
  episodes 21/40 and 26/40; it remained healthy on four East H20 GPUs. Once its
  second marker appears, the scheduler will locally merge and audit exactly 720
  outcomes before any formal query collection can start. An audited local
  builder now converts an accepted 600-record query manifest into ordinary and
  task-normalized terminal-outcome-weighted 50-step action chunks. It is gated
  on both outcome and three-camera query markers, runs on zero GPUs, and cannot
  authorize policy training. Its focused R4 suite passes seven tests and the
  complete scheduler suite passes 105 tests. Terminal-outcome weighting remains
  explicitly distinct from action advantage, Q-value, or world-critic
  estimation.

  The balanced A shard completed at 12:05 UTC. The finalizer initially remained
  exhausted by three correct rejections from the pre-amendment input epoch; the
  scheduler now preserves those attempts but excludes failures older than the
  declared balanced-A readiness marker. A separate permission defect was also
  closed by writing derived manifests under the local `logs` tree rather than
  root-owned East result directories. The rerun accepted all 720 records and
  352,608 transitions: the train split contains exactly 100 episodes per task,
  with success/failure counts of 95/5 (hammer), 93/7 (ranking RGB), 70/30
  (ranking size), 49/51 (handover), 71/29 (stack-three), and 93/7 (stack-two).
  Three fresh recommendation audits then selected the only data-local eligible
  targets. Balanced query collection `t-20260805201041-zlxtb` and base query
  collection `t-20260805201044-9qrsz` now occupy all eight East H20 GPUs, while
  the hammer query collector runs on both local GPUs as PID 311677. All three
  initialized their frozen scene schedulers and began writing under distinct
  result roots. North remains occupied by the P1 recovery; A0 reached step
  46800 at 12:12 UTC with approximately 41 minutes remaining. The scheduler
  inventory is now 288 completed, 120 disabled, 20 gated pending, and four
  running tasks: three platform tasks plus one local task in the active R4/P1
  DAG. The
  17 obsolete R1 seed-1001/1002 replication/gate nodes are explicitly disabled
  because the seed-1000 necessary comparison was rejected. The scheduler suite
  passes 106 tests.

  At 13:36 UTC the formal query collection has completed base (120 records),
  hammer support (80 records), and balanced-support B (160 records). Balanced
  A has written 208 of its required 240 query records and remains healthy on
  four East H20 GPUs. The finalizer can now rebuild shard-local manifests from
  immutable rollout roots when a pre-fix East process wrote derived manifests
  into a root-owned directory. R4 data materialization is pinned to a new
  LeRobot 0.6.1 training environment rather than the incompatible LeRobot 0.1
  environment. A direct-chunk compatibility test found and closed three
  launch blockers: explicit dataset finalization is required in 0.6.1, the
  public processor must use the local PaliGemma tokenizer plus pinned
  SentencePiece/Protobuf dependencies, and `sample_weight` must be admitted to
  LeRobot's complementary-data contract or preprocessing silently drops it.
  The opt-in runtime now preserves the public normalization processors and
  verifies the weight field after the complete official pi0.5 preprocessing
  pipeline. This runtime preparation does not authorize R4 policy training;
  At that point, exact public-checkpoint loading and an accepted 600-record
  query audit were still required.

  At 13:51 UTC the 600-record query gate passed after resolving an important
  rollout-identity issue. Independent query reruns are stochastic, so five
  terminal outcomes and trajectory lengths differed from the earlier outcome
  collection; attaching the earlier labels was correctly rejected. The
  predeclared query reruns now use their own aligned trajectories and terminal
  outcomes, with no scene or outcome filtering. The train-only outcome audit
  accepts all 600 records and 303,772 executed actions. The query audit accepts
  all six tasks and 6,313 executable policy queries with exact three-camera,
  state, action, scene, and behavior-policy alignment. Five additional planning
  requests occurred exactly at `frame == len(executed_actions)` after the final
  action chunk; they are retained in source evidence but excluded from training
  because no action was executed. Any other out-of-range query remains a hard
  rejection. Action-chunk and LeRobot materialization may now proceed; exact
  public-checkpoint loading against the resulting dataset remains the final
  runtime gate before any R4 training can be designed or authorized.

  At 14:12 UTC the direct-chunk LeRobot 0.6.1 materializer had completed 408
  of 600 rollout episodes and 4,198 of 6,313 query samples without an error.
  Exact public-checkpoint loading remains downstream of that atomic build. A
  separate matched-weighting protocol now closes the previously undefined
  outcome-free control: it first projects the accepted query manifest onto a
  form that contains no success, reward, return, trajectory, or video field;
  a frozen DINOv3-base CRAVE teacher then assigns progress change only between
  consecutive real query observations from the same rollout. Ordinary,
  terminal-outcome, and outcome-free CRAVE arms share the identical ordered
  6,313 image/state/action chunks. Their only difference is respectively no
  weighter, task-normalized exponential terminal-return weights, or the same
  temperature and normalization applied to CRAVE progress changes. A final
  query without a subsequent observation receives neutral raw weight and is
  marked unlabeled rather than receiving a fabricated future target. Sidecar
  task, scene, query-index, and frame arrays must exactly equal the action
  chunks, and the matched runtime must prove sidecar length equals dataset
  length while loading the exact public policy. This protocol only authorizes
  data preparation; policy training remains blocked on both runtime gates and
  the accepted sidecar report.

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
