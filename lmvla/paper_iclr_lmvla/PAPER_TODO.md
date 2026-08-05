# pi0.5-Preserving Predictive and Recurrence-Aligned Control TODO

Updated: 2026-08-05 18:35 UTC

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

At 14:50 UTC both final P1 checkpoint trees are materialized locally. The
frozen seed-1000 gate evaluation is active without failed cells: local A0 has
completed 2/24 task-seed cells, while the four-GPU East normal and action-masked
arms have each completed 4/24. Those two East tasks occupy all eight H20 GPUs;
zero-gate and action-shuffled remain pending for the next eligible allocation.
Every platform launch has a saved recommendation audit. The earlier training
and materialization statements remain as execution provenance rather than
current status.

At 15:15 UTC all five P1 gate conditions are executing concurrently. A frozen
North evaluation amendment versions the previously untracked P1 launchers,
pins the North H20 YAML and supporting evaluator files, and requires both a
75-file source-overlay inventory check and the unchanged P1 source-freeze
verifier. The first zero-GPU stage attempt correctly rejected East-targeting
overlay symlinks on North. The repaired stage transferred their immutable file
contents, verified 17 runtime files, passed the source and inventory checks,
and wrote independent local and North readiness markers. Only then did the
scheduler run fresh recommendation audits. Both audits selected
Robot-North-H20, after which shuffled
(`t-20260805231243-j8c46`) and zero-gate
(`t-20260805231248-57hn4`) started on four H20 GPUs each. Both jobs verified
that imported `openpi` resolves inside the frozen overlay, passed checkpoint
and normalization checks, and created four seed schedulers with four active
cells and zero failures. At the same observation point, local A0 had completed
2/24 cells, East normal 8/24, and East action-masked 7/24, all without failed
cells. North results cannot satisfy the P1 gate directly: each result must be
hash-preservingly synchronized back, pass the local fixed-scene verifier, and
produce the canonical local report and marker. The R4 CRAVE sidecar remains
ready but waits for a data-local East or local allocation while these higher
priority P1 evaluations occupy all ten such GPUs.

At 15:40 UTC the five-arm evaluation remains healthy and fully parallel. Local
A0 has completed 3/24 cells, East normal 12/24, East action-masked 11/24, and
North action-shuffled and zero-gate 4/24 each, with zero failed cells in every
arm. The North controls still require reverse synchronization and local
verification before they count toward the gate.

At 15:53 UTC no arm has a failed cell; local A0 is 4/24, East normal and
action-masked are 12/24 each, and both North controls remain 4/24. To shorten
the North bottleneck without changing any evaluation evidence, an audited
accelerator amendment adds one attach-only worker per existing seed scheduler.
It preserves the frozen overlay, checkpoint, fixed-scene manifest,
intervention, 50-episode cells, and canonical 24-cell verifier, while using a
disjoint worker-index range. Fresh recommendation audits independently chose
Robot-North-H20 for shuffled and zero-gate. The corresponding four-H20 tasks
`t-20260805235042-72pqb` and `t-20260805235047-8b2kx` are queued in Beijing and
will attach only when physical North capacity becomes available. They were not
redirected to Shanghai and cannot write a canonical marker before all 24 cells
pass the unchanged verifier.

The slower local A0 path also has a conditional East accelerator, but it is
not yet eligible and has produced no submission audit. It waits for the
original evaluator to create seed-2 and seed-3 schedulers, then assigns two
independent attach workers to each seed so all four requested H20 GPUs perform
useful work. If A0 finishes naturally first, its canonical marker makes the
accelerator a no-op. At 16:00 UTC shuffled has advanced to 6/24; A0 remains
4/24, normal and action-masked 12/24 each, and zero-gate 4/24, with zero failed
cells throughout.

At 16:16 UTC all five canonical schedulers still have fresh heartbeat leases
and zero failed cells. A0 is 4/24; normal and action-masked are 16/24 each;
action-shuffled and zero-gate are 8/24 each. The East arms are executing
`stack_blocks_two` and have only the four per-seed `stack_blocks_three` cells
left after the current claims. The two North accelerator jobs remain in
`Queueing`, so they reserve submitted-job slots but consume no additional GPU.
No Shanghai task was added while East is full and robot-task submission remains
disabled. The resource recommendation CLI, static topology catalog, tests, and
submission documentation are now tracked on `main`; the router/scheduler suite
passes 123 tests from a clean Git identity.

At 16:29 UTC normal and action-masked have each completed 20/24 cells; all
four active claims in each arm are the final `stack_blocks_three` cells, with
no pending or failed cell. Action-shuffled and zero-gate have each reached
12/24 with four healthy active claims. A0 remains 4/24 with two active ranking
cells. East therefore remains 8/8 only until the two final four-cell waves
finish; R4 smoke remains unsubmitted and has no recommendation audit yet. The
scheduler must take a fresh resource snapshot and write that audit after East
capacity is actually released.

At 16:41 UTC A0 advanced to 6/24 with zero failures; its seed-0 and seed-1
schedulers each have one active `handover_block` cell and two pending stacking
cells. A hash-pinned partial-only East helper is now staged to claim those four
pending cells with disjoint worker and port ranges once a four-GPU East slot is
available. It cannot write the canonical A0 report or marker, and its DAG node
also treats an already-existing canonical marker as a no-op completion. The
helper has not been submitted and has no recommendation audit while East
remains 8/8; dispatch requires a fresh router decision after capacity release.
Readiness diagnosis then found that both East A0 accelerator nodes referenced
the North overlay `READY` path even though their containers use the local East
overlay. The DAG now uses the matching East path for both nodes. The seed-0/1
helper has no missing input and reports `ready=true`; the seed-2/3 accelerator
remains blocked only on its two not-yet-created scheduler files. This correction
does not weaken a hash gate or alter either amendment.

At 16:58 UTC normal and action-masked are both canonical 24/24 results with
zero failed cells. Each platform parent was closed only after its shared
artifact marker existed; no North copy was required. The first released
four-GPU East slot produced a fresh recommendation audit selecting data-local
Robot-East-H20 and launched the R4 exact-runtime smoke as
`t-20260806005236-qz75g`. The second released slot independently produced a
fresh audit and launched the partial-only A0 helper as
`t-20260806005709-l9prw`. R4 passed dataset construction and entered policy
initialization. The helper passed overlay/checkpoint checks and activated four
disjoint workers; across seed 0/1, A0 now has 6 completed, 5 active, 1 pending,
and zero failed cells. East is again 8/8 with exactly these two four-GPU jobs.

The first two R4 smoke attempts failed before any optimizer step and wrote no
marker. The second had already been submitted with the old launcher before the
runtime repair was committed and reproduced the identical error.
The exact error was an offline lookup for
`google/paligemma-3b-pt-224`: the runtime verifier exercised the patched policy
factory, but `lerobot_train.py` retained its module-level reference to the
unpatched processor factory and therefore ignored the local tokenizer override.
This is infrastructure evidence only. Automatic dispatch was paused before the
retry cooldown expired. A new fail-closed training entrypoint now verifies the
R4 patch marker, binds the training module to the patched factory, and emits a
sentinel before accelerate launch. The smoke amendment pins this entrypoint,
the sitecustomize overlay, and the repaired launcher. The exact LeRobot binding
probe passes, all three R4 runtime tests pass, and the 123 scheduler/router tests
remain green. At 17:10 UTC the A0 helper has all six remaining seed-0/1 cells
actively claimed with zero pending or failed cells.
The smoke DAG now uses the amended protocol file as a rearm epoch. Its mtime is
later than both old-runtime failures, so they do not consume the repaired
runtime's failure budget; any attempt finishing after this epoch is still
counted normally.

At 17:24 UTC the repaired R4 smoke completed two real optimizer steps on four
East H20 GPUs. The entrypoint emitted the binding sentinel, loaded the public
checkpoint and local tokenizer entirely offline, and ended with finite losses
0.011/0.013 and nonzero gradient norms 0.367/0.395. The first compiled update
took 297.228 seconds; the second steady-state update took 1.731 seconds at
45.74 GiB peak memory. The exact marker, generated config, log, runtime patch,
launcher, builder, matched protocol, and three formal YAMLs are pinned by a
separate formal-training amendment. That amendment authorizes only the matched
seed-1000, 5000-step ordinary, terminal-outcome, and outcome-free CRAVE screen;
replication and policy-effect claims remain blocked on fixed-final-checkpoint
closed-loop evaluation.

At 17:33 UTC the scheduler took a fresh resource snapshot after the A0 East
helper released its allocation. Two independent recommendation audits selected
the data-local Robot-East-H20 queue with eight immediately available GPUs.
Ordinary training launched as `t-20260806013323-snsrr`, and outcome-free CRAVE
launched as `t-20260806013326-xv5tk`; terminal-outcome remains pending rather
than queueing in Shanghai. Both active arms passed the exact config preflight
and emitted `R4_TRAIN_ENTRYPOINT_BINDING_OK` before entering first-step
compilation. A separate preregistered evaluation protocol now fixes final step
5000, the six tasks, four eval seeds, 50 episodes per cell, and the exact
24-cell scene manifest before any trained checkpoint exists. Its DAG prefers
an immediately available four-GPU East allocation but can use the two local
GPUs in two seed waves after P1 releases them; either route must first produce
a fresh recommendation audit and must pass the same 1,200-episode verifier.
The paired gate implementation is also frozen before evaluation: it verifies
identical task/eval-seed/scene-seed keys, reports every task delta and a paired
task/seed/episode hierarchical bootstrap interval, and accepts seed 1000 only
if terminal-outcome macro success exceeds both ordinary and outcome-free CRAVE.
The intervals are descriptive and do not retroactively change that gate.

At 17:47 UTC all four candidate/control evaluations are canonical 24/24
results with zero failed cells. Their six-task macro success rates are normal
82.42%, zero-gate 78.50%, action-shuffled 81.17%, and action-masked 78.00%.
Normal therefore exceeds all three interventions, but its margin over shuffled
actions is only 1.25 points and is not interpreted before the preregistered
paired gate runs. North materialization verified every copied file and the
frozen scene manifest. A0 seed 0/1 are also complete at 12/12 cells with zero
failures; seed 2/3 are now active on the two local A100 GPUs. The final P1 gate
remains blocked only on those 12 A0 cells and cannot be inferred from the four
candidate/control macros alone.

At 18:08 UTC the two local A100 evaluators remain healthy and fully occupied.
A0 seeds 0/1 are complete; seeds 2/3 have each completed `beat_block_hammer`,
are running `blocks_ranking_size`, retain four pending tasks, and have zero
failed cells. The four completed candidate/control arms therefore still do not
unlock the P1 or R1 gates. On East, matched R4 ordinary and outcome-free CRAVE
training have both reached step 1,838/5,000 on disjoint four-H20 allocations.
Their steady updates remain approximately 0.86--0.94 seconds with 40.25 GiB
per process and negligible data-loader time. East is 8/8 occupied; the
terminal-outcome arm waits for the first four-GPU release and has not been
blindly queued. Migrating it to North was rejected after audit because that
filesystem lacks the approximately 8.8 GB public model, 7.6 GB pinned LeRobot
runtime, and 0.9 GB R4 dataset; staging and revalidating those dependencies
would exceed the remaining time of the active East arms. The verified North
failover monitor now bounds displayed progress at 100% while retaining the
actual staged byte count.

At 18:35 UTC A0 is executing through independent first-completer paths without
changing its frozen evidence. The original local seeds 2/3 evaluator was
cleanly interrupted after each seed completed `beat_block_hammer`; both active
claims were released with zero failed cells. A fresh recommendation then chose
the already verified North stage and launched the four-H20 parent as
`t-20260806021522-2zksv`. All four remote seed schedulers have completed one
cell and are active. In parallel, a hash-pinned v2 local accelerator received
a separate recommendation selecting the two local A100s. Its first attempt
failed before rollout because a root-owned log directory was not writable and
exposed a scheduler bug that treated any task without `completion_glob` as
complete even when explicit `completion_locations` were present. The completion
contract now requires those declared artifacts, repairs the false historical
state on reload, and has dedicated regression coverage. The corrected local
attempt uses two workers per GPU with disjoint indices 4000/4001 and 5000/5001;
it has 14/24 cells complete, four active, six pending, and zero failed. Local
memory is approximately 28 GiB per A100 and host load is approximately 54/56,
so further local oversubscription is not authorized. A second audited North
attach-only helper, `t-20260806023359-ft7sh`, passed the frozen overlay,
official A0 config, exact checkpoint, and four-scheduler preflight. It adds
worker 2000 to each remote seed scheduler while preserving their locked task
queues. A separate East attach node remains pending at priority 0 and will use
the first released four-H20 slot; the R4 terminal-outcome arm remains priority
1 for the other slot. Ordinary and outcome-free CRAVE R4 training have reached
approximately step 3,600/5,000 with finite losses and gradients. A redundant
zero-gate North helper submitted under the old completion contract was stopped
and accelerator tasks now close immediately when their canonical condition
marker already exists.

At 19:06 UTC the A0 canonical result root remains healthy at 14/24 completed
cells: seeds 0 and 1 are complete, while seeds 2 and 3 each have one completed,
four active, one pending, and zero failed cells. The local two-A100 accelerator,
the four-worker North attach helper, and the four-worker East attach helper
operate on disjoint worker indices and all report fresh heartbeats. R4 ordinary
and outcome-free CRAVE both completed the exact fixed step-5,000 checkpoint
with finite final state. The terminal-outcome arm was selected by a fresh
submission recommendation, launched as `t-20260806025734-dbx5h`, passed the
unchanged runtime checks, and reached step 64 after compilation at approximately
0.79 seconds per steady-state step. The two completed platform checkpoints
exposed a reproducibility issue: tensor files are root-owned mode 0600 even
though their directories and metadata are readable. A hash-pinned operational
amendment therefore adds a priority-0 East one-H20 permission-normalization
task after all three training markers exist. It may only add read/traverse
bits, records all three model SHA256 digests and step values, and emits a marker
required by every formal R4 evaluation. It cannot change checkpoint bytes,
checkpoint selection, evaluation parameters, or any scientific gate. The task
is not yet ready and therefore has not been submitted; the scheduler must run a
fresh recommendation audit once terminal-outcome training completes.

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

  At 14:50 UTC the atomic LeRobot build is complete: 600 episodes and 6,313
  executable query-action samples. The strict basic runtime gate now loads the
  exact 4,143,404,816-parameter public pi0.5 checkpoint, preserves the public
  normalization and scalar sample-weight fields, verifies action shape
  `(50,14)`, and pins `accelerate==1.14.0`, `protobuf==7.35.1`, and
  `sentencepiece==0.2.1`. The first CRAVE-sidecar attempt exposed an ordering
  error: query annotations were grouped by task while action chunks retained
  frozen manifest order. A local exact reproduction additionally showed five
  terminal planning observations with no executed action target. The repaired
  builder now joins by the unique `(task, scene_seed, query_index,
  query_frame)` key, permits dropping only those five explicitly unlabeled
  terminal observations, and normalizes weights over the final 6,313 training
  samples. Its focused scheduler/runtime suite passes 139 tests.

  At 15:38 UTC the sidecar and matched-runtime gates are accepted. A dedicated
  North amendment first materialized exactly 600 query images and 1,200
  selected reference features in an isolated stage, verified all 1,920 files
  by SHA-256, and passed the pinned DINOv3-base model check. A fresh submission
  audit selected Robot-North-H20, and the one-H20 task
  `t-20260805233129-dqxfd` encoded all 6,318 query observations. It emitted
  6,313 action-aligned weights after dropping only the five predeclared
  non-actionable terminal queries. Reverse materialization accepted sidecar
  SHA-256 `071a926a087e73a279148b403d4607ddd9d0001b37db7466c87ea8970d181c8e`.
  The local matched-runtime gate then loaded the exact 4,143,404,816-parameter
  public policy and accepted the 600-episode, 6,313-sample dataset, `(50,14)`
  actions, sample-weight processor, and sidecar indexing.

  A separate hash-pinned `smoke_only` amendment now authorizes only the
  ordinary arm for two optimizer steps on four East H20 GPUs. It preserves
  global batch 16 as per-process batch 4, the public AdamW and scheduler,
  unfrozen vision encoder, gradient checkpointing, and `max-autotune` compile
  settings. The scheduler may submit it only after a fresh target audit and an
  immediately available East allocation; it must not queue blindly in
  Shanghai. Successful smoke execution still does not authorize the 5,000-step
  ordinary, terminal-outcome, or outcome-free CRAVE arms. Formal three-arm
  training requires a separate amendment after smoke acceptance.

- [x] Materialize success and failure rollouts with true outcome labels before
  authorizing any R4 policy training.
- [ ] After the smoke-only amendment passes and a formal-training amendment is
  accepted, compare terminal-outcome weighting against outcome-free CRAVE
  weighting and ordinary pi0.5 fine-tuning.
- [ ] Do not make a Q-value, advantage, world-critic, or model-predictive-control
  claim without action-diverse rollout consequences. Expert demonstrations and
  action-shuffled latent probes are insufficient.

## Current stop rules

- P1 failure closes predictive-adapter replication.
- P2 failure retains the current negative-integration paper and archives the
  predictive-adapter branch as another bounded interface result.
- R4 replication remains blocked on the seed-1000 fixed-checkpoint closed-loop
  gate. CRAVE progress alone must never be reported as reward, action advantage,
  or control value.
- AHEAD-style interception and world-critic RL require new benchmarks or reward
  data and remain outside this graph.
- LeWM/DINO visual replacement, from-PaliGemma initialization, A2/A3 variants,
  and closed MT3--MT6 tasks must not be scheduled from this TODO.
