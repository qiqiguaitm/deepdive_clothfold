# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-13 09:37 UTC

This file contains only unfinished training/evaluation evidence and current
scientific gates. Completed evidence, rejected protocols, and superseded
execution history are in `PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`, Sections
41--50.

The resource-aware scheduler is the sole execution owner. A checkbox records
scientific completion; it does not authorize manual launch, stop, restart,
reprioritization, or replacement. Mutable execution state is authoritative only
in `logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`. Per-task canonical attempts override stale
heartbeat rows. Unmet publication gates below record claim eligibility; they do
not create scheduler tasks or authorize reopening a closed experiment plan.

## 1. Current evidence boundary

The paper asks:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

Operationally, fixed-checkpoint `content use` means that replacing the
condition changes paired outcomes, while `method utility` means that an
independently matched training package improves success. Neither estimand
implies the other, and a claim must name which one it addresses.

The closed evidence establishes the following boundaries:

- The released LaWAM route is endpoint-aligned. Historical raw milestones are
  usually multi-chunk targets without time-to-go. This is timing evidence, not
  control utility.
- TG2 and TG2R were both rejected before evaluation for training-comparison
  integrity failures. They provide no policy or target-horizon comparison.
- TG1B did not detect local-WM-specific cadence sensitivity: difference-in-
  differences +1.42 percentage points, hierarchical 95% CI [-3.00, +5.92].
  This rejects the audited cadence gate, not LMWM as a general method.
- TG1A does establish fixed-checkpoint content use. Normal success is 94.00%
  versus 40.33% under the prespecified within-task, different-episode shuffled
  intervention: +53.67 points, hierarchical 95% CI [+36.08, +68.58],
  Holm-adjusted exact McNemar `p=6.75e-180`. All six task effects and all four
  evaluation-seed effects are positive. Null and persistence controls also pass.

TG1A therefore shows that high performance at this released checkpoint depends
strongly on its episode-matched predicted condition. The different-episode
shuffle can also disrupt scene coherence or create an off-distribution
condition, so TG1A does not by itself isolate endpoint semantics from every
form of matched visual information. It also does not attribute the dependence
among LaWAM policy pretraining, downstream auxiliary shaping, and inference-time
conditioning. TG4 is the only remaining claim-bearing experiment and performs
that source decomposition under one fresh matched protocol.

## 2. Active TG4 source decomposition

The frozen manifest specifies six arms (`clean_base`, `future_off`,
`auxiliary_only`, `conditioning_only`, `parameter_matched_null`, and `full`) at
training seeds 1100--1102, for 18 total 20,000-step cells. All arms use four
GPUs, global batch 128, exact final-checkpoint selection, and matched rank data
orders within seed. The released TG1A checkpoint and the official pi0.5 A0 score
are excluded from within-architecture causal contrasts.

- [ ] **TG4-T01--T18 [ACTIVE; 8/18 COMPLETE]** At the 12:00 UTC canonical
  snapshot, eight unfinished training cells are Running and two are Queueing on
  North; every cell is completed, running, or submitted, with no undispatched
  training cell. All three `auxiliary_only`
  seeds completed all 20,000
  steps and persisted 7.17-GB final models plus 9.13-GB optimizer states. Their
  platform shells were reported Failed only after the training child returned,
  because the shared runner changed line offsets while the long-lived shell was
  blocked in training. A task-scoped recovery verifier checked the frozen
  config, initialization route, four-rank data order, final step, model and
  optimizer sizes and hashes, and exact post-training error before admitting
  the first two artifacts. All three `clean_base` seeds and `auxiliary_only`
  seed 1102 then completed with the same post-training shell error; independent
  North-side audits verified their final models, optimizer states, exact step
  20,000, frozen configs, initialization identities, and rank orders before
  admitting them. All six `clean_base` and `auxiliary_only` artifacts are now
  materialized locally. The two East-produced `auxiliary_only` artifacts
  satisfy materialization directly through the shared local checkpoint path;
  unlike North-produced artifacts, they do not require a North transfer marker.
  No general Failed-terminal exemption was introduced. Per-cell background
  watchers now cover the remaining old-runner North jobs, including `full`
  seed 1102. For the two old-runner East `parameter_matched_null` jobs, a
  zero-GPU watcher waits for the exact terminal log and complete checkpoint,
  then a one-GPU East audit reads the root-owned sidecars, applies the same
  strict verifier, and exposes them read-only to the joint local integrity
  gate. Neither path can admit a task until its own complete audit marker
  exists. The East readiness watcher also carries an explicit terminal-evidence
  latch: timeout with a checkpoint and log but without every exact completion
  token is rejected and cannot publish a marker; positive and negative shell
  regressions cover this boundary. A submission dry-run resolves the audit YAML
  to the intended
  `Robot-East-H20`, `cn-shanghai-e`, one-H20 shape and shared East mount; this
  root-access requirement intentionally overrides the generic local-first
  one-GPU preference. An end-to-end replay on a completed East legacy cell
  also passed both layers: the zero-GPU watcher recognized the exact terminal
  evidence, and the root audit validated the 20,000-step model, optimizer,
  four-rank order, config/route, and precise post-training error before
  atomically publishing `complete=true`. The
  platform-reported `clean_base` seed-1102 failure is therefore already closed:
  its exact step-20,000 recovery audit completed at 02:56 UTC and its verified
  checkpoint was materialized locally at 04:23 UTC; it must not be resubmitted.

  The four immediate East failures (`conditioning_only` seeds 1101/1102 and
  `future_off` seeds 1100/1101) were startup refusals caused by stale partial
  roots from failed gf1 attempts. Those exact roots were moved to a dated
  quarantine without touching active runs, and the four cells were rearmed;
  three additional stale North roots from the original DDP
  failures blocked the first repaired `conditioning_only` retries; those exact
  roots were also quarantined. All three conditioning cells were then submitted
  in parallel on the primary North identity and are now training near steps
  12.5k/10.5k/10.3k. `full` seeds 1100 and 1101 reached exact step 20,000;
  their per-cell recovery watchers verified the frozen configuration,
  initialization, rank orders, final model, optimizer state, and exact
  post-training shell error before admitting them, and both artifacts are now
  materialized locally. `future_off` seeds 1100/1101 are healthy near
  16.5k/16.3k, parameter-matched-null seeds 1101/1102 near 17.3k/17.4k, and
  `full` seed 1102 near 14.4k. A fresh exact-token health scan of all eight
  active logs found no NaN/Inf, OOM, CUDA, NCCL, dataloader, or traceback
  failure; observed throughput remains stable at 1.93--2.26 seconds per step.
  The temporary gf1 processes for `future_off`
  seed 1102 and parameter-matched-null seed 1100 were independently confirmed
  dead after reaching about 12.2k and 11.6k. Because the frozen recipe writes
  no admissible intermediate checkpoint, neither partial run can be resumed or
  counted; both exact cells were resubmitted to North and are Queueing under the
  primary and backup GPU limits. Their gf1 candidates remain exhausted so the
  lost long runs cannot repeat. Their two 8.5-MB local non-checkpoint roots and
  one truncated transfer temporary were moved intact to
  `logs/resource_scheduler_local/tg4_failed_partial_quarantine_20260813T071500Z`;
  the active checkpoint root now has zero stale matches for either run ID, so
  the eventual North materializers and 18-cell uniqueness check cannot consume
  those partials. Every future TG4 launch executes an immutable snapshot of the
  frozen runner. The two North queue-sink retries now carry an opt-in,
  capacity-reserved escape to an immediately free East/gf1 candidate: the
  scheduler first plans against a copied live snapshot, respects per-resource
  failure exhaustion, and stops the queued job with its original identity
  before redispatch. When gf1 became physically free again at 10:54 UTC, both
  old partial roots were already quarantined and the formal output paths were
  empty. A new recovery revision reopened exactly these two exhausted gf1
  candidates. The primary-identity `future_off` seed-1102 North queue copy was
  stopped before a clean gf1 dispatch. That first recovery passed model loading
  and step zero but received `SIGTERM` at step 201 when the scheduler tmux was
  restarted. Its output root contained no model or optimizer checkpoint and was
  renamed in place to a hidden dated quarantine. A second clean recovery was
  launched at 11:12 UTC and reached about step 279 at 1.95 seconds/step. During
  the same window, a non-claim-bearing evaluator preflight intended for free
  GPU 4 was found to remap its inner model server to physical GPU 0, and an
  unrelated WorldArena fleet began taking all eight gf1 GPUs at 11:23 UTC. The
  training DDP group then disappeared without an explicit terminal record. The
  available timing cannot uniquely attribute the termination to either
  concurrent event, so no narrower causal claim is made. The second root again
  contained no model or optimizer checkpoint and was
  renamed in place to a distinct hidden quarantine. No preflight will run on
  gf1 concurrently with training. The evaluator now passes host-visible device
  indices directly to its model servers, and this mapping is frozen and tested
  before any formal panel. After the external fleet briefly drained, a `v6`
  recovery started on physical GPUs 0--3 while the repaired preflight used only
  GPU 7. Both processes disappeared within 40 seconds without script terminal
  records, no training output root was created, and the external fleet then
  refilled all eight GPUs. This independently confirms that gf1 is currently an
  externally reclaimed rather than stable long-job resource. Its temporary
  submission marker is disabled. The scheduler therefore submitted a fresh
  primary-North queue copy (`t-20260813193650-njfhb`) as the persistent sink.
  The
  backup-identity
  `parameter_matched_null` seed-1100
  queue copy could not be stopped by the available identities (`AccessDenied`),
  so no duplicate gf1 run was launched and the North attempt remains tracked.
  Do not inspect partial
  outcomes to alter the protocol. The East migration candidates now require at
  least five free GPUs before taking four, so at most one queued training cell
  can migrate when the two active legacy East cells terminate. This preserves
  capacity for their one-GPU strict recovery audits instead of delaying
  checkpoint admission behind another full training wave. At this snapshot,
  East is 8/8, North primary is 20 active plus 4 queued under its 25-GPU
  limit, North backup is 4 active plus 4 queued under its 8-GPU limit, and
  gf1 is 8/8 under the unrelated workload. Thus every safely admissible
  four-GPU training slot is used or queued, while the idle local two-GPU host cannot
  execute a frozen four-GPU training cell and formal evaluation remains gated.
- [ ] **TG4-I1 [BLOCKED by T01--T18]** Eighteen conditional materializers and
  the joint verifier are implemented. Reject the complete matrix before
  evaluation unless all final checkpoints, optimizer states, initialization
  trees, exact per-rank data orders, dataset statistics, and non-arm configs
  pass. North materialization now uses two bounded seed-sharded transfer slots
  instead of one global serial lock; each run still uses an immutable sync
  script, remote/local SHA256 verification, and an atomic destination move.
  The joint verifier now prefers a local zero-GPU execution path and retains
  East 1-GPU only as a fallback, so platform deployment cannot delay the gate
  after the final materializer completes.
  Rank-order hashes must match across arms within seed and differ across seeds.
  A non-gating 8-of-18 pre-integrity audit found that every currently accepted
  and materialized cell has a complete final model and optimizer at exact step
  20,000; all currently decidable parameter-tree, trainable-tree,
  optimizer-tree, normalization, initialization, route, within-seed order,
  across-seed order, and non-arm configuration checks pass. The report remains
  explicitly `complete=false` and `claim_bearing=false` with ten missing run
  IDs, so it cannot satisfy I1. This pre-audit also exposed a verifier-only
  false mismatch: serialized `log_dir` necessarily contains each timestamped
  run path. The final verifier now excludes `log_dir` alongside `run_id`,
  `output_dir`, and `seed` while still rejecting drift in true non-arm fields;
  positive varying-path and negative action-horizon regressions cover the
  distinction. No training artifact or scientific comparison changed.
  A pre-completion audit also corrected a verifier-only distinction for the
  matched-parameter route ablations. Their frozen serialized configs
  deliberately retain `future_prediction=true` and
  `enable_loss_distill=true`; the independently audited
  `LAWAM_AUXILIARY_OFF` route makes `conditioning_only` auxiliary-free, while
  `LAWAM_FUTURE_OFF` makes `parameter_matched_null` future/auxiliary-null.
  The manifest's effective `auxiliary_loss=false` semantics are unchanged; no
  training input or comparison definition changed. Live loss decomposition
  independently confirms the routing: all active `conditioning_only` and
  `parameter_matched_null` cells report zero perceptual/distillation loss,
  while the active `full` cell reports nonzero values for both. Recent loss
  windows for all eight active cells are finite, with no NaN, Inf, OOM,
  CUDA/NCCL error, traceback, or dataloader-worker failure in their current
  logs.
- [ ] **TG4-E1 [IMPLEMENTED; BLOCKED by I1]** Twenty-one scheduler tasks are
  registered under the independently frozen evaluation manifest: normal for
  all 18 arm/seed cells and within-task shuffled content for all three `full`
  checkpoints. Each task requires exactly 24 fixed-scene summaries. Normal
  panels can use local 2-GPU, East 4-GPU, North 4-GPU, or either of two
  disjoint gf1 4-GPU slices; shuffled panels remain on shared-storage resources
  so the captured condition cannot cross an unverified storage boundary. The gf1
  slices carry explicit `CUDA_VISIBLE_DEVICES` mappings and separate port
  namespaces so two evaluations can execute concurrently after training frees
  that host. `full` shuffled additionally depends on
  the matching normal capture. Partial rollouts cannot change the panel or
  support a claim. Before any TG4 evaluation started, a runtime preflight found
  that the local and East launchers referenced untracked `lmwm` renderer paths;
  both now point to the existing, previously exercised `lmwam` helpers. This
  path-only repair changed no checkpoint, task, scene, seed, episode, or
  intervention definition. A non-claim-bearing local end-to-end preflight then
  passed at 09:57 UTC using an isolated one-episode subset of the frozen scene
  manifest: the accepted `full` seed-1100 checkpoint loaded onto GPU, RoboTwin
  completed the fixed `beat_block_hammer` scene, the policy served three model
  queries for three simulator observations, and future-feature capture was
  nonempty. Its result root and marker are disjoint from all 21 formal panels,
  and it cannot satisfy any claim-bearing monitor gate. A second isolated
  preflight passed at 10:10 UTC for the full shuffled-content route: normal
  mode first captured both frozen target and no-self source scenes, then
  shuffled mode consumed the source-scene feature selected by the frozen
  mapping and completed the target scene with three model queries and three
  simulator observations. Missing, self-matched, or shape-incompatible source
  features remain hard failures. This preflight is also non-claim-bearing and
  cannot satisfy a formal panel or monitor gate. Before any formal evaluation
  started, a gf1-only preflight also exposed and repaired a physical-device
  routing bug: inner model servers now receive the host-visible GPU index
  directly instead of resetting every slice to physical GPU 0. Its repaired
  retry was externally reclaimed before completion, so gf1 remains disabled;
  this optional preflight does not gate the already-passed local route or any
  formal panel. The execution bundle was also made strictly resumable without
  changing the frozen panel: an existing run is accepted only when checkpoint,
  arm, training seed, condition, evaluation seed, six-task order, and 50-episode
  metadata match exactly; the established evaluator then schedules only
  incomplete tasks. Every resource candidate permits one bounded retry. A
  verified North result now quarantines any shared-storage partial before
  atomic materialization instead of silently replacing it, and `full` normal
  resumes reject summaries without their feature-capture tree. The frozen runtime
  bundle was further extended with a hash-gated North staging task. It reuses
  exact North-native checkpoints by verified hard link, uploads only missing
  final models, and atomically materializes every North result. Four
  East-native cells also have zero-GPU asynchronous prefetch tasks: each waits
  for its strict training acceptance, verifies the exact accepted SHA256 and
  byte count, and shares a transfer lock with formal staging. The formal stage
  independently rechecks any prefetched model against the final 18-cell
  integrity manifest and can upload it itself if prefetch fails, so this
  optimization cannot weaken or block the gate. For `full`, staging
  also re-verifies and materializes the complete feature capture before the
  matching shuffled panel can start. Analysis now depends on those materialized
  normal panels rather than platform terminal state alone. The symlink healer,
  renderer helpers, North wrapper, staging script, and materializer are explicit
  frozen hash dependencies. A final pre-evaluation audit at 10:53 UTC also
  closed the result-schema boundary: the TG4-specific frozen verifier now
  rejects a wrong or missing evaluation seed, inconsistent `n_episodes`,
  malformed scene seeds, and non-boolean success values before a panel can
  complete. The older generic fixed-seed verifier remains byte-identical so
  previously frozen P2 evidence is not rewritten. The formal runner invokes
  the strict verifier directly, and North staging copies and hash-checks both
  verifiers. A focused 239-test scheduler, watcher, resume, bundle,
  training-integrity, evaluation, analysis, and finalizer suite passes,
  including compatibility against an existing complete 24-cell result tree.
  Fresh live verification of both frozen TG4 source and evaluation manifests
  also passed at 10:53 UTC.
  The independent hourly audit now freezes all 79
  claim-bearing execution gates, including the North staging task and all 18
  normal-result materializers, so it cannot declare completion at a platform
  eval terminal before the artifacts are verified locally. Its 12:00 UTC poll
  consumed the current TODO SHA256 and a 19-second-old scheduler snapshot with
  no errors, found all 79 nodes registered, and scheduled the next poll for
  13:00 UTC. A pre-integrity
  transfer check completed the first accepted East model prefetch at 09:01 UTC,
  including TOS upload/download, remote SHA256 verification, atomic commit, and
  temporary cleanup. The second accepted model then acquired the same lock and
  completed the identical chain at 09:15 UTC; both scheduler artifact checks
  are complete. The corresponding resource audit confirms that all 21
  panels remain unattempted, the frozen
  evaluation bundle still passes its hash verifier, and North has the staged
  Python 3.12 runtime, 332-GB RoboTwin dataset tree, required weights, and
  sufficient space for the four expected East-native checkpoint uploads. A
  no-launch capacity replay using the scheduler's actual candidate ordering,
  reservation logic, 25-GPU primary limit, 8-GPU backup limit, and current gf1
  disablement confirms that, after I1 and the gated North stage complete, 11 of
  18 normal panels can start within two 15-second dispatch polls: six on North
  primary, two on North backup, two on East, and one on the local two-GPU host.
  The remaining seven form later waves; the replay did not create a stage
  marker or bypass the integrity dependency.
- [ ] **TG4-A1 [IMPLEMENTED; BLOCKED by E1]** The scheduler-registered analyzer
  depends on all 21 evaluations and implements the seven frozen contrasts,
  training-seed/task/evaluation-seed/paired-episode hierarchical bootstrap,
  Holm correction, and per-training-seed/task tolerance gate. It writes the
  canonical result and immutable decision marker before manuscript claims can
  change. The hourly completion audit now rejects a partial or inconsistent
  terminal analysis unless `complete=true`, the ordered Holm family and exact
  seven-comparison set are present, every `accepted` value is boolean, and all
  seven marker verdicts agree with the JSON report. A scheduler-owned zero-GPU
  analysis attempt is now admissible only after its local launcher reports
  `FINISHED rc=0`; a marker left by a nonzero or interrupted process cannot
  complete A1, and one bounded retry is available. Local and SSH terminal
  handling regressions cover both successful and failed marker-bearing paths.
  A scheduler-owned zero-GPU
  finalizer now depends on that accepted analysis artifact. It atomically
  writes a seven-row Markdown result table, changes only the four TG4 execution
  gates to complete, records accepted and rejected contrasts without launching
  follow-up work, and emits hashes for the report, decision marker, summary, and
  resulting TODO. The hourly monitor includes this finalizer as its 79th frozen
  completion node, removing the previous manual TODO-sync gap. Its publication
  wrapper commits only the TODO, canonical JSON, and tracked TG4 summary with
  `git commit --only`, preserves unrelated staged and working-tree changes,
  pushes `main`, verifies `HEAD == origin/main`, and only then exposes the
  completion marker. The independent hourly monitor now also requires the
  finalizer task to be complete, parses an exact marker schema, independently
  recomputes the report, analysis-marker, summary, and TODO SHA256 values, and
  requires a full 40-character publication commit. Missing markers, incomplete
  tasks, unexpected fields, and post-finalization file mutation cannot be
  reported as overall TODO completion.

### TG4 claim gates

The seven prespecified Holm-family contrasts are:

1. pretraining: `future_off - clean_base`;
2. auxiliary shaping: `auxiliary_only - parameter_matched_null`;
3. conditioning without auxiliary loss:
   `conditioning_only - parameter_matched_null`;
4. full total effect: `full - parameter_matched_null`;
5. full versus historical future-off: `full - future_off`;
6. route interaction:
   `full - auxiliary_only - conditioning_only + parameter_matched_null`;
7. content use: `full_normal - full_shuffled` at each fixed checkpoint.

Every claimed positive contrast requires a hierarchical 95% CI lower bound
above zero, Holm-adjusted paired `p<0.05`, and no regression worse than five
percentage points on any training-seed/task cell. Report every task effect and
every negative effect. A macro mean cannot hide a regression. A source may be
called causal only for its matched contrast; neither representation quality nor
public-system performance identifies a component's control contribution.

## 3. PredictiveActionAdapter core-method publication gates

PredictiveActionAdapter remains an admissible architectural contribution: P0
establishes that future-target predictions are detectably sensitive to expert
actions, exact inherited-parameter isolation, and zero-output initialization;
it does not establish semantically useful future dynamics because the measured
cosine contrast is small and lacks persistence/current-grid baselines. The
completed efficiency audit establishes 0.50% additional parameters and low
measured runtime overhead. To
present it instead as a primary *control-improving method*, all three gates
below must pass. They are separate claims; passing one cannot substitute for
another.

- [ ] **PA-U1 [UNMET; CLOSED UNDER THE CURRENT PLAN] Matched-seed utility.**
  Compare candidate and A0 trained independently from the same initialization,
  data order, optimization recipe, checkpoint step, and evaluation scenes at
  each training seed. The hierarchical paired 95% CI lower endpoint for the
  equally weighted candidate-minus-A0 effect must exceed zero, with training
  seed as the highest resampling unit. P3 completed this test at seeds
  1000--1002: effects were +13.42, -5.50, and -2.08 points; mean +1.94 points,
  95% CI [-5.78, +12.75]. The gate is not met.
- [ ] **PA-S1 [UNMET; CLOSED UNDER THE CURRENT PLAN] Task-level tolerance.** A claim of
  safe or broadly consistent improvement additionally requires no candidate-
  minus-A0 regression worse than five percentage points in any prespecified
  training-seed/task cell. P3 fails this gate: seed 1001 regresses by 13 points
  on Ranking-size and 9 points on Stack-3, and seed 1002 regresses by 5.5 points
  on Ranking-RGB. These task-level failures must remain visible beside the
  macro result.
- [ ] **PA-C1 [UNMET; CLOSED UNDER THE CURRENT PLAN] Content-specific
  causality.** At fixed checkpoints and exactly paired scenes, normal inference
  must outperform the prespecified shuffled-action intervention with a
  hierarchical 95% CI lower endpoint above zero and Holm-adjusted paired
  `p<0.05`. P4 completed the three-seed panel: normal minus shuffled is +0.53
  points, 95% CI [-2.14, +3.08], Holm-adjusted `p=0.534`. The gate is not met.
  The separate normal-minus-zero-route and normal-minus-masked gates are also
  unresolved, so route necessity and action-conditioning use cannot be used as
  substitutes for correct-content evidence.

**Current claim boundary:** the manuscript may present PredictiveActionAdapter
as a new, zero-initialized predictive-control interface that preserves the base
function at initialization, and report its detectable action sensitivity and
efficiency properties. It must not claim independently replicated utility,
broad task-safe improvement, useful learned dynamics, or causal use of the
predicted content. P6 and P7 remain closed, and this section authorizes no new
training or evaluation. Reopening any gate would require a new
result-independent frozen protocol and explicit operator authorization;
existing P3/P4 rollouts may not be selectively extended or reinterpreted.

## 4. Stop and reporting rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, TG2/TG2R, TG3,
  TG5, outcome weighting, oracle-transition, or failed helper jobs to search for
  a positive result.
- Do not evaluate either rejected TG2 matrix or substitute one of its
  checkpoints into another protocol.
- Partial rollouts, smoke tests, training losses, representation metrics,
  checkpoint existence, or unmatched seeds cannot pass a utility gate.
- The four evidence questions are claim-specific, not a mandatory linear
  ladder: training-package utility and deployed content use need not imply one
  another. Representation prediction does not establish control utility.
  Cadence sensitivity does not establish correct-content use. A public system
  score does not identify its causal component.
- Do not tune target horizon, task groups, training seeds, checkpoint step,
  intervention mapping, retry recipe, arm set, or loss weight against outcomes.
- Task_N remains outside this paper plan by operator instruction.

## 5. Canonical sources

- Active scheduler summary: `logs/resource_scheduler_snapshot.md`
- Canonical mutable state: `logs/resource_scheduler_snapshot.json` and
  `logs/resource_scheduler_state.json`
- TG4 frozen protocol:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_source_decomposition_v1.json`
- TG4 training verifier:
  `lmvla/lmwm/scripts/verify_temporal_grounding_tg4_training.py`
- TG4 frozen evaluation protocol:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json`
- TG4 evaluation runner and analysis:
  `train_scripts/kai/eval/run_temporal_grounding_tg4_eval.sh` and
  `lmvla/lmwm/scripts/analyze_temporal_grounding_tg4.py`
- Completed TG1A result:
  `lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg1a.json`
- Completed predictive-adapter matched-seed result:
  `lmvla/lmwm/docs/pi05_predictive_adapter_p3_matched_seed_gate.json`
- Completed predictive-adapter intervention result:
  `lmvla/lmwm/docs/pi05_predictive_adapter_p4_intervention_gate.json`
- Completed evidence and protocol history:
  `lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`
