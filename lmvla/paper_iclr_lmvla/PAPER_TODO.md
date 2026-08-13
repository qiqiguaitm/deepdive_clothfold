# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-13 05:35 UTC

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

TG1A therefore shows that the particular predicted endpoint content can improve
closed-loop control for this released checkpoint. It does not attribute the
gain among LaWAM policy pretraining, downstream auxiliary shaping, and
inference-time conditioning. TG4 is the only remaining claim-bearing experiment
and performs that source decomposition under one fresh matched protocol.

## 2. Active TG4 source decomposition

The frozen manifest specifies six arms (`clean_base`, `future_off`,
`auxiliary_only`, `conditioning_only`, `parameter_matched_null`, and `full`) at
training seeds 1100--1102, for 18 total 20,000-step cells. All arms use four
GPUs, global batch 128, exact final-checkpoint selection, and matched rank data
orders within seed. The released TG1A checkpoint and the official pi0.5 A0 score
are excluded from within-architecture causal contrasts.

- [ ] **TG4-T01--T18 [ACTIVE; 6/18 COMPLETE]** At the 05:18 UTC canonical
  snapshot, ten cells are executing and two are platform-Queueing; every cell
  is completed
  or submitted, with no undispatched training cell. All three `auxiliary_only`
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
  admitting them.
  No general Failed-terminal exemption was introduced. Per-cell background
  watchers now cover the remaining old-runner North jobs and cannot admit a
  task until its own complete audit marker exists.

  The four immediate East failures (`conditioning_only` seeds 1101/1102 and
  `future_off` seeds 1100/1101) were startup refusals caused by stale partial
  roots from failed gf1 attempts. Those exact roots were moved to a dated
  quarantine without touching active runs, and the four cells were rearmed;
  they were rearmed. Three additional stale North roots from the original DDP
  failures blocked the first repaired `conditioning_only` retries; those exact
  roots were also quarantined. All three conditioning cells were then submitted
  in parallel on the primary North identity: seed 1100 is training at about
  1.94 s/step, while seeds 1101/1102 remain platform-Queueing for physical
  cards. The two
  `future_off` retries are healthy near steps 3.8k/3.7k, and every future TG4
  launch executes an immutable snapshot of the frozen runner. Do not inspect
  partial outcomes to alter the protocol.
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
- [ ] **TG4-E1 [IMPLEMENTED; BLOCKED by I1]** Twenty-one scheduler tasks are
  registered under the independently frozen evaluation manifest: normal for
  all 18 arm/seed cells and within-task shuffled content for all three `full`
  checkpoints. Each task requires exactly 24 fixed-scene summaries and can use
  local 2-GPU or East 4-GPU execution. `full` shuffled additionally depends on
  the matching normal capture. Partial rollouts cannot change the panel or
  support a claim. Before any TG4 evaluation started, a runtime preflight found
  that the local and East launchers referenced untracked `lmwm` renderer paths;
  both now point to the existing, previously exercised `lmwam` helpers. This
  path-only repair changed no checkpoint, task, scene, seed, episode, or
  intervention definition. The two renderer helpers are now explicit frozen
  hash and readiness dependencies, the repaired bundle verifier passes, and a
  local `SapienRenderer` construction completed successfully.
- [ ] **TG4-A1 [IMPLEMENTED; BLOCKED by E1]** The scheduler-registered analyzer
  depends on all 21 evaluations and implements the seven frozen contrasts,
  training-seed/task/evaluation-seed/paired-episode hierarchical bootstrap,
  Holm correction, and per-training-seed/task safety gate. It writes the
  canonical result and immutable decision marker before manuscript claims can
  change.

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
establishes action-conditioned latent prediction, exact inherited-parameter
isolation, and zero-output initialization, and the completed efficiency audit
establishes 0.50% additional parameters and low measured runtime overhead. To
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
- [ ] **PA-S1 [UNMET; CLOSED UNDER THE CURRENT PLAN] Task safety.** A claim of
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
as a new, policy-preserving predictive-control interface and report its latent
prediction and efficiency properties. It must not claim independently
replicated utility, task-safe improvement, or causal use of the predicted
content. P6 and P7 remain closed, and this section authorizes no new training or
evaluation. Reopening any gate would require a new result-independent frozen
protocol and explicit operator authorization; existing P3/P4 rollouts may not
be selectively extended or reinterpreted.

## 4. Stop and reporting rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, TG2/TG2R, TG3,
  TG5, outcome weighting, oracle-transition, or failed helper jobs to search for
  a positive result.
- Do not evaluate either rejected TG2 matrix or substitute one of its
  checkpoints into another protocol.
- Partial rollouts, smoke tests, training losses, representation metrics,
  checkpoint existence, or unmatched seeds cannot pass a utility gate.
- Representation prediction does not establish control utility. Cadence
  sensitivity does not establish correct-content use. A public system score
  does not identify its causal component.
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
