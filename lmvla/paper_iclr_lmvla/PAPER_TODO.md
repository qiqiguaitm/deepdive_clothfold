# MINT-VLA Remaining Evidence TODO

Updated: 2026-08-02 01:15 UTC

This file tracks only evidence that is still required for the paper. Completed
jobs, platform incidents, and per-attempt identifiers are intentionally omitted.
Live execution state is maintained in `logs/resource_scheduler_snapshot.md` and
`logs/resource_scheduler_state.json`.

Completed evidence and execution history are preserved in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`.

## Locked protocol

- Primary VLA and benchmark: pi0.5 on RoboTwin 2.0.
- Confirmatory recipe: absolute actions, mean/std normalization, batch 16,
  50k updates, and identical initialization, data, and evaluator across arms.
- Primary arms: A0 no hint, A2-Abs offline absolute, and A3 MINT-VLA with a
  current-encoder predictor.
- Do not replicate the legacy joint-delta/quantile/batch-64/20k matrix.
- Matched A2/A3 and additional training seeds may run speculatively in parallel
  to reduce wall-clock time. Their results remain inadmissible for the paper
  until corrected A0 passes the 70% operational gate and provenance audit.

## Frozen evidence

These items are complete and are no longer TODOs.

- **Data gate:** `robotwin_milestone_all6_confirmatory_v1` covers 1,200
  episodes across all six tasks and passed coverage, monotonic-target, source
  hash, and READY audits.
- **Content interventions:** A2 and A3 correct/current/zero/feature-permuted/
  cross-task/within-task-instance controls are complete on final 518-scene
  intersections. Correct future is not detectably better than any key control;
  all pooled Holm-adjusted p-values are 1. The paper must not claim
  inference-time future-content causality from these checkpoints.
- **Inference cost:** A0/A2/A3 core latency is 57.19/58.35/57.99 ms and
  WebSocket latency is 107.20/108.91/107.01 ms, corresponding to
  9.33/9.18/9.34 requests/s. No material A3 inference penalty is detected.
- **Training memory:** A0/A2-Abs/A3 peak at 67,247/67,247/67,265 MiB under
  the matched four-A100 protocol. A2 exactly matches A0 and A3 adds only
  18 MiB (0.027%), so E2 is complete.
- **Evaluator calibration:** the public pi0.5 checkpoint reaches 78.42% on the
  frozen six-task bridge. The legacy A0 reaches 35.50% through that bridge and
  31.83% over 600 episodes under the native RoboTwin evaluator, with the same
  task-level failure pattern. Its deficit is therefore primarily a weak-recipe
  checkpoint problem rather than a bridge artifact, and it is not the
  confirmatory baseline.
- **LaWAM diagnostic:** Future-off averages 90.83% over three training seeds,
  versus 88.78% for combo over three seeds, and 89.21% for absolute, 89.04%
  for gradient isolation, 87.50% for residual, and 86.67% for local-WM over
  two seeds each.
  Future-off inherits WM-aware initialization and is not a pure-VLA baseline;
  neither gradient isolation nor any other completed downstream WM-objective
  replication shows a gain over it.
- **Task-regime audit:** mean demonstration duration has an exploratory
  association with absolute/combo recovery over local-WM (`rho=0.829`, six
  post-hoc tasks), but long tasks split sharply and duration is not a sufficient
  explanation. This offline analysis is complete and archived, not a TODO.
- **Held-out future prediction:** two task-stratified folds covering 480
  held-out episodes reach latent cosine 0.8134 versus 0.7479 for persistence
  (lift +0.0655), with milestone retrieval top-1/top-5 of 46.9%/80.8%.
  The predictor therefore learns nontrivial future information; poor control
  cannot be attributed solely to an empty predictor.
- **State-dependent retrieval upper bound:** on 1,161 paired episodes, a
  task-scoped demonstration retrieval milestone changes pooled success by only
  +0.34 pp (89.58% versus 89.23%, exact McNemar p=0.738; Holm p=1.0).
  Better milestone content does not repair the current one-token interface.
- **Six-task hint diagnosis:** all nine strict method-condition comparisons are
  complete on 1,200 paired scenes each (216/216 cells, 9/9 markers). For
  absolute, residual, and combo, correct hints do not beat zero, cross-task,
  or within-task controls at the pooled level after Holm correction. The only
  corrected task-level difference is adverse: combo reaches 72.5% on handover
  with the correct hint versus 82.5% with a same-task foreign-instance hint
  (10.0 pp lower, Holm p=0.0337). This closes L2 and strengthens the conclusion
  that the current one-token interface has no demonstrated future-content
  utility.
- **Privileged spatial-interface gate:** the parameter-matched 1,000-update
  S0-N/S0-C/S0-P probe is complete on 320 frozen held-out samples. On Stack-3,
  privileged endpoint L2 is 0.3378 versus 0.3274 for no-goal and 0.3580 for
  current-image patches, so it does not beat both controls. On Hammer it is
  0.3080 versus 0.2355/0.2403. Episode-level paired bootstrap confirms the
  Hammer regression versus no-goal (+0.0725, 95% CI [+0.0391, +0.1110]) and
  current (+0.0677, [+0.0332, +0.1054]). T3a therefore fails its frozen gate;
  T3b predictor expansion is closed and no pass marker is published.

Canonical result files:

- `logs/pi05_a2_causal_with_instance.json`
- `logs/pi05_a3_causal_final.json`
- `logs/efficiency/pi05_a0_a2_a3_latency.json`
- `logs/efficiency/pi05_train_memory_a0.json`
- `logs/efficiency/pi05_train_memory_a2_abs.json`
- `logs/efficiency/pi05_train_memory_a3_live.json`
- `logs/eval_reports/robotwin_all6_v2_training_seed_matrix.json`
- `logs/eval_reports/robotwin_lmwm_heldout_twofold.json`
- `logs/l2_six_task_intervention_analysis.json`
- `logs/spatial_s0/s0_offline_verdict.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_confirmatory_eval_protocol.json`
- `lmvla/lmwm/docs/PROGRESS_pi05_vla_baseline_2026-08-01.md`

## Remaining priority queue

| Priority | ID | Remaining evidence | Current gate |
|---|---|---|---|
| P0 | T0 | Exact A0 seed-1000 training and frozen six-task evaluation | Training on gf1 4xA100; checkpoint sync, evaluation, and gate automated |
| P1 | T1 | Corrected A2-Abs and A3 seed-1000 training plus matched A0/A2/A3 evaluation | Seed-1000 training runs speculatively; result acceptance waits for T0 |
| P2 | T2 | A0/A2/A3 seeds 1001 and 1002 with hierarchical uncertainty | Training runs speculatively in parallel; reporting waits for accepted T0/T1 |
| P4 | T4 | Preregistered selector and task-regime panel | Run only if T1/T2 or T3 establish utility; retain the one-token null boundary |
| P5 | T5 | Mature-initialization transfer | Run after T2 or by explicit reviewer-risk decision |
| P6 | T6 | Matched instantiation on a second VLA | Run only if T1 supports MINT-VLA |
| S1 | L1 | Clean VLA versus LaWAM-init/Future-off training comparison | Required only for a claim that LM pretraining benefits a VLA |

## Publication evidence contract

Every confirmatory result entering the main paper must satisfy this reporting
contract. These are reporting requirements, not additional model arms.

- Report all six task cells, the macro average, and all three training-seed
  values. Do not use the macro average to conceal task-level regressions.
- For A2-Abs--A0 and A3--A0, report paired scene-level effect sizes and 95%
  hierarchical bootstrap intervals. State the number of training seeds,
  scenes, and episodes in each caption.
- Keep success-rate evidence separate from representation metrics. Predictor
  cosine and retrieval accuracy establish signal quality, not control utility.
- Treat the strict hint interventions and spatial privileged probe as boundary
  evidence. The former uses 1,200 paired episodes per method-condition
  comparison; the latter is a 1,000-update, two-task offline action-endpoint
  probe, not a closed-loop success result.
- Main-paper figures use a white canvas, final-size 6--8 pt sans-serif type,
  strokes of at least 0.5 pt, sentence-case labels, and colourblind-safe
  colours paired with marker or line-style cues. Captions define every symbol,
  sample size, interval, and whether higher or lower is better.
- Reserve the headline result figure for the accepted A0/A2/A3 matrix. Until
  it is complete, legacy pilot values remain visibly labelled exploratory.

Already-running LaWAM replications may finish as supporting diagnostics, but
they do not gate the pi0.5 paper and must not create new method arms. Their live
status belongs in the scheduler snapshot, not in this file.

## Experimental funnel

1. **Validate the policy baseline (T0).** No method comparison is interpretable
   until exact A0 passes the operational and provenance gates.
2. **Measure the current interface (T1).** Complete the already prepared matched
   A0/A2/A3 comparison. A gain here is an integration or representation effect,
   because one-token content causality is already null.
3. **Spatial-interface decision (T3, closed).** The privileged patch condition
   did not beat both controls on Stack-3 and regressed Hammer with a paired
   confidence interval excluding zero. Predictor expansion is stopped.
4. **Replicate and broaden (T2/T4--T6).** Training-seed replication may run
   speculatively to use idle resources. Task panels, mature initialization, and
   a second VLA still require a positive utility gate before launch.

The implementation, controls, logging contract, and resource escalation rules
for T3 are frozen in
`lmvla/lmwm/docs/PLAN_pi05_spatial_future_interface_2026-08-01.md`.

## Acceptance criteria

### T0: corrected baseline

- [ ] Finish exact A0 seed 1000 at 50k updates.
- [ ] Pass immutable checkpoint, config, normalization, dataset, and
  launch-provenance audit.
- [ ] Complete the frozen 24-cell evaluation and reach macro >=70%.

The exact replacement is running on four A100s with the validated 27,500-
episode official-prompt mirror, `augment_level="none"`, and immutable launch
hashes. It has passed 48.4k/50k updates at approximately 0.77 s/update without
a throughput regression; the remaining training ETA is approximately 0.35
hours. The 20k
non-gating smoke reaches 32/40 on Hammer (80%) and 4/40 on Stack-3 (10%), or
36/80 overall, up from 3/40 at 10k. This establishes a rising learning curve
but also shows that the multi-stage task is still immature. A non-gating full
24-cell evaluation of the 20k checkpoint has produced all 24/24 shard
artifacts; only the final 50k evaluation can close T0. Checkpoint
synchronization and the final frozen
evaluation are automated. The earlier prompt-defective reproduction is invalid
for the confirmatory table; its checkpoint, evaluation, and diagnosis are
retained in the evidence archive.

If T0 fails, repair the baseline before accepting or interpreting any
speculatively trained A2/A3 result.

### T1: first matched utility matrix

- [ ] Train corrected A2-Abs seed 1000 from the same initialization as A0.
- [ ] Train corrected A3 seed 1000 with the frozen six-task pair artifact.
- [ ] Evaluate A0/A2/A3 on identical scene manifests and report every task.

A2-Abs and A3 seed 1000 are running concurrently on Robot-East-H20 at about
1.3 updates/s (roughly 10.4 hours from step 0). A3 has a nonzero weighted LMWM
loss from step 100 onward, so its live-target path is active. Their frozen
evaluations are queued behind the 50k checkpoints. The frozen evaluator
protocol is audited in `manifests/pi05_confirmatory_eval_protocol.json`.
A 1,200-episode scene-seed manifest is now frozen from the independent public
pi0.5 same-bridge run (`pi05_public_samebridge_4seed_v3`, 78.42% macro). The
scheduler injects it into every A0/A2/A3 evaluator and refuses completion
unless all 24 cells pass exact scene-order and manifest-SHA verification.
A2 uses server-side So400m encoding followed by the matched LMWM
predictor/generator to produce an absolute predicted `g_next`; it does not use
a zero fallback, current-feature passthrough, or residual hint. Shared and
North evaluator code, LMWM checkpoint, and So400m weights have matching
SHA-256 hashes.
Runtime preflight found and corrected a configuration error before any final
evaluation launched: A2/A3 candidates had referenced their training configs,
which would invoke offline hint/target lookup. They now use the dedicated
`pi05_robotwin_a2_prefix_official_eval_bj` and
`pi05_robotwin_a3_live_residual_prefix_official_eval` configs on every backend;
their sidecars override mean/std assets on those inference configs. The A0 20k
one-cell preflight completed all 50 frozen Hammer scenes and passed the
partial-manifest verifier. The A2 5k preflight restored the checkpoint, loaded
the checkpoint norm statistics, enabled server-side So400m, loaded the LMWM
predictor, and entered fixed-scene simulation. A2 and A3 one-cell verification
now run concurrently on the two local GPUs; A3 started automatically as soon
as the A0 preflight released GPU 0.

Continue interpreting T2 only if A3 is competitive with A0 and A2-Abs. T3a is
already closed: preserving a 4x4 privileged patch topology did not pass its
two-task action-utility gate, so no predicted spatial expansion is admissible
under the frozen protocol. Because the completed content interventions are
null, any T1 gain must initially be described as an integration or
representation effect, not as evidence that the policy uses correct future
content.

### T2: training-seed replication

- [ ] Train matched seeds 1001 and 1002 for A0/A2/A3.
- [ ] Report seed-level intervals and paired A3-A0 and A2-A0 contrasts.
- [ ] Add another seed only if one outlier controls the interpretation.

All six replication trainings are Running. A0 seed 1002 and A2/A3 seeds
1001--1002 exactly fill the Robot-North-H20 primary 20-GPU limit; A0 seed 1001
runs on gf1 GPUs 4--7 after L2 released them. Together with A0 seed 1000 on
gf1 GPUs 0--3 and A2/A3 seed 1000 on Robot-East-H20, every confirmatory
training arm is active with no platform queueing. All nine frozen evaluations
are queued behind their respective 50k checkpoint and normalization artifacts;
all share `robotwin_pi05_confirmatory_scene_seeds_v1.json` (SHA-256
`08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`).
The four evaluations whose checkpoints live on shared vePFS have validated
Robot-East-H20 candidates in addition to gf1/North/robot-task fallbacks. This
preserves the resource order `gf1 > East > North > robot-task` when training
releases capacity; the East renderer gate already passed a real episode.

### T3: spatial-interface utility gate

#### T3a: privileged condition before predictor training

- [x] Implement a policy-side spatial condition that preserves milestone patch
  topology from the current pi0.5 visual encoder; do not global-average it into
  one prefix token.
- [x] Keep milestone/world-model losses out of the shared policy encoder while
  allowing the action objective to train the adapter, confidence gate, and
  action expert.
- [x] Run a matched two-task smoke on Stack-3 (high predictor lift and stage
  structure) and Hammer (low predictor lift and short horizon).
- [x] Compare privileged milestone patches, current-image patches, and no-goal
  with identical parameters, scene manifests, initialization, and updates.

S0 is complete under the frozen T3 plan. The three
parameter-matched condition arms, 4x4 topology-preserving adapter/gate,
stop-gradient target route, target-frame data path, and deterministic no-goal
fallback are implemented. A task-stratified split freezes 160 train and 40
held-out episodes for each of Stack-3 and Hammer. Twelve targeted model,
data-index, and checkpoint-loading tests pass. The one-step privileged-arm
end-to-end preflight also passes with a finite action loss and gradient norm,
including base-checkpoint loading, target lookup, forward/backward, and
checkpoint export. A task-balanced 32-sample evaluator smoke passes on both
tasks with strict finite JSON, 100% target availability, and nonzero action
sensitivity to shuffled privileged targets. The matched 1,000-update
S0-N/S0-C/S0-P arms and all three frozen 320-sample held-out evaluations finish
successfully from update 999 with identical sample, evaluator, and noise
protocol hashes. Privileged patches fail to improve Stack-3 over both controls
and regress Hammer; the episode-level paired verdict is archived at
`logs/spatial_s0/s0_offline_verdict.json`. The T3a implementation boxes are
complete, but `pi05_spatial_s0_offline.ok` is intentionally absent because the
utility gate failed.

T3a passes only if privileged milestone patches improve the prespecified
Stack-3 endpoint without a material Hammer regression. If it fails, do not
train a larger predictor or launch a six-task spatial matrix.

#### T3b: predicted spatial condition (closed by gate)

T3b was not run because T3a failed. Predicted/retrieved spatial conditions,
generated-condition mixtures, confidence gating, refresh policies, and route
ablations are no longer remaining paper tasks. Reopening them requires a new
preregistered interface hypothesis rather than tuning against this S0 result.

### T4: selector and task scope

- [ ] Compare fixed-horizon, random-future, terminal-frame,
  recurrence-milestone, and oracle-milestone targets.
- [ ] Submit matched evaluations on the already frozen block/bowl stacking,
  reactive/contact, geometry, and relational-transfer panel.

A scoped claim requires replication on both block and bowl stacking and an
advantage over fixed horizon. It must describe the frozen null result as a
failure of the evaluated one-token interface, not erase or contradict it.

### T5: mature initialization

- [ ] Fine-tune matched A0/A2/A3 from one mature initialization with identical
  optimizer-state policy, data mixture, updates, and evaluator.

### T6: second VLA

- [ ] Establish a clean no-milestone baseline on a second VLA.
- [ ] Reuse milestone selection and native-space prediction while matching the
  second VLA's feature width and conditioning interface.
- [ ] Train at least two seeds per arm and report task effects, parameters, and
  inference cost.

Do not claim multi-architecture evidence unless MINT-VLA improves over the
second architecture's matched baseline.

### L1: identify whether LM pretraining benefits the VLA

- [ ] Train a clean VLA/Future-off arm without LaWAM pretraining under a
  matched downstream recipe and evaluation protocol.
- [ ] If this contrast is used for a pretraining-benefit claim, submit a
  budget-matched LaWAM-pretrained/Future-off control and replicate both arms.

The existing Future-off checkpoint inherits LaWAM pretraining and cannot serve
as the clean VLA endpoint. Until L1 is complete, neither “LM benefits VLA” nor
“LM is useless to VLA” is identified by the supporting matrix.

### L2: six-task hint-condition evaluation

- [x] Replace the unmatched episode IDs in all nine method-condition pairs and
  reach 1,200 identical scene IDs per comparison using frozen per-cell
  manifests. The strict matrix is complete at 216/216 cells and 9/9 markers.
  No method has a pooled correct-hint advantage over zero, cross-task, or
  within-task controls after Holm correction. Combo handover is significantly
  worse with the correct hint than with a same-task foreign-instance hint
  (72.5% versus 82.5%, Holm p=0.0337).
- [x] Finish both task-stratified held-out predictor folds and report latent
  cosine/error plus nearest-milestone retrieval by task. Across the two folds,
  latent cosine is 0.8134 versus 0.7479 for persistence (lift +0.0655), with
  retrieval top-1/top-5 of 46.9%/80.8%.
- [x] Finish the 24-cell state-dependent demo-retrieval upper-bound evaluation
  and retain episode outcomes for paired analysis. On 1,161 paired episodes,
  retrieval changes pooled success by only +0.34 pp (89.58% versus 89.23%,
  exact McNemar p=0.738; Holm-adjusted p=1.0), so it does not establish
  inference-time future-content utility.

The earlier unrestricted cells remain excluded because independent
accepted-seed searches changed 34--47 episode IDs per comparison. The strict
fixed-manifest reruns replace them; no further unrestricted reruns should be
submitted. The retrieval condition uses a task-scoped demonstration bank and
must be reported as a state-dependent demo-retrieval upper bound, not as a
future-peek rollout oracle.

Canonical retrieval outputs:

- `logs/eval_reports/rt_all6_v2_combo_oracle_retrieval_seed2026_unseen.json`
- `logs/eval_reports/robotwin_combo_oracle_retrieval_paired.json`
- `logs/l2_six_task_intervention_analysis.json`

## Decision outcomes

- **Scoped method:** replicated corrected-recipe utility with a clearly bounded
  task or integration regime; a future-content claim additionally requires the
  spatial privileged and predicted gates.
- **Integration study:** utility or representation effects persist, but correct
  future content remains causally null. This is the outcome most directly
  supported by current evidence.
- **Negative result:** corrected A0 removes the gain; retain the integration
  diagnosis and evaluation assets without a method-benefit claim.
