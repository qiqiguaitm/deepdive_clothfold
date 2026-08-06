# MINT-VLA Completed Evidence Archive

Updated: 2026-08-06 00:52 UTC

This document preserves completed evidence and execution notes removed from
`PAPER_TODO.md`. It is a lookup record, not an active task list. Canonical JSON
artifacts take precedence over prose if values differ.

## 1. Confirmatory data gate

The frozen `robotwin_milestone_all6_confirmatory_v1` artifact contains 1,200
successful episodes across all six RoboTwin tasks. The audit verifies nonzero
pair coverage, monotone future targets, a selection manifest, source hashes,
and a READY hash.

- `pairs.npz` SHA-256 prefix: `592f1e739105951e`
- READY audit SHA-256 prefix: `4353c725f7e55f9c`
- Historical preflight job: `t-20260801124942-kj5c9`

The earlier five-task artifact omits Stack-3 and must not be used for
confirmatory A3 training.

## 2. Fixed-checkpoint content interventions

E1 is complete for A2 and A3. Each architecture has correct, current, zero,
feature-permuted, cross-task, and within-task different-episode controls. Final
analysis uses one 518-scene intersection per architecture, exact McNemar tests,
and Holm correction.

### Final pooled success rates (%)

| Architecture | Correct | Current | Zero | Feature-permuted | Cross-task | Instance-shuffle |
|---|---:|---:|---:|---:|---:|---:|
| A2 | 70.46 | 70.27 | 66.80 | 72.20 | 71.04 | 69.88 |
| A3 | 71.24 | 72.97 | 72.97 | 72.97 | 72.20 | 74.13 |

Correct future is never detectably better than the key controls. All pooled
Holm-adjusted p-values are 1. The frozen interpretation is therefore a
token/package or representation effect, not inference-time future-content
causality.

Important terminology: feature-dimension permutation is not an instance
shuffle. The instance-shuffle control uses a future token from another episode
of the same task.

Canonical reports:

- `logs/pi05_a2_causal_with_instance.json`
- `logs/pi05_a3_causal_final.json`
- `logs/eval_reports/pi05_rt_a2_instance_shuffle_causal.json`
- `logs/eval_reports/pi05_rt_a3_instance_shuffle_causal.json`

Historical execution records:

| Work | Job or completion record | Note |
|---|---|---|
| A2 feature permutation | `t-20260801105042-9vmlj` | Completed |
| A3 current | `t-20260801113452-2dl8p` | Completed |
| A2 cross-task | `t-20260801120833-b92jb` | Completed |
| A3 zero | `t-20260801121908-fvngd` | All cells written |
| A3 feature permutation | `t-20260801122021-pnk8q` | All cells written |
| A3 cross-task feature/eval | `t-20260801124821-xbx7s`, `t-20260801125057-psl7p` | Completed |
| A3 instance shuffle | `t-20260801145730-59kq5`, completion `t-20260801160446-qn9c2` | First job stopped after 8/12 cells; immutable completion wrote the remainder |
| A2 instance shuffle | Local completion | All 12 cells written before wrapper returned `rc=2` |

Some platform wrappers returned failure after result cells were complete.
Artifact counts and immutable paired reports, rather than terminal wrapper
state, are authoritative for these evaluations.

## 3. Efficiency evidence

### Inference

The paired benchmark uses one GPU, three CHW `uint8` 480 x 640 camera views,
a 14-D state, five warm-up calls, 30 measured calls, and bootstrap 95%
confidence intervals.

| Metric | A0 | A2 | A3 |
|---|---:|---:|---:|
| Core model mean (ms) | 57.19 | 58.35 | 57.99 |
| WebSocket round-trip mean (ms) | 107.20 | 108.91 | 107.01 |
| Throughput (requests/s) | 9.33 | 9.18 | 9.34 |
| Warm inference memory (GiB) | 8.69 | 8.65 | 8.65 |

A2 and A3 throughput differ from A0 by -1.57% and +0.17%, respectively. No
material A3 inference-cost penalty is detected.

Canonical report: `logs/efficiency/pi05_a0_a2_a3_latency.json`.

### Training memory

The matched protocol uses four Shanghai A100s, batch 16, one FSDP device per
process, 10 measured updates, and 0.2 s telemetry.

| Arm | Peak memory (MiB) | State at freeze time |
|---|---:|---|
| A0 | 67,247 | Complete |
| A2 | 67,247 | Complete |
| A3 | 67,265 | Complete |

A2 exactly matches A0; the observed A3-A0 difference is 18 MiB (0.027%). Warm
inference memory must not be reported as peak training memory.

Canonical reports:

- `logs/efficiency/pi05_train_memory_a0.json`
- `logs/efficiency/pi05_train_memory_a2_abs.json`
- `logs/efficiency/pi05_train_memory_a3_live.json`

Historical jobs: A0 `t-20260801170905-s66sz`; A2
`t-20260801174722-zt5cw`; A3 `t-20260801171017-9852h`. All three reports
passed the artifact gate, closing E2.

## 4. Baseline and evaluator calibration

The public pi0.5 checkpoint completes the frozen four-seed, six-task bridge at
941/1,200 successes and a 78.42% macro:

| Task | Success (%) |
|---|---:|
| Hammer | 93.0 |
| Ranking size | 64.0 |
| Ranking RGB | 96.5 |
| Handover | 53.5 |
| Stack two | 93.5 |
| Stack three | 70.0 |

This establishes that the evaluator bridge can reproduce strong pi0.5
performance. The legacy internal A0 reaches 35.50% on the same bridge and
31.83% under the native protocol. Both share the same task-level failure
pattern, so the dominant issue is the checkpoint/training recipe rather than a
globally broken evaluator.

The legacy pilot used joint-delta actions, quantile normalization, batch 64,
and 20k updates. The corrected recipe uses absolute actions, mean/std, batch
16, and 50k updates. The pilot remains a matched exploratory ablation but must
not be reported as the confirmatory baseline.

Canonical report: `logs/eval_reports/pi05_public_samebridge_4seed_v3.json`.

### A0 public-recipe reproduction defect

The first nominally corrected 50k A0 reproduction is not a protocol-matched
baseline despite matching the headline optimizer, normalization, action,
batch, step, and seed settings. The North v2.1 data conversion has
`total_tasks=921,032`; within episode 20238, 168 frames contain 74 distinct
task indices. The released `lerobot/robotwin_unified` metadata has 23,559
tasks, and all 168 frames in that episode use one task index. The North videos
are encoded at 50 Hz, so their timestamps must remain at 50 Hz even though the
released v3 videos encode the same frame sequences at 30 Hz.

This defect is behaviorally consistent with the partial frozen evaluation:
the first four balanced tasks give hammer 94.5%, ranking RGB 71.5%, ranking
size 30.0%, and handover 11.5%, versus 93.0%, 96.5%, 64.0%, and 53.5% for the
released checkpoint on the same bridge. The evaluator is therefore not the
dominant explanation; language supervision changes within demonstrations.

A non-destructive replacement mirror now contains all 27,500 episodes and
6,075,103 frames with one official prompt per episode, while symlinking the
unchanged videos. The replacement A0 config also disables OpenPI's default
mild image augmentation to match the released recipe. The old 50k checkpoint
and its 24-cell evaluation are forensic artifacts and cannot unlock A2/A3.

## 5. Legacy pi0.5 pilot

| Arm | Six-task macro (%) | Delta from A0 (pp) |
|---|---:|---:|
| A0 no hint | 35.50 | -- |
| A1 shallow absolute | 41.17 | +5.67 |
| A2 offline absolute | 48.75 | +13.25 |
| A2 offline residual | 43.83 | +8.33 |
| A3 live visual residual | 49.58 | +14.08 |

A3 exceeds A2 absolute by only 0.83 pp. Offline residualization is 4.92 pp
below A2 absolute. These numbers motivate the corrected matrix but do not
establish a stable method gain.

## 6. LaWAM supporting diagnostics

The LaWAM `no-WM` row must be named **LaWAM-init / Future-off** because it
inherits WM-aware pretraining. It diagnoses downstream WM objectives but is
not a pure-VLA baseline.

Completed training-seed means at the latest archive update:

| Arm | Training seeds | Macro (%) |
|---|---:|---:|
| Future-off | 3 | 90.83 |
| Absolute | 2 | 89.21 |
| Combo | 3 | 88.78 |
| Gradient isolation | 2 | 89.04 |
| Residual | 2 | 87.50 |
| Local-WM | 2 | 86.67 |

The current matrix does not show a downstream WM-objective gain over matched
Future-off initialization. Combo and Future-off have three completed training
seeds; absolute, residual, local-WM, and gradient isolation have two. The
authoritative matrix is
`logs/eval_reports/robotwin_all6_v2_training_seed_matrix.json`.

This aggregate result does not imply that LMWM has no effect. Relative to the
active `local-WM` arm, absolute and combo improve by `+2.54 pp` and `+2.11 pp`,
respectively. At task level, the three-seed combo mean improves stack-three by
`+7.33 pp`, ranking size by `+2.67 pp`, handover by `+2.42 pp`, and ranking RGB
by `+1.00 pp` relative to the two-seed local-WM mean. Only stack-three also
exceeds the three-seed Future-off mean (`+7.33 pp`).
The correct interpretation is selective recovery of an active LaWAM
interface, not a demonstrated net benefit over disabling downstream future
training.

### Completed task-duration audit

The frozen pair artifact gives mean demonstration lengths of 114.1 frames for
hammer, 284.3 for handover, 313.2 for stack-two, 459.2/459.7 for ranking
RGB/size, and 470.7 for stack-three. Mean duration has an exploratory
association with absolute-minus-local and combo-minus-local gains (Spearman
`rho=0.829`, unadjusted `p=0.042`, six post-hoc task points). This is not a
validated regime law: ranking-RGB is long but residual-only loses `10.00 pp`,
and stack-two is saturated.

Headroom alone is also insufficient. With `100 - Future-off success` as the
proxy, its exploratory association is `rho=0.486` (`p=0.329`) with
combo-minus-local recovery and `rho=-0.657` (`p=0.156`) with
combo-minus-Future-off net benefit across the six tasks.

The completed six-task content interventions also reject a duration-only
explanation. Correct hints do not beat zero, cross-task, or within-task
controls in any of the nine pooled comparisons after Holm correction, and the
only corrected task-level contrast is adverse on Handover. The current
diagnosis is a mixture of weak content use and route-dependent harmful
conditioning; the post-hoc duration association is not a validated interaction.

For no-WM and absolute seed-2027 training, terminal platform failures occurred
after the 20k state and final model had been written and NCCL ranks had shut
down cleanly. Those artifacts are authoritative and the trainings must not be
repeated solely because of wrapper state.

## 7. Archived interpretation

- The strongest current positive claim is low-cost integration: +0.25%
  parameters, no material A3 inference penalty, and a 0.027% A3-A0 training
  memory difference under the completed three-arm protocol.
- Correct future content is causally null at the completed fixed checkpoints.
- The corrected A0/A2/A3 utility matrix remains the decision gate between a
  scoped method paper, an integration study, and a negative result.
- Ordered construction is exploratory: it was identified post hoc, contains
  only two block-stacking tasks in the pilot, and requires preregistered bowl
  stacking, fixed-horizon controls, and independent training seeds.

## 8. Held-out prediction and retrieval diagnostics

Two task-stratified folds cover 480 held-out episodes. The learned predictor
reaches latent cosine 0.8134 versus 0.7479 for persistence, a lift of 0.0655;
milestone retrieval reaches 46.9% top-1 and 80.8% top-5. This establishes that
the predictor contains nontrivial future information. It does not establish
that the policy uses this information to improve control.

A task-scoped, state-dependent demonstration retrieval condition was evaluated
on 1,161 paired episodes. It reaches 89.58% success versus 89.23% for the
reference condition, a +0.34 pp difference (exact McNemar p=0.738;
Holm-adjusted p=1.0). Better retrieved milestone content therefore does not
repair the evaluated one-token interface.

Canonical reports:

- `logs/eval_reports/robotwin_lmwm_heldout_twofold.json`
- `logs/eval_reports/rt_all6_v2_combo_oracle_retrieval_seed2026_unseen.json`
- `logs/eval_reports/robotwin_combo_oracle_retrieval_paired.json`

## 9. Strict six-task hint diagnosis

All nine method-condition comparisons are complete on the frozen six-task
panel: 216/216 cells and 9/9 completion markers, with 1,200 identical scenes
per comparison. Correct hints do not beat zero, cross-task, or within-task
foreign-instance controls at the pooled level for Absolute, Residual, or
Residual+stop-gradient; every pooled Holm-adjusted p-value is 1.

The only multiplicity-corrected task-level contrast is adverse. On Handover,
Residual+stop-gradient reaches 72.5% with the correct hint and 82.5% with a
same-task foreign-instance hint (difference -10.0 pp; Holm-adjusted p=0.0337).
This closes the one-token future-content claim but does not generalize to every
world-model interface.

Canonical report: `logs/l2_six_task_intervention_analysis.json`.

## 10. Privileged spatial-interface gate

The parameter-matched S0-N/S0-C/S0-P probe completed 1,000 updates and frozen
evaluation on 320 held-out samples across Stack-3 and Hammer. On Stack-3,
privileged endpoint L2 is 0.3378 versus 0.3274 for no goal and 0.3580 for
current-image patches, so the privileged target does not beat both controls.
On Hammer, it is 0.3080 versus 0.2355 and 0.2403. Episode-level paired
bootstrap estimates the Hammer regressions at +0.0725 versus no goal (95% CI
[+0.0391, +0.1110]) and +0.0677 versus current patches ([+0.0332, +0.1054]);
lower is better.

The frozen T3a gate therefore failed. T3b predictor expansion was not run, and
no pass marker was published. This is a two-task offline action-endpoint probe,
not a closed-loop success result.

Canonical report: `logs/spatial_s0/s0_offline_verdict.json`.

## 11. Corrected exact A0 seed-1000 baseline

The corrected A0 seed-1000 run completed 50k updates under the public-recipe
protocol: absolute actions, mean/std normalization, batch 16, augmentation
disabled, and one official prompt per episode in the 27,500-episode mirror.
The zero-indexed step-49,999 checkpoint, normalization assets, launch snapshot,
dataset manifest, and source provenance passed the immutable gate with no
failures.

The frozen evaluation contains 24/24 task-by-evaluation-seed cells and 1,200
episodes on manifest SHA-256
`08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`.
Task-level success rates are:

| Task | Successes / episodes | Success (%) |
|---|---:|---:|
| Hammer | 176 / 200 | 88.0 |
| Ranking size | 123 / 200 | 61.5 |
| Ranking RGB | 188 / 200 | 94.0 |
| Handover | 104 / 200 | 52.0 |
| Stack two | 174 / 200 | 87.0 |
| Stack three | 141 / 200 | 70.5 |
| **Macro / micro** | **906 / 1,200** | **75.5** |

The 75.5% macro exceeds the preregistered 70% operational gate. This accepts
A0 as the corrected seed-1000 baseline; it is not an A2/A3 method effect and
does not provide multi-training-seed uncertainty. The weaker Handover and
Ranking-size cells remain visible rather than being hidden by the macro.

Canonical evidence:

- `logs/pi05_a0_public_exact_gate.json`
- `logs/resource_markers/pi05_a0_public_exact_seed1000_eval.ok`
- `lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a0_public_exact_seed1000/`
- `lmvla/paper_iclr_lmvla/manifests/pi05_confirmatory_a0_seed1000_eval_launch.json`

Training completed at 2026-08-02 01:22 UTC. The verified checkpoint copy to
North completed at 02:11 UTC, and the canonical evaluation gate completed at
04:18 UTC. Independent seed-0 Stack-2 and Stack-3 probes of the same final
checkpoint (88% and 68%) are diagnostic only and do not replace the 24 frozen
cells. The earlier prompt-defective reproduction remains invalid.

## 12. Action-fixed midpoint diagnostics

The first step-20k A2/A3 diagnostics were invalid because inference applied an
absolute-action conversion to checkpoints already trained in absolute-action
space. Their roots are excluded from paper tables. After the inference route
was repaired, the `_actionfix` panel completed all 16 frozen seed-0 cells. A2
reaches 125/300 (41.7%) and A3 95/300 (31.7%) across the six correct-hint
tasks. On Stack-2, correct-minus-zero/current differences are +8/-6 pp for A2
and -4/-4 pp for A3; all paired-bootstrap intervals cross zero and all exact
McNemar p-values are at least 0.52.

These results verify plausible control and preserve the one-token content-null
boundary. They are diagnostic only because there is no matched step-20k A0 and
they cannot substitute for the step-49,999 T1 matrix.

Canonical report: `logs/diagnostics/pi05_actionfix_midpoint_step20000.json`.

## 13. Single-seed interpretation boundary (superseded)

This 11:18 UTC boundary is retained as chronological history and is superseded
by the complete three-seed result in Section 16.

- Corrected A0 seed 1000 is a valid 75.5% baseline, with task rates ranging
  from 52.0% on Handover to 94.0% on Ranking RGB.
- Matched A3 seed 1000 is a valid adverse result at 66.92%, 8.58 pp below A0;
  A2 seed 1000 and the remaining training-seed evaluations are unfinished.
- Future-state representation quality remains separate from control utility.
- The adverse A3 result is direct control evidence for one training seed, but
  it is not a final multi-seed estimate.

## 14. Completed confirmatory training and safety diagnostics

All nine A0/A2/A3 confirmatory trainings for seeds 1000, 1001, and 1002 reached
50k updates and wrote their checkpoint and normalization artifacts. Completion
of training does not make an arm admissible: each checkpoint still requires a
frozen 24-cell evaluation and protocol audit.

The step-40k Hammer/Stack-3 safety panel is also complete on 50 frozen seed-0
scenes per task. Relative to final A0, A2-Abs is -8 pp on Hammer (84% versus
92%) and +20 pp on Stack-3 (78% versus 58%); A3 is -10 pp and -2 pp (82% and
56%). A2 Stack-3 has 4 A0-only versus 14 A2-only successes (exact McNemar
`p=0.0309`, uncorrected), but the pooled two-task comparison is not significant
(10 versus 16, `p=0.327`). These are unmatched-checkpoint diagnostics and are
excluded from confirmatory paper tables.

Canonical report: `logs/diagnostics/pi05_step40000_safety.json`.

## 15. Corrected A3 seed-1000 control result

The corrected A3 seed-1000 evaluation completed all 24 frozen cells and 1,200
episodes on the accepted scene manifest. The protocol audit accepts the A0 and
A3 rows with zero unmatched episode keys. A3 records 803/1,200 successes
(66.92%) versus 906/1,200 (75.50%) for matched A0, a paired macro difference
of -8.58 pp with a hierarchical-bootstrap 95% interval of [-11.75, -5.25] pp.

| Task | A0 success (%) | A3 success (%) | A3 - A0 (pp) |
|---|---:|---:|---:|
| Hammer | 88.0 | 83.5 | -4.5 |
| Ranking RGB | 94.0 | 90.0 | -4.0 |
| Ranking size | 61.5 | 40.5 | -21.0 |
| Handover | 52.0 | 45.5 | -6.5 |
| Stack two | 87.0 | 84.0 | -3.0 |
| Stack three | 70.5 | 58.0 | -12.5 |
| **Macro / micro** | **75.5** | **66.92** | **-8.58** |

Every task-level difference is negative, with the largest regressions on
Ranking size and Stack three. This is direct control evidence that the tested
A3 integration underperforms A0 at training seed 1000. It does not establish a
multi-training-seed effect, and it does not turn the separate predictor-quality
result into evidence of control utility.

Canonical evidence:

- `logs/eval_reports/pi05_rt_a3_live_confirmatory_s1000.json`
- `logs/eval_reports/pi05_confirmatory_training_seed_matrix.json`
- `lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a3_live_confirmatory_s1000/`

## 16. Corrected three-seed confirmatory matrix

All nine A0/A2-Abs/A3 rows are complete and accepted under the frozen
24-cell protocol. Each trained policy is evaluated on six tasks, four evaluator
seeds, and 50 episodes per task--evaluator-seed cell: 1,200 episodes per policy
and 10,800 episodes overall. All reports pass checkpoint, configuration,
normalization, dataset, launch, source, exact scene-order, and manifest audits.
The accepted-scene manifest SHA-256 is
`08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`.

### Seed-level macro success (%)

| Arm | Seed 1000 | Seed 1001 | Seed 1002 | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| A0 | 75.50 | 80.00 | 82.67 | 79.39 | 2.96 |
| A2-Abs | 70.00 | 74.67 | 81.67 | 75.44 | 4.79 |
| A3 | 66.92 | 68.92 | 76.58 | 70.81 | 4.17 |

The paired hierarchical bootstrap resamples training seeds and then paired
episodes within task. A2-Abs--A0 is -3.94 pp (95% interval [-6.58, -0.97]) and
A3--A0 is -8.58 pp ([-11.47, -5.75]), each over 3,600 paired episodes with zero
unmatched keys. Per-training-seed deltas are -5.50/-5.33/-1.00 pp for A2-Abs
and -8.58/-11.08/-6.08 pp for A3; every paired seed delta is negative.

### Three-seed task means (%)

| Task | A0 | A2-Abs | A2 - A0 (pp) | A3 | A3 - A0 (pp) |
|---|---:|---:|---:|---:|---:|
| Hammer | 89.50 | 86.33 | -3.17 | 80.83 | -8.67 |
| Ranking RGB | 95.00 | 89.17 | -5.83 | 88.50 | -6.50 |
| Ranking size | 61.83 | 55.83 | -6.00 | 48.83 | -13.00 |
| Handover | 59.83 | 54.33 | -5.50 | 51.00 | -8.83 |
| Stack two | 92.83 | 93.33 | +0.50 | 90.00 | -2.83 |
| Stack three | 77.33 | 73.67 | -3.67 | 65.67 | -11.67 |

A2-Abs improves only Stack two and regresses the other five tasks. A3 regresses
all six tasks. The corrected official-aligned baseline therefore removes the
legacy milestone gain. This is direct, replicated control evidence for the
tested interfaces; it does not imply that the predictor lacks future-state
information or that every world-model interface is ineffective.

Canonical evidence:

- `logs/eval_reports/pi05_confirmatory_training_seed_matrix.json`
- `lmvla/paper_iclr_lmvla/RESULTS_pi05_confirmatory_3seed_2026-08-02.md`
- the nine accepted per-policy reports under `logs/eval_reports/`
- the nine raw evaluation roots under
  `lmvla/lawam/results/eval_runs/robotwin/`

## 17. Closed expansions and selected outcome

T1 and T2 are complete. No fourth seed is required because all three paired
deltas are negative for both conditioned arms. The preregistered positive-
utility prerequisite failed, so T4 selector/task-scope experiments, T5 mature-
initialization transfer, and T6 second-VLA instantiation are closed rather than
unfinished. They must not be tuned or reopened against the confirmatory result.

The selected paper outcome is a negative integration result. The publishable
claim is that future representations can be predictable and inexpensive while
the tested native-space milestone interfaces still reduce closed-loop control
success. L1 remains conditional and is required only for a claim about whether
LM pretraining benefits a clean VLA.

## 18. Milestone-transition diagnostics and frozen preparation

This section archives completed preparation and intermediate diagnostics for a
new event-level milestone-transition hypothesis. The diagnostics themselves do
not change the paper result in Section 16. The completed seed-1000 formal gate
is recorded in Section 19; the claim-bearing authority is still the unfinished
three-training-seed interval.

MT1 uses a privileged same-scene expert trajectory to align current joint
state monotonically to automatically mined task-local transition boundaries.
MT2 is a parameter-matched null-transition control trained with the same
initialization, data, optimizer, update count, and adapter capacity. Both use
the corrected 27,500-episode pi0.5 recipe. Directed implementation tests, a
real one-step forward/backward/checkpoint smoke, checkpoint-sentinel audits,
and East/North execution preflights passed. The step-40,000 recovery
checkpoints committed atomically, with root, parameter, train-state, and
normalization metadata in both approximately 31-GiB artifacts. At this
preparation boundary, both final checkpoints were still unfinished; their later
completion is archived below. Training completion alone is not control-utility
evidence.

### Five-scene trajectory diagnostics

All probes use only Hammer, evaluator seed 0, and five frozen scenes from
`robotwin_pi05_mt_preflight_scene_seeds_v1.json` (manifest SHA-256
`bcb6886e23a079c33f86df32127005785d7d99748c74b7c76e5e73915f9396b9`).
They are smoke/trajectory diagnostics and cannot replace the 24-cell protocol.

- At step 5,000, MT1 correct conditioning reaches 1/5.
- At step 10,000, MT1 correct and null each reach 1/5 and succeed on the same
  scene; checkpoint-age-matched A0 reaches 0/5.
- At step 25,000, MT1 correct and null each reach 3/5, while
  checkpoint-age-matched A0 reaches 2/5. Correct versus null has one discordant
  win in each direction, so the paired net content effect is zero. Each MT1
  condition gains two scenes and loses one relative to A0.

The step-25,000 result is compatible with faster recovery of the shared
transition-training path on this small probe, but it provides no evidence that
correct transition content improves control. No protocol or hyperparameter was
changed in response.

Canonical diagnostic record:

- `lmvla/paper_iclr_lmvla/RESULTS_pi05_mt_transition_step25000_probe_2026-08-02.md`
- `logs/resource_markers/pi05_mt1_oracle_seed1000_step25000_correct_probe.ok`
- `logs/resource_markers/pi05_mt1_oracle_seed1000_step25000_null_probe.ok`
- `logs/resource_markers/pi05_a0_seed1000_step25000_mt_matched_probe.ok`
- the three corresponding raw roots under
  `lmvla/lawam_local/results/eval_runs/robotwin/`

### Frozen learned-tracker branch

The MT3 task-stratified split, held-out metrics and selector, same-pi0.5-encoder
feature interfaces, tracker-only training path, joint-policy path, intervention
matrix, task-scope analysis, and gate logic were frozen before any MT1 final
result. The split contains 960 training and 240 validation episodes with zero
episode leakage. The data audit covers 420,238 labeled rows and records the
strong stage imbalance, including only 36 stage-8 rows. Tracker metrics remain
representation evidence only; MT3 feature extraction, tracker optimization,
policy training, and closed-loop evaluation are blocked unless MT1/MT2 pass.

Canonical frozen records:

- `lmvla/lmwm/data/robotwin_mt_stage_tracker_split_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/robotwin_mt3_protocol_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/robotwin_mt3_data_audit_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/robotwin_mt6_scope_v1.json`

### Frozen selected-method efficiency protocol

The conditional MT6 efficiency task compares the selected MT3 seed-1000 graph
with the accepted clean pi0.5 A0 seed-1000 graph on the same allocated GPU and
process environment. It reports exact model-state parameter counts, compiled
XLA cost-analysis FLOPs, warm inference memory, direct-model latency,
WebSocket round-trip latency, and throughput over five warmups and 30 trials.
The task is blocked on the MT4 content gate and frozen tracker selection.

Peak training memory uses a separate matched 10-step probe on four Shanghai
A100s with batch 16, eight data workers, and the same pi0.5 initialization and
transition inputs. The selected MT3 probe is compared directly with the frozen
A0 peak of 67,247 MiB. A diagnostic-only `save_final_checkpoint=false` switch
prevents an irrelevant approximately 31-GiB final checkpoint write while
preserving the default final-save behavior for all training jobs. This probe
may run on gf1 GPUs 0--3 or the four-A100 robot-task flavor; H20 measurements
are excluded from the matched comparison.

A two-trial A0 engineering smoke validated the measurement path: 3,353,433,872
parameters, 172.51 GFLOPs per request from the compiled executable, about 8.65
GiB warm inference memory, and about 67 ms direct-model latency. The latency and
throughput estimates from two trials are not admissible paper results; only the
conditional matched 30-trial report is.

Canonical implementation records:

- `train_scripts/kai/analysis/benchmark_pi05_policy_latency.py`
- `train_scripts/kai/eval/local_pi05_mt6_efficiency_1gpu.sh`
- `train_scripts/kai/volc/pi05_mt6_efficiency_east_1h20.yaml`
- dynamic task `pi05_mt6_selected_efficiency`
- `train_scripts/kai/eval/pi05_mt6_selected_train_memory_4a100.sh`
- `train_scripts/kai/volc/pi05_mt6_train_memory_cnsh_4a100.yaml`
- dynamic task `pi05_mt6_selected_train_memory`

### Frozen MT5 temporal-scale 2x2

The conditional MT5 comparison was made executable before observing MT4. The
fixed local target is exactly 1.0 seconds (50 frames at the official 50 Hz),
chosen to match pi0.5's 50-step action chunk. Targets are never clamped at an
episode tail: all policy samples remain, while unavailable auxiliary targets
receive mask zero. The frozen index covers 4,700,103 valid pairs over all
27,500 episodes; the 97-GiB cam-high cache covers episode 0 through 27,499.

The local adapter consumes stop-gradient current pi0.5 visual features and the
mean, displacement, and standard deviation of the candidate action chunk. It
predicts a residual +1.0-s feature, receives a stop-gradient current-encoder
target, and injects a gated residual only into action-expert tokens. The
combined arm is the exact union of this route and the selected MT3 transition
route. A batch-16 one-step GPU smoke completed with main loss 0.3381 and local
auxiliary loss 7.2940 (weight 0.05); this is implementation evidence only.

The staged experiment contains local and combined arms at seeds 1000--1002,
six frozen 24-cell evaluations, and one hierarchical analysis. Complementarity
is accepted only if combined beats both local and selected-transition arms and
both 95% interval lower bounds exceed zero.

Canonical records:

- `lmvla/paper_iclr_lmvla/manifests/robotwin_mt5_protocol_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/robotwin_mt5_fixed_horizon_data_v1.json`
- `lmvla/lmwm/data/robotwin_fixed_horizon_1s_v1/pairs.npz`
- `train_scripts/kai/run_pi05_mt5_train.sh`
- `train_scripts/kai/eval/run_pi05_mt5_formal.sh`
- dynamic tasks `pi05_mt5_*` (13 tasks)

### Step-40,000 robot-task A100 transition smoke

A two-GPU robot-task diagnostic loaded the MT1 oracle and MT2 null-trained
step-40,000 checkpoints concurrently on A100s, restored 6.2 GiB of parameters
per process in about eight seconds, started both WebSocket policy servers, and
completed one `stack_blocks_two` simulator episode per arm. The MT1-correct
episode succeeded in 292 steps with six model queries; the MT2-null episode
reached the 800-step limit with 16 model queries. Both task schedulers exited
with status `ok` and emitted complete `summary.json` records.

This one randomly seeded episode per arm is implementation evidence only. It
does not enter the frozen 24-cell matrix and cannot support an effect claim.
Its purpose is to establish that the last-priority robot-task fallback can
load both transition configurations and execute the full A100 closed loop
before final checkpoint availability.

Canonical records:

- `logs/resource_markers/pi05_mt_transition_cnsh_step40000_smoke.ok`
- `train_scripts/kai/volc/pi05_mt_transition_cnsh_step40000_smoke_2a100.yaml`
- `lmvla/lawam/results/eval_runs/robotwin/pi05_mt1_s40000_correct_cnsh_smoke_v2/`
- `lmvla/lawam/results/eval_runs/robotwin/pi05_mt2_s40000_null_cnsh_smoke_v2/`

### MT1/MT2 seed-1000 final training completion

The frozen seed-1000 MT1 oracle-transition and parameter-matched MT2
null-transition arms completed on gf1 at 02:40 and 02:43 UTC on 2026-08-03.
Both four-A100 jobs exited with code zero after 50,000 updates. Their
step-49,999 checkpoints are approximately 31 GiB each and contain atomically
committed root, parameter, train-state, and checkpoint-local normalization
metadata.

The launch manifests preserve the raw pi0.5 initialization, 27,500-episode
dataset, batch size 16, absolute actions, mean/std normalization, no image
augmentation, eight data workers, transition-pair hash, and execution-source
hashes. This closes the training and artifact-audit items only. The five
1,200-episode closed-loop control evaluations and their paired gate analysis,
archived below, remain the authority for utility and content claims.

Canonical records:

- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1000_train_launch.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt2_null_seed1000_train_launch.json`
- `kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/pi05_robotwin_mt1_oracle_seed1000/49999/`
- `kai0/checkpoints/pi05_robotwin_mt2_null_exact/pi05_robotwin_mt2_null_seed1000/49999/`

## 19. MT1/MT2 seed-1000 formal control gate

All five frozen seed-1000 matrices are complete and accepted. Each contains
24/24 task-by-evaluator-seed cells and 1,200 episodes on the accepted scene
manifest (SHA-256
`08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`).
All five reports contain identical scene keys.

### Task-level success (%)

| Task | A0 | Correct | Null input | Within-task | Cross-task | Null-trained |
|---|---:|---:|---:|---:|---:|---:|
| Hammer | 88.0 | 88.5 | 86.5 | 84.0 | 81.0 | 86.5 |
| Ranking RGB | 94.0 | 93.5 | 90.5 | 86.0 | 92.0 | 92.0 |
| Ranking size | 61.5 | 56.0 | 53.5 | 52.0 | 59.5 | 51.5 |
| Handover | 52.0 | 60.5 | 64.0 | 61.5 | 58.0 | 61.5 |
| Stack two | 87.0 | 91.5 | 94.5 | 94.5 | 93.5 | 92.0 |
| Stack three | 70.5 | 76.5 | 75.0 | 75.5 | 75.0 | 78.0 |
| **Macro / micro** | **75.50** | **77.75** | **77.33** | **75.58** | **76.50** | **76.92** |

Correct is +2.25 pp over A0, +0.42 pp over null input, +2.17 pp over
within-task permutation, +1.25 pp over cross-task input, and +0.83 pp over the
parameter-matched null-trained checkpoint. Relative to A0, correct improves
Handover (+8.5 pp), Stack-2 (+4.5 pp), Stack-3 (+6.0 pp), and Hammer (+0.5 pp),
but regresses Ranking RGB (-0.5 pp) and Ranking size (-5.5 pp).

The content contrasts are not uniformly positive by task. Correct is lower
than null input on Handover (-3.5 pp) and Stack-2 (-3.0 pp), lower than
within-task permutation on Handover (-1.0 pp) and Stack-2 (-3.0 pp), lower than
cross-task input on Ranking size (-3.5 pp) and Stack-2 (-2.0 pp), and lower
than null-trained on Handover (-1.0 pp), Stack-2 (-0.5 pp), and Stack-3
(-1.5 pp).

The frozen paired analysis contains 35 pooled and task-level exact McNemar
tests in one Holm family. No adjusted comparison is significant; the minimum
Holm-adjusted p-value is 0.96366. The pooled correct-versus-A0 discordance is
185 correct-only versus 158 A0-only successes (unadjusted p=0.1603), and the
pooled correct-versus-null-input discordance is 158 versus 153
(p=0.8206). The evidence therefore does not yet support a publishable robust
utility or transition-content claim.

The preregistered directional pilot gate nevertheless accepts replication:
correct exceeds all declared pooled controls and improves all three
predeclared multistage tasks relative to A0. This decision only authorizes
frozen seeds 1001--1002. The final method claim still requires the three-seed
hierarchical 95% interval to exclude zero and cannot be inferred from the
seed-1000 macro.

Canonical records:

- `lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_seed1000_controls.json`
- `lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_seed1000_gate.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1000_correct.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1000_null.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1000_within_task.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1000_cross_task.json`
- `lmvla/lmwm/docs/pi05_mt2_null_seed1000_eval.json`
- the five corresponding `.ok` markers under `logs/resource_markers/`

## 20. MT1 oracle-transition replication launch

The directional seed-1000 pilot gate unlocked the frozen replication branch.
At 05:37:42 UTC on 2026-08-03, seed 1001 and seed 1002 began concurrently on
gf1 GPUs 0--3 and 4--7, respectively. Both runs use the same immutable recipe
as seed 1000: raw pi0.5 initialization, 27,500 episodes, batch 16, 50,000
updates, absolute actions, mean/std normalization, no image augmentation, and
eight data workers.

Before dispatch, the scheduler validator was found to be applying the atomic
saved-step sentinel to the legacy parameter-only `pi05_base` initialization.
The rule was narrowed to numeric saved-step directories. Final trained
checkpoints and every formal evaluator retain their stricter atomic root,
parameter, train-state, and checkpoint-local normalization gates. A regression
test covers the parameter-only base case; the scheduler suite passes 52 tests.

Canonical launch records:

- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1001_train_launch.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1002_train_launch.json`

The first resumable checkpoints for both replications committed at step 5,000
at 06:49 UTC. Each is approximately 31 GiB and includes atomic root, parameter,
train-state, and checkpoint-local normalization metadata. Both jobs continued
past step 5,200 after the save, establishing that checkpoint serialization did
not stall training and that a subsequent retry can select a complete recovery
point rather than restart from raw initialization.

The post-training shared evaluation fan-out was expanded before final
checkpoint availability. Each replication now has one four-GPU East helper,
four independent four-GPU robot-task helpers, and one two-GPU local helper in
addition to its four-GPU gf1 parent. The two seeds can therefore consume the
full 32-GPU robot-task allowance rather than only eight GPUs. Worker-index and
marker namespaces are disjoint, while all workers claim cells through the same
four frozen per-seed schedulers and the same atomic 24-cell finalizer. The
resource priority remains gf1, East, North when data-local, robot-task, then
local; the additional robot-task shards affect only otherwise idle capacity.

Both step-10,000 replication checkpoints committed atomically at 07:52--07:53
UTC. Serialization took 15.36 seconds for seed 1001 and 16.59 seconds for seed
1002. Each 31-GiB checkpoint contains root, params, train-state, and
checkpoint-local normalization metadata; both jobs continued past step 10,600
with finite losses and no health event.

Submission placement is now exposed through
`train_scripts/kai/volc/recommend_submission_target.py` and its versioned
resource catalog. The router combines requested card count, live free cards,
queue state, East/North data locality, and independently configurable Beijing
GPU and submitted-job limits. It prefers an immediately runnable target over
queueing; if every target must wait, North is the designated queue sink. This
operational policy does not alter any frozen scientific configuration.

At 08:56 UTC, the seed-1001 and seed-1002 step-15,000 checkpoints committed
atomically in 18.84 and 17.75 seconds, respectively. Both 31-GiB artifacts pass
the root, params, train-state, and checkpoint-local normalization audit. The
jobs continued past step 15,100 with finite losses and 87--100% sampled GPU SM
utilization.

At 10:00 UTC, both step-20,000 checkpoints committed atomically. Seed 1001
required 16.85 seconds and seed 1002 required 16.04 seconds. Both 31-GiB
artifacts pass the same four-part audit, and training continued past step
20,200 with finite losses near 0.007--0.010.

At 11:04 UTC, the seed-1001 and seed-1002 step-25,000 checkpoints committed
atomically in 15.23 and 16.32 seconds. Both 31-GiB artifacts pass the four-part
audit, and both jobs continued past step 25,100 with finite losses near
0.0053--0.0056. The frozen replications are therefore beyond half of their
update budget without a health or checkpoint-integrity event.

At 11:43 UTC, a transient SSH transport failure during the gf1 launcher PID
probe caused the resource scheduler to reclaim both reservations in local
state. This was an orchestration-state error only: both launchers and both
trainers remained alive, training logs continued, and no duplicate process was
started. Scheduler control was stopped, the four remote PIDs and disjoint GPU
assignments were re-audited, both records were restored to `running`, and the
control loop was restarted. The monitor now distinguishes transport errors from
an explicit dead PID and requires three consecutive `DEAD` probes before
reclaim; an `ALIVE` probe resets that counter. The combined scheduler, router,
MT1--MT6 gate, tracker, complementarity, and scope suite passes 96/96 tests.

At 12:08 UTC, the seed-1001 and seed-1002 step-30,000 checkpoints committed
atomically in 15.00 and 16.26 seconds. Both approximately 31-GiB artifacts
contain the root sentinel, parameters, train state, and checkpoint-local
`robotwin2.0_absolute_meanstd` normalization metadata. Both trainers continued
past step 30,300 with finite losses and no health event.

At 12:30 UTC, the scheduler submission path was hardened so every nonzero-GPU
launch first invokes the shared placement router on the current in-memory
resource snapshot and the candidate's concrete data/checkpoint paths. Each
decision is atomically preserved under `logs/submission_recommendations/` with
the global and task-eligible rankings and the final selection; router failure
blocks launch. A dry audit of the pending MT1 replication fan-out selected gf1
for a parent when free and robot-task for a robot-task-only helper. The expanded
scheduler, router, MT1--MT6 gate, tracker, complementarity, and scope suite
passes 98/98 tests. The restarted control loop retained both live replication
reservations and cleared stale pre-dispatch waiting text from their running
state.

A full effective-queue audit at 12:34 UTC mapped every triggered P0--P3 TODO
item to its task, gate, completion artifact, and downstream consumer. The gate
closure regression now includes MT5 explicitly and verifies that MT1/MT3
rejection propagates through MT5, while MT4 content rejection closes every MT5
training/evaluation/complementarity task and the MT6 scope/efficiency branch.
The suite passes 99/99 tests. The second-architecture and separate LM-pretraining
branches remain uninstantiated because their frozen triggers are false.

## 21. MT1 step-35,000 and matched 4/8-GPU scaling audit

At 13:12 UTC, both replication step-35,000 checkpoints committed atomically.
Seed 1001 finalized in 16.04 seconds and seed 1002 in 15.05 seconds. Each
approximately 31-GiB directory contains the atomic root sentinel, parameter
metadata, train-state metadata, and checkpoint-local
`robotwin2.0_absolute_meanstd` normalization statistics. Both trainers
continued after finalization with finite losses and no health event.

A non-claim-bearing strong-scaling probe compared four and eight A100 GPUs
under the same MT1 configuration, global batch 16, seed 4242, and 300 updates.
The four-GPU run used per-device batch four and measured 0.7090346 seconds per
step (22.5659 samples/s). The eight-GPU run used per-device batch two and
measured 0.5187063 seconds per step (30.8460 samples/s). The resulting speedup
is 1.3669x, or 68.35% parallel efficiency. Eight GPUs reduce one arm's wall
clock by 26.84%, but increase GPU-seconds per completed step by 46.31%; two
concurrent four-GPU arms provide 46.31% more aggregate sample throughput than
one eight-GPU arm. Four GPUs therefore remain the default for the claim-bearing
experiment matrix, while eight GPUs are justified only for a single
critical-path arm whose latency dominates aggregate throughput.

The eight-GPU probe also exposed a resource-control edge case. A robot-task job
reclaimed after Queueing from zero observed occupancy could remain in cooldown
indefinitely. The scheduler now retries after the ordinary cooldown when
occupancy has not fallen, and retries immediately when it has; the diagnostic's
queue tolerance was raised from 120 to 900 seconds. The complete router,
scheduler, MT1--MT6 gate, tracker, complementarity, and scope suite passes
110/110 tests. After both probes completed and the GF1 race fallbacks were
installed, the effective queue contained 335 tasks: 218 completed, two running,
77 gate-blocked pending, and 38 disabled.
The four-GPU static placement order was also reconciled to the latest declared
policy: gf1, Robot-East-H20, Robot-North-H20, then robot-task, while preserving
immediate-capacity precedence, filesystem locality, and North as the all-queued
fallback.
The same 900-second tolerance is used by the final MT1 East and robot-task
helpers, which have no useful alternative target after dispatch. Parent
evaluations retain their shorter timeout because they can genuinely switch
between GF1 and East.

The final MT1 replication evaluation fan-out was also exercised through the
real dispatch allocator with launch APIs mocked only after target selection.
Two scheduler polls place two four-GPU parents on gf1, two four-GPU helpers on
Robot-East-H20, eight four-GPU helpers on robot-task, and one two-GPU helper on
local, for 13 simultaneous tasks and 50 physical GPUs. The fourteenth task is
the second local helper and correctly remains pending until the sole local
two-GPU slot is released.

Each seed additionally has a fixed-index four-GPU GF1 fallback helper. It is
suppressed by capacity while the parent occupies GF1, but can fill GPUs 0--3 or
4--7 if a parent starts immediately on East before its trainer releases GF1.
A race simulation reserves both parents on East, releases GF1, and then fills
all eight GF1 GPUs with the two non-overlapping fallback helpers.

Canonical records:

- `logs/scaling/pi05_mt1_b16_scaling_probe_4g.json`
- `logs/scaling/pi05_mt1_b16_scaling_probe_8g.json`
- `logs/submission_recommendations/pi05_mt1_b16_scaling_probe_4g/`
- `logs/submission_recommendations/pi05_mt1_b16_scaling_probe_8g/`

The corresponding completed platform jobs are `t-20260803203803-d6vhk`
(four GPUs) and `t-20260803205105-np26w` (eight GPUs).

Two follow-up eight-GPU probes isolated loader concurrency and FSDP grouping.
With global batch 16 unchanged, workers 16 and `fsdp_devices=1` measured
0.5136364 seconds per step and 31.1504 samples/s, a 0.98% step-time reduction
relative to the workers-8 eight-GPU baseline. Workers 16 with
`fsdp_devices=2` measured 0.9124909 seconds per step and 17.5344 samples/s,
which is 75.9% slower than the pure-data-parallel baseline. Stronger grouping
was dominated and canceled before execution. The selected policy is therefore
workers 16 with `fsdp_devices=1` for a single critical-path arm when eight
colocated GPUs are otherwise idle; parallel experiment arms stay at four GPUs
to maximize aggregate throughput. MT3 learned-policy training now exposes this
eight-GPU GF1 candidate before its four-GPU GF1 and East fallbacks. The jobs are
`t-20260803220636-dq6pn` and `t-20260803220639-wtnf2`; canonical records are
`logs/scaling/pi05_mt1_b16_8g_opt_w16-fsdp1.json` and
`logs/scaling/pi05_mt1_b16_8g_opt_w16-fsdp2.json`.

Both follow-up jobs required approximately ten minutes before the first
training step, while the measured 300-step loop itself took roughly four
minutes. Frame-index construction accounted for only about 38 seconds; model
restore, state initialization, and JAX compilation dominate short-run latency.
The next safe optimization target is a persistent cache keyed by compatible
mesh and configuration, not a change to the frozen global batch or update
count. Formal MT1/MT3 launches already point at the shared vePFS cache. A mixed
8/4-GPU routing regression verifies that MT3 uses all eight GF1 cards when
available and falls back to four cards when that is the only immediately
runnable allocation. An MT3-specific warm compile remains artifact-gated on
tracker selection. The scheduler/router and MT1--MT6 regression suite passes
116/116 tests.

The final MT1 replication topology now has a verified North overflow path for
seed 1002. After the local step-49,999 checkpoint passes its root, params,
train-state, normalization, and audit gates, a zero-GPU task transfers only
parameters and assets to North and verifies every file by SHA-256 before an
atomic staging marker is published. The North parent uses four H20s and may be
joined by a two-H20 attach group on the same North scheduler files. If that
parent completes remotely, a materializer copies its 24 cells back, reruns the
fixed-scene verifier and summary, and only then publishes the shared completion
marker. The earlier seed-1000 transfer established the same path at roughly
12 GB and 37 minutes; optimizer state is deliberately excluded. This lets seed
1001 occupy GF1 immediately while seed 1002 gains North capacity after staging,
without weakening checkpoint or scene-manifest acceptance.
The staging wrapper always publishes a distinct decision marker. Transfer
success enables the North candidate; transfer failure records its return code,
removes any stale success marker, and releases the unchanged GF1/East
fallbacks. Executable success/failure tests verify that this optional overflow
path cannot block the seed-1002 claim-bearing evaluation.
An explicit cross-filesystem audit then compared the six provenance-bearing
evaluator sources, transition pairs, task map, and frozen scene manifest. All
nine SHA-256 values match between the current shared worktree and North vePFS,
so the overflow branch does not run stale evaluator code or protocol data.
The GF1 replication helpers were also assigned to the half-node opposite their
matching parent. Seed 1001 therefore uses parent GPUs 0--3 and helper GPUs
4--7 while seed 1002 stages; the reciprocal assignment remains available as a
fallback. Tests require disjoint index sets with complete 0--7 coverage, which
prevents a nominal helper from contending with its parent and leaving half of
GF1 idle.
An allocator-level simulation executes the same reservations, observes GF1 at
zero free cards, and then ranks the primary-identity North candidate first for
seed 1002 under the live quota model. The complete suite passes 116/116 tests.

The final seed-1001 and seed-1002 parent evaluations were also tightened to
require parameter metadata, atomic root `_CHECKPOINT_METADATA`, train-state
metadata, and checkpoint-local normalization statistics. Their previous
parameter-metadata-only static dependency could become visible during the final
save before the full checkpoint and root commit were complete. The full MT
scheduler/router regression suite now includes an explicit parent-evaluation
artifact assertion and passes 110/110 tests.

A scheduler-native checkpoint auditor now validates the seed-1001/1002 MT1
replications at every 5,000-step save and step 49,999. It requires nonempty
root, parameter, train-state, and normalization artifacts, parses the
normalization JSON, records SHA-256 values and total checkpoint bytes, and
publishes a separate atomic audit marker. Final parent evaluations consume the
step-49,999 marker in addition to the four checkpoint files. At 13:49 UTC the
auditor backfilled 14 accepted checkpoints through step 35,000, spanning
32,299,897,370--32,387,901,113 bytes; later saves are audited on the first poll
after root commit.

The pre-launch fan-out audit also found that the final replication helpers used
worker-index bases 46,000--47,600 while the attach wrapper adds port base
22,200. This would request ports above 65,535 and fail every final helper despite
the earlier seed-1000 helpers succeeding with smaller indices. The two final
seed namespaces are now disjoint at 20,000--22,400 and 24,000--26,400, yielding
a maximum requested port near 49,020. The existing 110/110 regression suite now
asserts worker-index uniqueness and the TCP port bound.

The new startup hard validator immediately exposed a second latent issue in the
gated MT3 attach branch: later intervention tasks exceeded the port range, and
the platform and local helpers for one evaluation shared a worker namespace.
MT3 evaluations have independent task schedulers, so they now safely reuse base
28,000 for platform helpers and base 30,000 for local helpers. This keeps the
two helper classes distinct within each evaluation and caps requested ports at
52,620. The queue rejects any generated MT attach candidate whose computed
port reaches 65,536. The scheduler supervisor resumed normal polling at
13:57:11 UTC; both GF1 trainers remained alive throughout validation restarts.

The resource-aware dispatcher now uses the submission router's complete live
score for its actual candidate order instead of merely recording the router
output after a fixed-order choice. Immediate capacity, per-GPU-count preference,
and checkpoint/data locality therefore affect the real launch. This makes
1/2-GPU jobs local-first when placement permits and preserves the declared
4/8-GPU order for matched locality. When every eligible target must wait, North is now a
persistent queue sink: queued attempts reserve a submitted-job slot but no
physical GPU, honor independent primary/backup job limits, and bypass the
short timeout used to reclaim opportunistic Shanghai queue attempts. The
updated scheduler was loaded by a supervisor restart at 13:40:55 UTC without
interrupting either active GF1 trainer.

## 22. MT1 step-40,000 audited recovery milestone

At 14:16 UTC, both frozen MT1 replications atomically committed their
step-40,000 checkpoints. Seed 1001 finalized in 15.67 seconds and seed 1002 in
14.84 seconds. Each checkpoint contains nonempty root, parameter, train-state,
and checkpoint-local `robotwin2.0_absolute_meanstd` normalization metadata.
The scheduler-native auditor accepted seed 1002 at 14:16:42 UTC and seed 1001
at 14:17:58 UTC, recording 32,318,638,496 and 32,421,374,866 bytes,
respectively, together with SHA-256 values for all four required artifacts.
Both trainers continued past step 40,100 with finite losses and no health
event.

These are resumable engineering artifacts at 80% of the frozen training
budget. They are not final checkpoints, closed-loop evaluations, or evidence
of control utility. The step-49,999 checkpoints, two frozen 24-cell evaluations,
and paired three-seed hierarchical analysis remain unfinished.

Canonical records:

- `logs/checkpoint_audits/pi05_mt1_oracle_seed1001_step40000.json`
- `logs/checkpoint_audits/pi05_mt1_oracle_seed1002_step40000.json`
- `logs/resource_markers/pi05_mt1_oracle_seed1001_step40000_checkpoint_audit.ok`
- `logs/resource_markers/pi05_mt1_oracle_seed1002_step40000_checkpoint_audit.ok`

## 23. MT1 seed-1001/1002 final training completion and evaluation launch

Both frozen oracle-transition replications completed on GF1 without a launcher
error. Seed 1001 ran from 05:37:42 to 16:25:01 UTC and seed 1002 from 05:37:42
to 16:25:00 UTC; both status records report `FINISHED rc=0`. At 16:24:59 UTC,
the scheduler-native auditor accepted both atomically committed step-49,999
checkpoints. The seed-1001 checkpoint contains 32,312,304,731 bytes and the
seed-1002 checkpoint 32,319,381,873 bytes. Each has nonempty root checkpoint,
parameter, training-state, and checkpoint-local mean/std normalization
metadata, with per-artifact SHA-256 values recorded in the audit JSON. No
traceback, non-finite loss, or out-of-memory event was found in either final
training record.

This closes the replication training and artifact-audit prerequisite only. It
does not establish representation quality, closed-loop utility, or use of
transition content.

The seed-1001 formal evaluation launched on GF1 at 16:26:26 UTC under a
manifest that freezes 24 cells, 50 episodes per cell, absolute actions,
mean/std normalization, oracle trajectories, and scene-manifest SHA-256
`08ed8eb7fa7e166e470dff99071639fec6e33bbd55104fe51be749418b820d17`.
The seed-1002 parameter-and-asset staging task completed at 16:55:58 UTC after
1,312 seconds, verified 47/47 files, and atomically published an 11.59-GB North
checkpoint using the TOS transport. Its formal North evaluation launched at
16:56:18 UTC with the same frozen scientific protocol and provenance hashes.

At the 17:19 UTC review cutoff, neither evaluation had a final report or
completion marker. Seed 1001 had 7/24 cells complete, 10 in progress, and zero
failed; seed 1002 had 14 cells in progress and zero failed. These partial task
rollouts are execution telemetry, not admissible paper evidence. They must not
be pooled into a macro result or used to make a control-utility claim. The two
accepted 24-cell summaries and the predeclared paired three-seed analysis remain
unfinished.

Canonical records:

- `logs/checkpoint_audits/pi05_mt1_oracle_seed1001_step49999.json`
- `logs/checkpoint_audits/pi05_mt1_oracle_seed1002_step49999.json`
- `logs/resource_markers/pi05_mt1_oracle_seed1001_step49999_checkpoint_audit.ok`
- `logs/resource_markers/pi05_mt1_oracle_seed1002_step49999_checkpoint_audit.ok`
- `logs/local_train/gf1_pi05_mt1_oracle_seed1001_4g/status`
- `logs/local_train/gf1_pi05_mt1_oracle_seed1002_4g/status`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1001_train_launch.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1002_train_launch.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1001_correct_eval_launch.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_mt1_oracle_seed1002_correct_eval_launch.json`
- `logs/resource_markers/pi05_mt1_seed1002_north_eval_checkpoint.ok`
- `logs/resource_markers/pi05_mt1_seed1002_north_stage_decided.ok`

## 24. MT1 three-seed closed-loop verdict and conditional closure

The seed-1001 and seed-1002 formal evaluations completed with 24/24 accepted
cells, 1,200 episodes, and zero protocol errors each. Both checkpoints achieve
81.00% macro and micro success. Together with seed 1000 and the accepted A0
matrix, the predeclared hierarchical bootstrap resamples training seeds and
then paired episodes within each task over 3,600 paired episodes.

The oracle-transition arm's macro difference from A0 is +0.53 percentage
points, with 95% CI [-2.06, +3.00]. Per-training-seed differences are +2.25,
+1.00, and -1.67 points for seeds 1000, 1001, and 1002. The predeclared
multistage task differences are positive for Handover (+1.67), Stack-3 (+3.17),
and Stack-2 (+2.17 points). Those gains coexist with regressions on Hammer
(-1.50), Ranking RGB (-0.83), and Ranking size (-1.50 points). The confidence
interval includes zero, and seed 1002 reverses the pooled direction; MT1
therefore fails its frozen gate. MT3--MT6 were disabled without launch, and the
paper retains the bounded negative-integration conclusion rather than claiming
milestone-transition routing as a method contribution.

The scheduler graph subsequently reconciled to 252 completed, 97
scientifically disabled, zero pending, and zero running tasks. The two historic
North materialization placeholders were closed from durable local report
markers; no additional evaluation was submitted.

Canonical records:

- `lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_three_seed.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1001_correct.json`
- `lmvla/lmwm/docs/pi05_mt1_oracle_seed1002_correct.json`
- `logs/resource_markers/pi05_mt1_oracle_seed1001_correct.ok`
- `logs/resource_markers/pi05_mt1_oracle_seed1002_correct.ok`
- `logs/analysis/pi05_mt1_three_seed_analysis/status`

## 25. Archived TODO state before the pi0.5-preserving reset

The active TODO was frozen at 2026-08-03 20:29 UTC after the MT1 verdict. It
contained no unfinished training or evaluation. Its scientific gate and
reporting boundary are preserved here so that `PAPER_TODO.md` can describe the
new experiment rather than retain completed history.

- Both MT1 replication reports passed the frozen 24-cell, 1,200-episode audit.
  Seeds 1001 and 1002 each reached 81.00% macro success.
- Across seeds 1000--1002, oracle transition conditioning changed A0 by +0.53
  points (95% CI [-2.06, +3.00]); seed effects were +2.25, +1.00, and -1.67.
- Task effects were -1.50 on Hammer, -0.83 on Ranking RGB, -1.50 on Ranking
  size, +1.67 on Handover, +2.17 on Stack-2, and +3.17 on Stack-3.
- MT1 failed because its interval included zero. MT3--MT6 and the conditional
  LM-pretraining branch were closed without launch.
- The final graph contained 252 completed and 97 scientifically disabled
  tasks, with zero pending and zero running.
- The paper retained a bounded negative-integration conclusion. It did not
  claim that milestone-transition routing was effective or that the three
  positive multistage task means generalized beyond this post-hoc pattern.
- The A0/A2/A3 conclusion remained unchanged: both tested one-token milestone
  interfaces regressed under the corrected three-seed pi0.5 recipe.
- No claim-bearing figure was unfinished. Completed figure constraints were a
  white canvas, final-size 6--8 pt sans-serif text, at least 0.5-pt strokes,
  sentence-case labels, colourblind-safe encoding with a second visual cue,
  and captions defining sample size and interval type.

The next active plan was opened only after changing the scientific question to
pi0.5-preserving, action-conditioned predictive transfer. Its rationale is in
`ANALYSIS_pi05_preserving_wm_integration_2026-08-04.md`.

## 26. Predictive-adapter frozen prerequisites

The new pi0.5-preserving branch asks whether short-horizon, action-conditioned
prediction can be transferred through a gradient-insulated residual adapter.
This is a new hypothesis and does not change the completed MINT-VLA or
oracle-transition conclusions. The records below establish implementation and
protocol validity only; they are not prediction-quality or control-utility
results.

The P0 implementation audit passes. All 51 leaves inherited from the official
`pi05_base` checkpoint are present, with no missing, modified, or unexpected
inherited leaves after the recorded bfloat16 casts. The cast base tree and the
checkpoint's inherited tree share SHA-256
`982d53a37ab7af6fc9be7e40ca588c20d41199ea051bd2db38d00497bbd2b567`.
The adapter adds 15 leaves. Its final routing kernel and bias are exactly zero,
and the gradient-route test checks that predictive loss updates the adapter
without entering the visual backbone. These checks establish initialization
identity and route isolation, not policy performance.

The audit JSON's `gradient_route_test.test` field retains the obsolete name
`test_predictive_adapter_stops_visual_gradients_and_updates_adapter`. The
collected replacement is
`test_predictive_loss_detaches_visual_tokens_but_updates_adapter`. A focused
review run passed that test together with the exact-zero route,
action-conditioning/control, and adapter-only freeze tests (4/4). The stale
label is a provenance defect in the audit metadata, not a missing gradient
test.

The frozen +1.0-s artifact covers all 27,500 episodes and 6,075,103 frames in
the accepted official-prompt dataset. It contains 4,700,103 valid pairs after
excluding 1,375,000 tail frames rather than clamping their targets. The
episode-level split assigns 24,763 episodes and 4,234,163 pairs to training and
2,737 episodes and 465,940 pairs to held-out evaluation, with no episode
leakage. The preregistered evaluation sample contains 8,192 pairs spanning all
2,737 held-out episodes and 2,621 task identifiers. P0 passes only if normal
action conditioning beats both shuffled and masked actions in paired
episode-cluster bootstrap intervals on mean patch cosine similarity. Even a
pass would be representation evidence only and would merely authorize P1.

The P1 baseline-reuse audit rejects the accepted historical A0 as an exact
control for the new branch. Its normalization hash matches, and current dry-run
configurations differ only at the eight declared candidate-only paths, but the
historical `config.py` and `pi0.py` hashes differ from the current sources and
the historical launch lacks the required base-metadata hash. If P0 passes, P1
must therefore train a current-source A0 alongside the candidate before any
closed-loop comparison.

Canonical records:

- `lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p0_implementation_audit.json`
- `lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/audit.json`
- `lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/heldout_eval.npz`
- `lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p1_baseline_audit.json`
- `lmvla/paper_iclr_lmvla/ANALYSIS_pi05_preserving_wm_integration_2026-08-04.md`

## 27. Predictive and recurrence branch protocol audits

Several protocol and evidence-availability audits completed after the P0
prerequisites. They constrain later experiments but do not provide a completed
prediction or control result. At this review cutoff, P0 remained in training;
its steps 5,000, 10,000, and 15,000 checkpoints were resumable intermediate
artifacts and had not entered the preregistered held-out evaluation.

The predictive-adapter P2 protocol freezes candidate seeds 1001 and 1002, the
same 24-cell scene manifest used by P1, a 20,000-draw hierarchical bootstrap,
and matched inference and training-memory measurements. It pins 12 execution,
evaluation, analysis, and scene-manifest files by SHA-256. Every P2 task remains
conditional on an accepted P1 gate. The matched `fsdp_devices=1` versus
`fsdp_devices=2` scaling result was already archived in Section 21; it is an
engineering throughput decision, not evidence of adapter quality.

The historical RoboTwin rollout-artifact audit found 1,245/1,245 readable
summaries covering 61,343 outcome-labelled episodes: 48,245 successes and
13,098 failures. Successful episodes averaged 332.34 steps and failed episodes
943.04 steps. The 9,948 files contain no frame observations, actions, states,
or trajectory arrays. These artifacts therefore support outcome and duration
analysis only; they cannot support post-hoc CRAVE progress, recurrence density,
stall lead time, or regression detection. A new frozen visual-rollout
collection is required. This is an evidence-availability finding, not a null
result for CRAVE or a control-utility result.

The pi0.5 semantic-interface audit found that the reviewed LeRobot policy and
public RoboTwin checkpoint accept visual, language, and state inputs and expose
only `ACTION` outputs. No semantic API or semantic-subtask output head appears
in the audited source contract. Generic stage or milestone embeddings are
therefore experimental conditioning paths, not a native pi0.5 semantic
prediction channel. This audit requires the privileged semantic-prompt upper
bound to precede any learned predictor; it does not test whether semantic
subtasks improve control.

The R2 adaptive-execution protocol pins 35 evaluator, analysis, encoder,
configuration, launch, and transfer sources plus the three public checkpoint
artifacts. It compares a frozen fixed-four policy with a causal 1/2/4/8-step
schedule on 24 cells and 240 episodes. The R3 semantic screen pins 25 sources
and the same three public artifacts, and compares five same-scene prompt
conditions. These freezes establish reproducible gates only. The canonical
scheduler contains 53 zero-attempt pending tasks across P1, P2, R0, R1, R2,
and R3;
none is a completed experiment or evaluation.

Canonical records:

- `lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json`
- `lmvla/lmwm/docs/robotwin_rollout_artifact_audit_2026-08-04.json`
- `lmvla/lmwm/docs/ROBOTWIN_ROLLOUT_ARTIFACT_AUDIT_2026-08-04.md`
- `lmvla/lmwm/docs/AUDIT_pi05_semantic_subtask_interface_2026-08-04.json`
- `lmvla/lmwm/docs/AUDIT_pi05_semantic_subtask_interface_2026-08-04.md`
- `lmvla/paper_iclr_lmvla/manifests/pi05_r2_adaptive_execution_protocol_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_r3_semantic_screen_protocol_v1.json`
- `logs/resource_scheduler_snapshot.json`

## 28. Predictive-adapter P0 verdict and R2 readout stop

The frozen predictive-adapter P0 gate completed on 2026-08-04. The final
step-19,999 checkpoint passed the parameter-isolation audit: all 51 inherited
parameter leaves are unchanged, no unexpected leaves were introduced, and the
policy route output kernel and bias remain exactly zero initialized. The
checkpoint metadata SHA-256 is
`1faabd944f840e4a10d4501f53bc53bd30a9cdedd63a27e60a1122d1c8012e44`;
the frozen normalization SHA-256 is
`8e4894bc762100c23cd435caff20653ccb39a9614ce210971dcf4018ef0a9f09`.

The preregistered held-out panel contains 8,192 pairs spanning all 2,737 heldout
episodes. Mean patch cosine similarity is 0.9960149 under normal actions,
0.9924060 under deterministic action-horizon reversal, and 0.9932628 under
masked actions. The episode-cluster bootstrap normal-minus-shuffled difference
is 0.0035883 with 95% CI [0.0034812, 0.0036983]; normal-minus-masked is
0.0027302 [0.0026551, 0.0028055]. P0 is therefore accepted. This establishes
action-conditioned latent prediction only and authorizes the matched P1
closed-loop experiment; it is not evidence that the adapter improves control.

The independent R2 causal recurrence readout failed its preregistered all-task
gate. Exact source reconstruction and progress/density MAE passed. Boundary AUC
passed on `beat_block_hammer` (0.5662), `blocks_ranking_rgb` (0.5531), and
`stack_blocks_two` (0.5907), but failed on `blocks_ranking_size` (0.4902),
`handover_block` (0.3984), and `stack_blocks_three` (0.5338). The downstream
adaptive-versus-fixed execution comparison is therefore stopped without a
closed-loop launch. This is a negative readout-sufficiency result, not an
adaptive-control result.

Canonical records:

- `logs/predictive/p0_eval/gate.json`
- `logs/predictive/p0_eval/p0_gate.accepted`
- `lmvla/lmwm/data/pi05_r2_causal_readout_v1/readout_manifest.json`
- `lmvla/lmwm/data/pi05_r2_causal_readout_v1/r2_readout.rejected`

## 29. CRAVE R0 offline control-semantics gate

The frozen R0 offline gate completed and was accepted on 2026-08-04. The
episode-disjoint panel contains 8,190 training rows and 1,620 heldout rows from
357 heldout episodes. The feature export restores the accepted P0 checkpoint
and records exact checkpoint, normalization, extractor, and model-source
hashes. The normal readout is fit once and reused unchanged for the shuffled
and masked action interventions; current-only and normalized-time controls use
independent matched-capacity readouts.

Normal action-conditioned features have lower aggregate loss than every
control for progress change, target recurrence density, and phase-boundary
crossing. Against current-only features, control-minus-normal loss is 0.00698
with episode-bootstrap 95% CI [0.00423, 0.00967] for progress, 0.01041
[0.00861, 0.01216] for density, and 0.01651 [0.00897, 0.02407] for boundary
crossing. Against shuffled actions the corresponding differences are 0.03409
[0.02906, 0.03930], 0.04076 [0.03784, 0.04373], and 0.05475
[0.04299, 0.06710]. All preregistered gate components pass.

This result establishes that action-conditioned pi0.5 predictive features are
sufficient for offline CRAVE progress, recurrence, and boundary readout. It
authorizes the R1 recurrence-aligned training branch. It does not establish
closed-loop policy improvement, failure prediction, reward, advantage, or
control value. At that gate cutoff, the outcome-linked rollout panel remained
pending; its later completed diagnostic is recorded in Section 30.

Canonical records:

- `logs/crave_r0/probe_features_v2/merged.json`
- `logs/crave_r0/probe_gate/report.json`
- `logs/crave_r0/probe_gate/r0_gate.accepted`

## 30. CRAVE R0 outcome-linked rollout diagnostics

The frozen R0 rollout panel completed after the offline readout gate. All 12
task--simulator-seed summaries and all 120 videos passed the artifact audit;
the panel contains 99 successful and 21 failed episodes. Episode-cluster
bootstrap intervals separate outcomes on four preregistered trajectory
statistics. Successful-minus-failed terminal progress is 0.11129 (95% CI
[0.04397, 0.18444]) and progress gain is 0.16166 [0.08935, 0.24513].
Failed-minus-successful stall fraction is 0.34827 [0.30419, 0.39356], and
regression fraction is 0.12523 [0.02950, 0.22978].

The same report rules out a stronger detector interpretation. Success
false-positive rates are 98.99% for low density, 89.90% for stall, and 38.38%
for regression. Ranking RGB contains no failures and is not estimable for
within-task outcome separation; the remaining tasks have only 1--9 failures
each. The completed panel therefore establishes outcome-linked separation in
the frozen rollout sample and motivates causal readout testing. It does not
establish reward, action advantage, failure value, online detection, or
closed-loop control utility.

Canonical records:

- `logs/crave_r0/rollouts/artifact_audit.json`
- `logs/crave_r0/rollout_analysis/report.json`
- `logs/crave_r0/rollout_analysis/report.md`
- `logs/resource_markers/pi05_crave_r0_rollout_analysis.ok`

## 31. R1 supervision-coverage reset

Before any R1 closed-loop outcome was available, a coverage audit found that
the first CRAVE-only and combined launches used the 8,190-row offline probe
sample as a sparse label lookup over 6,075,103 policy frames. Only 0.1348% of
sampled frames were eligible for recurrence supervision; with batch size 16,
the expected labelled count was 0.0216 per batch and the probability of any
labelled row was 2.14%. The resulting 100-step windows with zero recurrence
loss show that these launches do not test the recurrence objective. They were
superseded before policy evaluation and must not enter a scientific
comparison.

The corrected deterministic target builder materializes 359,823 valid
50-frame-horizon rows from all 1,200 frozen reference episodes across six
tasks. Coverage is 85.71% within those reference trajectories and 5.923% over
the unchanged full policy dataset, giving 0.9477 expected labelled rows per
batch and a 62.35% probability of recurrence supervision in a batch. The
dense-target SHA-256 is
`d795596174d5280f61e09274cd57c3930ba604fa77b130ea714ee97efbc23119`;
the regenerated protocol SHA-256 is
`f9a651df824faaa14caedda7b3517fcda18e0b29fb27c611d6af6e801c29e67d`.
This closes the input-validity defect only. Corrected training, interventions,
closed-loop evaluation, and the seed-1000 gate remain unfinished.

Canonical records:

- `lmvla/lmwm/docs/AUDIT_pi05_r1_dense_target_coverage_2026-08-04.md`
- `lmvla/lmwm/data/pi05_crave_r0_v1/r1_dense_targets.npz`
- `lmvla/lmwm/data/pi05_crave_r0_v1/r1_dense_targets_manifest.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json`

## 32. R3 runtime validity correction

The first five R3 screen launches failed before model loading because their
server wrapper omitted `kai0/src` from `PYTHONPATH`; they are engineering
failures rather than semantic results. A later one-scene semantic-next
preflight succeeded on 1/1 episode, but remains a smoke test. After source
repair, all five formal arms completed the frozen 24-cell, 240-episode screen.
Only those repaired launches enter the analysis; the failed wrappers and the
one-scene preflight remain engineering records. Section 33 contains the final
five-arm matrix, paired analysis, and rejected gate.

## 33. R3 privileged semantic-subtask gate verdict

The fifth R3 arm and the preregistered same-scene paired analysis completed on
2026-08-04. Each of the five conditions contains 24 validated cells, 40
episodes per task, and 240 episodes overall. Macro success is 65.42% for
generic stage, 80.83% for no subtask, 76.67% for semantic current, 77.08% for
semantic next, and 77.50% for within-task shuffled semantic.

Semantic next does not improve the primary comparison. Its difference from no
subtask is -3.75 percentage points with a paired episode-bootstrap 95% CI of
[-10.42, +2.92]. Its difference from shuffled semantics is -0.42 points
[-7.08, +6.25], and its difference from semantic current is +0.42 points
[-5.83, +6.67]. Semantic next exceeds generic stage by +11.67 points
[+4.58, +18.75], but the gate requires the privileged condition to clear the
meaningful controls rather than only the weakest prompt.

The task-level semantic-next differences from no subtask are -7.5 points on
Hammer, 0.0 on Ranking RGB, -5.0 on Ranking size, -25.0 on Handover, +12.5 on
Stack-3, and +2.5 on Stack-2. The 25-point Handover regression violates the
predeclared five-point task-safety threshold, so the positive Stack-3 effect
cannot be promoted through the macro average or a selected-task subgroup.

The R3 gate is rejected because both its primary confidence-interval criterion
and task-safety criterion fail. No semantic predictor is authorized. This is a
closed-loop negative result for these privileged prompt conditions on one
frozen public pi0.5 checkpoint and the frozen six-task screen; it does not show
that semantic decomposition is universally ineffective or provide a learned
semantic-prediction result.

Canonical records:

- `lmvla/lmwm/docs/pi05_r3_generic_stage_screen_v1.json`
- `lmvla/lmwm/docs/pi05_r3_no_subtask_screen_v1.json`
- `lmvla/lmwm/docs/pi05_r3_semantic_current_screen_v1.json`
- `lmvla/lmwm/docs/pi05_r3_semantic_next_screen_v1.json`
- `lmvla/lmwm/docs/pi05_r3_shuffled_semantic_screen_v1.json`
- `logs/r3_semantic_screen/report.json`
- `logs/r3_semantic_screen/r3_gate.rejected`
- `logs/resource_markers/pi05_r3_shuffled_semantic_screen_v1.ok`

## 34. P1 predictive-adapter seed-1000 gate verdict

The pi0.5-preserving predictive branch is a new method hypothesis, not a
reinterpretation of the completed observation-only MINT-VLA experiment.  It
keeps the official pi0.5 visual-language and action paths, stops the predictive
loss at inherited features, and routes an action-conditioned, zero-initialized
adapter only into the action expert.  P0 established action-conditioned latent
prediction and authorized P1; it did not establish control utility.

P1 trained a current-source A0 and predictive candidate from the same official
initialization at training seed 1000.  Both final step-49,999 checkpoints passed
the frozen source, dataset, normalization, optimizer-state, checkpoint, and
24-cell scene-manifest audits.  Every evaluation contains 1,200 paired
episodes.  Macro success is 69.00% for A0, 82.42% for the normal predictive
adapter, 78.50% with the predictive route zeroed, 81.17% with shuffled action
conditioning, and 78.00% with masked actions.

The normal adapter improves over A0 by 13.42 percentage points.  Its task-level
differences are +8.5 on Hammer, +13.5 on Ranking RGB, +15.5 on Ranking size,
+14.0 on Handover, +2.0 on Stack-2, and +27.0 on Stack-3; no task regresses.
The pooled paired normal-versus-A0 comparison has Holm-adjusted
`p=7.02e-16`.  The normal adapter also exceeds the three preregistered controls
in point estimate, so the directional seed-1000 gate is accepted and P2
replication is authorized.

The content-specific interpretation is weaker than the gate verdict.  Normal
exceeds shuffled actions by only 1.25 points, with exact McNemar `p=0.383` and
Holm-adjusted `p=1.0`; normal versus zero-gate is +3.92 points with
Holm-adjusted `p=0.0939`.  P1 therefore establishes a positive, directly
measured seed-1000 closed-loop screen.  It does not yet establish a replicated
utility effect or a strong causal claim about correct action content.  Those
claims remain conditional on P2.

The P1 recovery history is operational provenance rather than extra evidence.
After gf1 was permanently retired, exact source and optimizer states were
recovered through a checksum-frozen North stage.  Failed pre-launch,
container-runtime, package-lock, CUDA-architecture, and materialization
attempts produced no admissible rollout evidence.  Only the five complete
canonical reports enter the gate.

Canonical records:

- `lmvla/paper_iclr_lmvla/RESULTS_pi05_predictive_adapter_p1_seed1000_gate.json`
- `lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_a0.json`
- `lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_normal.json`
- `lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_zero_gate.json`
- `lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_shuffled.json`
- `lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_masked.json`
- `logs/predictive/p1_eval/p1_gate.accepted`

## 35. R1 recurrence-aligned extension and R4 completed prerequisites

R1 tests whether recurrence-derived CRAVE targets improve the new predictive
adapter.  This is an auxiliary extension of P1, not the parent predictive
method.  Sparse-label launches were invalidated before evaluation; only the
corrected 359,823-row dense-target protocol and its final step-49,999
checkpoints enter the result.

All four R1 seed-1000 evaluations contain 24 cells and 1,200 paired episodes.
CRAVE-only reaches 68.67%, predictive-plus-CRAVE combined reaches 62.92%, the
combined zero-route control reaches 62.17%, and combined shuffled-action
reaches 64.67%.  Combined is 6.08 points below A0 (paired bootstrap 95% CI
[-9.33,-2.75]), 5.75 points below CRAVE-only ([-9.00,-2.50]), and 19.50 points
below the parent predictive adapter ([-22.50,-16.50]).  Relative to A0, its
task effects are -1.5 on Hammer, +3.0 on Ranking RGB, -6.5 on Ranking size,
-6.5 on Handover, -11.0 on Stack-2, and -14.0 on Stack-3.  Four tasks violate
the five-point regression guard.  The preregistered R1 gate is therefore
rejected, and R1 seeds 1001/1002 are permanently disabled.  This result rejects
the recurrence-aligned combined extension; it does not negate the accepted P1
screen.

R4 completed its data and training prerequisites without establishing a policy
effect.  Its audited dataset contains 600 rollout episodes and 6,313 executable
three-camera query/action samples.  The ordinary, task-normalized
terminal-outcome-weighted, and outcome-free CRAVE-weighted arms share identical
ordered action chunks and differ only in their weighting rule.  Exact public
checkpoint loading, sample-weight preservation, sidecar alignment, and the
two-step runtime smoke passed.  All three formal training arms completed the
frozen step-5,000 checkpoint and passed the checkpoint-integrity audit.  These
records authorize the fixed-checkpoint three-arm closed-loop screen; training
losses and checkpoint existence alone are not control evidence.

Canonical records:

- `lmvla/paper_iclr_lmvla/RESULTS_pi05_r1_seed1000_gate.json`
- `lmvla/lmwm/docs/pi05_r1_seed1000_crave.json`
- `lmvla/lmwm/docs/pi05_r1_seed1000_combined.json`
- `lmvla/lmwm/docs/pi05_r1_seed1000_combined_zero_gate.json`
- `lmvla/lmwm/docs/pi05_r1_seed1000_combined_shuffled.json`
- `logs/r1/seed1000/r1_gate.rejected`
- `logs/r4/training_dataset_audit.json`
- `logs/r4/checkpoint_integrity_v1.json`
- `lmvla/paper_iclr_lmvla/manifests/pi05_r4_formal_eval_protocol_v1.json`
