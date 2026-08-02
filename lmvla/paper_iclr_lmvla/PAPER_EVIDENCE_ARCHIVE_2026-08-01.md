# MINT-VLA Completed Evidence Archive

Updated: 2026-08-01 16:19 UTC

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
| Combo | 2 | 89.04 |
| Gradient isolation | 2 | 89.04 |
| Residual | 2 | 87.50 |
| Local-WM | 2 | 86.67 |

The current matrix does not show a downstream WM-objective gain over matched
Future-off initialization. Absolute, residual, combo, local-WM, and gradient
isolation all have two completed training seeds; Future-off has three. The
authoritative matrix is
`logs/eval_reports/robotwin_all6_v2_training_seed_matrix.json`.

This aggregate result does not imply that LMWM has no effect. Relative to the
active `local-WM` arm, absolute and combo improve by `+2.54 pp` and `+2.38 pp`
respectively. At task level, combo improves stack-three by `+7.00 pp`, ranking
size by `+4.25 pp`, ranking RGB by `+2.50 pp`, and handover by `+1.00 pp`
relative to local-WM. Only stack-three also exceeds Future-off (`+7.00 pp`).
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

The existing three-task content interventions also reject a duration-only
explanation. Absolute hints are causally null; correct residual-only hints are
worse than shuffled and cross-task controls; isolation reverses that harmful
pattern. The current diagnosis is a mixture of weak content use and
route-dependent harmful conditioning. A six-task intervention evaluation is
needed to estimate a duration-by-hint-condition interaction.

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
