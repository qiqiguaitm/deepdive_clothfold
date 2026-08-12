# Temporal-Grounding GPU Evidence TODO

Updated: 2026-08-12 12:40 UTC

This file contains only unfinished training/evaluation evidence and current
scientific gates. Completed evidence, rejected protocols, and superseded
execution history are in `PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`, Sections
41--48.

The resource-aware scheduler is the sole execution owner. A checkbox records
scientific completion; it does not authorize manual launch, stop, restart,
reprioritization, or replacement. Mutable execution state is authoritative only
in `logs/resource_scheduler_snapshot.{md,json}` and
`logs/resource_scheduler_state.json`.

## 1. Current evidence boundary

The paper asks:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

Three boundaries currently determine the answer:

- The local contract audit establishes endpoint alignment for the released
  LaWAM path and documents that historical raw milestones are usually
  multi-chunk targets without time-to-go. This is timing evidence, not control
  utility.
- Both TG2 training matrices are closed without policy evaluation. The original
  matrix failed matched rank-order integrity across arms; TG2R repaired that
  mismatch but then failed the preregistered requirement that training seeds
  induce distinct rank data orders. These protocol rejections say nothing about
  which target improves control.
- The complete TG1B diagnostic did not detect local-WM-specific cadence
  sensitivity: difference-in-differences +1.42 percentage points, hierarchical
  95% CI [-3.00, +5.92]. This rejects the cadence gate for the audited
  same-training-seed checkpoints, but it neither establishes correct-content
  use nor proves LMWM generally ineffective.

TG1A establishes released-checkpoint content use under the frozen retry500
panel. All three prespecified contrasts pass their hierarchical confidence,
Holm-adjusted significance, and task-safety gates. This is fixed-checkpoint
control evidence; it does not yet decompose whether the utility comes from WM
pretraining, downstream auxiliary shaping, or inference-time conditioning.

## 2. Active TG1A content-use panel

Normal, null, and persistence are complete at 24/24 task-by-evaluation-seed
cells and 1,200 fixed-scene episodes per condition. Their audited rates and
pre-analysis are archived in Section 48. They remain inputs to the frozen Holm
family and are not standalone paper claims before the four-condition analysis.

- [x] **TG1A-E4 [COMPLETE: parent `t-20260812040634-7kwps`, tail
  `t-20260812120538-n7jgs`]** The frozen within-task, different-episode shuffled
  mapping contains exactly 24 task-by-evaluation-seed summaries and 1,200
  accepted episodes. The tail filled only the four missing
  `stack_blocks_three` cells. All fixed-scene hashes and 50-episode cell counts
  verify.
- [x] **TG1A-A1 [COMPLETE; ALL THREE GATES ACCEPTED]** The frozen analysis
  consumed exactly 1,200 paired outcomes per condition. Success was 94.00% for
  normal and 40.33% for shuffled. `normal - shuffled` is +53.67 points with
  hierarchical 95% CI [+36.08, +68.58] and Holm-adjusted
  `p=6.75e-180`. All six task effects and four evaluation-seed effects are
  positive. `normal - null` is +58.83 points, CI [+36.83, +79.50], and
  `normal - persistence` is +69.58 points, CI [+48.00, +87.25]; both are also
  accepted. The canonical result and marker are present.

### TG1A acceptance gates

- **Correct-content use:** `normal - shuffled` requires a hierarchical paired
  95% CI lower bound above zero and Holm-adjusted paired `p<0.05`.
- **Route necessity:** apply the same two-part gate independently to
  `normal - null`.
- **Endpoint content beyond persistence:** apply it independently to
  `normal - persistence`.
- Report every task effect and every evaluation-seed effect. A macro average
  cannot hide a negative task.
- Passing null or persistence alone does not establish correct-content use.
  Only the prespecified shuffled contrast identifies whether the particular
  predicted content matters.

## 3. Conditional downstream gate

TG1A satisfies the prespecified correct-content-use and task-safety prerequisite.
No TG4 GPU task is admitted until the source-decomposition manifest and exact
checkpoint compatibility audit are frozen.

- [ ] **TG4 [ELIGIBLE; MANIFEST FREEZE IN PROGRESS]** Freeze a source
  decomposition before launch: compatible clean base, future-off,
  auxiliary-only, conditioning-only, parameter-matched null, and accepted full
  checkpoint. Attribute pretraining, downstream shaping, and inference content
  only from their prespecified contrasts. Every claimed contrast needs a
  positive hierarchical lower bound, Holm-adjusted `p<0.05`, the fixed
  -5-point task-safety gate, and a fixed-checkpoint content intervention.

TG3 temporal repair and TG5 external replication are closed because TG2R
produced no admissible control contrast. Do not reopen them from checkpoint
existence, training losses, or the failed integrity matrix. If TG1A rejects
correct-content use, close TG4 and end temporal-grounding GPU work under the
current plan.

## 4. Stop and reporting rules

- Do not reopen MINT-VLA, predictive-adapter P0--P5, R0--R4, outcome weighting,
  oracle-transition, or failed helper jobs to search for a positive result.
- Do not evaluate either rejected TG2 matrix or substitute one of its
  checkpoints into another protocol.
- Partial rollouts, smoke tests, training losses, representation metrics,
  checkpoint existence, or unmatched seeds cannot pass a utility gate.
- Representation prediction does not establish control utility. Cadence
  sensitivity does not establish correct-content use. A public system score
  does not identify its causal component.
- Preserve the fixed -5-point task-safety threshold and every task-level
  regression. Do not promote a macro-only improvement.
- Do not tune target horizon, task groups, training seeds, checkpoint step,
  intervention mapping, retry recipe, or loss weight against outcomes.
- Task_N remains outside this paper plan by operator instruction.

## 5. Canonical live sources

- Active scheduler summary: `logs/resource_scheduler_snapshot.md`
- Canonical mutable state: `logs/resource_scheduler_snapshot.json` and
  `logs/resource_scheduler_state.json`
- TG1 retry500 amendment:
  `lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json`
- TG1A frozen analyzer:
  `lmvla/lmwm/scripts/analyze_temporal_grounding_tg1a.py`
- Completed TG1B result:
  `lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg1b.json`
- TG2R integrity rejection:
  `lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg2r_integrity.json`
- Completed evidence and protocol history:
  `lmvla/paper_iclr_lmvla/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`
