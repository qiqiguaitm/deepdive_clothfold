# Paper Replan: Temporal Grounding of Predicted Future Representations

Date: 2026-08-07

Target: ICLR-style nine-page main paper plus references and appendix.

Status: narrative and evidence plan only. The current MINT-VLA manuscript
remains the frozen completed-evidence draft until the new gates resolve which
paper branch is admissible. Bracketed results below are placeholders, not
claims.

## 1. Reset the paper question

The old framing asks whether a new milestone module improves a VLA. That
framing is mismatched to the evidence: MINT predicts its target but regresses
in matched control, and the repaired predictive adapter does not replicate
against independently trained baselines.

The new paper asks a broader and testable question:

> When does a predicted future representation provide a usable constraint for
> fixed-horizon VLA action generation?

The proposed explanation is temporal grounding. The released RoboTwin LaWAM
contract has a 36-action model window and executes 36 actions per query; under
the current two-frame loader rule, its fixed future is the coincident endpoint.
Recurrence milestones represent later task stages at variable, usually
multi-chunk horizons. A distant subgoal may identify eventual progress without
specifying how much progress the current chunk should make. This explanation
is not yet a result; the new experiments must distinguish it from predictor
error, extra capacity, pretraining, gradient shaping, execution cadence, and
task heterogeneity. The original released training dataset and exact training
source are not local, so the new protocol must retain this provenance limit.

## 2. Systems and terminology

Use the following names consistently:

- **LaWAM system:** the released architecture, pretraining, fixed-near-future
  target, and action expert. A public system score does not isolate any one
  component.
- **Fixed-endpoint condition:** the LaWAM-style target sampled at `t+h`, where
  `H` is the valid action count and `h=H-1` is its last discrete offset.
- **LMWM:** the historical LaWAM testbed that replaces or augments the fixed
  target with a recurrence-defined milestone. Historical experiments are
  diagnostic unless they meet the new matched-seed protocol.
- **MINT-VLA:** the completed pi0.5 native-space, one-prefix-token
  instantiation. It provides a rigorous negative boundary and must not be
  conflated with the LaWAM architecture.
- **Temporally grounded milestone:** a recurrence target supplemented or
  modified by a prespecified temporal relation to the current action chunk.

Avoid using `world model` for a module that predicts one observation feature
without actions, transition rollout, or planning. Use `predicted future
representation` or `latent visual subgoal` where exact.

## 3. Evidence ledger before new experiments

### Established and admissible

1. The released RoboTwin LaWAM checkpoint has 36 valid actions at 30 Hz over
   1.2 s (`h=35`), and its evaluator executes 36 actions per query. Under the
   audited current two-frame loader rule, the public contract is endpoint-
   aligned; the exact original training sample stream is not independently
   reconstructed.
2. The historical local all6 LMWM matrix is different: 50 Hz over 1.0 s gives
   50 valid training actions (`h=49`), but its evaluator executes only 36
   actions before replanning. Its fixed target is not the endpoint of the
   executed prefix.
3. Recurrence milestones have variable horizons and no time-to-go. Their task
   means range from 53.64 to 130.44 frames; only 19.35% fall within the local
   50-action training chunk and 12.50% within the executed 36-action prefix.
4. MINT prediction is nontrivial: latent cosine is 0.8134 versus 0.7479 for
   persistence.
5. MINT control utility is rejected: -8.58 points versus matched no hint, with
   95% interval `[-11.47,-5.75]`, and all six tasks regress.
6. Correct-content and privileged controls do not establish action
   sufficiency.
7. A +11.61-point fixed-A0 adapter result becomes +1.94 points with interval
   `[-5.78,+12.75]` after independent matched A0 training.

### Descriptive or hypothesis-generating only

1. Historical LaWAM/LMWM comparisons suggest that fixed near-future and
   milestone targets behave differently across long-horizon and spatial
   tasks, but several comparisons lack matched training seeds.
2. `Future-off` outperforms active local-WM in one completed matrix, but it
   inherits LaWAM pretraining and is not a clean VLA baseline.
3. Duration correlates with selected milestone recovery over six post-hoc
   tasks, but duration, stage count, and horizon are confounded.
4. Timeout-heavy failures are compatible with an under-specified long-horizon
   condition, but no trajectory-level causal analysis establishes that link.
5. The historical local fixed target lies beyond the executed prefix. This is
   a measured protocol mismatch, not evidence that it caused the local-WM
   regression.

### Prohibited before new gates

- LaWAM works because its target is inside the action chunk.
- LMWM fails because its target lies outside the action chunk.
- Milestone prediction is generally ineffective.
- The LaWAM system score proves use of correct future content.
- A second architecture or benchmark inherits a result by compatibility.

## 4. Candidate thesis and paper branches

### Preferred branch: mechanism established

Use only if TG1--TG3 pass:

> Predicted future representations constrain VLA actions when their temporal
> meaning is matched to the action horizon. Fixed chunk-end targets are used by
> the action expert, whereas variable-horizon milestones are under-specified;
> explicit time-to-go or chunk clipping restores part of the lost utility.

### Valid fallback: controlled negative result

Use if TG2 shows fixed and raw targets both fail against future-off:

> Neither fixed-near-future nor recurrence-stage prediction improves matched
> downstream control despite nontrivial representation prediction. Public
> system performance cannot be attributed to active future content without
> clean component controls.

### Valid intermediate branch: system audit

Use if LaWAM gains utility but normal does not beat shuffled:

> The LaWAM training package can alter performance, but its predicted future
> content is not causally identified; matched seeds and content interventions
> separate package utility from mechanism.

Do not decide the title, abstract conclusion, or hero result before the branch
is selected by frozen gates.

## 5. Working titles

Preferred if mechanism passes:

1. **Temporal Grounding Makes Future Representations Actionable for VLAs**
2. **Aligning Predicted Future States with VLA Action Horizons**
3. **From Distant Milestones to Actionable Futures in Robot Policies**

Preferred for audit or negative branch:

1. **Do Predicted Future Representations Improve VLA Control?**
2. **Auditing Future-State Conditioning in Vision--Language--Action Models**
3. **Predictable Futures Do Not Ensure Action Utility in VLAs**

Do not lead with `MINT-VLA` unless a revised MINT arm passes a positive matched
gate. Retain MINT as a named controlled instantiation in Methods and prior
evidence.

## 6. Contributions, conditional on evidence

Contribution order should reflect the scientific question, not development
history:

1. **Temporal contract.** Formalize the relation between valid model action
   count `H`, executed actions per query `E`, their last offsets, future-target
   horizon `tau(t)-t`, target content, and the action expert's conditioning
   interface.
2. **Causal evaluation.** Introduce matched training seeds and fixed-checkpoint
   content, route, persistence, oracle, and time interventions that separate
   representation prediction, training-package utility, and content use.
3. **Empirical finding.** Fill after TG2/TG3: either temporal grounding repairs
   milestones, or active future prediction remains unsupported under matched
   controls.
4. **Evidence boundary.** Show how fixed-baseline replication and task-level
   averaging can misstate a method effect.

MINT target mining and native-space prediction may remain technical
contributions, but they should not be the first contribution unless they
produce a replicated control benefit.

## 7. Formal problem statement

Let the action expert predict a chunk with `H` valid actions

`A_t = (a_t, ..., a_{t+H-1})`.

and let `h_H=H-1` denote its last sampled discrete offset. Let the evaluator
execute `E <= H` actions before querying the policy again, with
`h_E=E-1`. The released RoboTwin checkpoint has `H=E=36`; the completed local
all6 matrix has `H=50` and `E=36`. Let a future-conditioning module produce
target representation

`z_t^* = E(o_{tau(t)})`

and prediction `z_hat_t`. Define normalized temporal displacement

`g_t^H = (tau(t)-t)/h_H` and `g_t^E = (tau(t)-t)/h_E`.

- Training-window endpoint: `g_t^H = 1` and target `z_{t+h_H}`.
- Executed-prefix endpoint: `g_t^E = 1` and target `z_{t+h_E}`.
- Execution-aligned condition: `H=E` and the two endpoints coincide.
- Multi-query milestone: `g_t^E > 1`.
- Unspecified milestone: the policy receives `z_hat_t` but neither temporal
  displacement.

The paper separates four questions:

1. **Predictability:** can the model estimate `z_t^*`?
2. **Package utility:** does training with the module improve matched control?
3. **Content use:** does correct `z_hat_t` beat shuffled or null content?
4. **Temporal use:** does correct `g_t` beat time-shuffled or constant timing?

No answer implies another.

## 8. Experimental narrative

### Experiment 1: establish the temporal contracts

Report `H`, `E`, both endpoints, and the milestone `g_t^H` and `g_t^E`
distributions without policy outcomes. This CPU-only descriptive result is
complete and belongs in the first panel of Figure 1.

### Experiment 2: test whether LaWAM uses future content

At one frozen checkpoint compare normal, shuffled, null, and persistence
representations on identical scenes. Use a privileged oracle chunk-end target
only when an exact same-scene expert trajectory exists; otherwise report it as
an offline probe. This experiment decides whether future content, rather than
route or capacity, constrains actions.

### Experiment 3: compare training targets under one architecture

Train future-off, fixed endpoint, and raw milestone at matched seeds with
`E=H` frozen before training. This is the primary policy table. It answers
whether active future prediction helps and whether target horizon changes the
effect without inheriting the historical 50-versus-36 contract mismatch.

### Experiment 4: intervene on temporal grounding

Only after a fixed-versus-raw difference, compare raw milestone with explicit
time-to-go and chunk-clipped milestone. Time-shuffle the accepted arm. This is
the decisive mechanism experiment.

### Experiment 5: locate the source of utility

Only after a positive upstream gate, compare clean VLA, future-off,
auxiliary-only, conditioning-only, and full fixed-endpoint training. This
separates pretraining, representation shaping, and inference content.

### Experiment 6: external validation

Replicate only the frozen winning contrast on complete LIBERO four-suite or a
prespecified second RoboTwin panel. Do not use a saturated suite or selected
positive tasks as the external claim.

## 9. Section-by-section outline

### Abstract: 170--190 words

1. Context: predicted future representations are increasingly supplied to VLA
   action policies.
2. Gap: benchmark improvements do not reveal whether correct future content is
   used or whether its time horizon matches the action chunk.
3. Design: introduce the temporal contract and matched content/time
   interventions across `[benchmarks]`, `[tasks]`, and `[training seeds]`.
4. Existing boundary: milestone prediction improves latent similarity but not
   current pi0.5 control.
5. New result placeholder: `[fixed/raw/grounded effects with intervals]`.
6. Significance: prediction metrics and system-level comparisons cannot replace
   horizon-matched causal controls.

### 1. Introduction: 1.2 pages

- Paragraph 1: future prediction is used to inform robot actions.
- Paragraph 2: a future representation has two semantics--what state and when.
  Existing evaluations often test the first while leaving the second implicit.
- Paragraph 3: LaWAM supplies a chunk-end future; recurrence milestones supply
  variable-stage futures. This contrast creates a testable temporal-grounding
  question.
- Paragraph 4: state the matched design and principal result after gates close.
- End with three concise contributions in the order in Section 6 above.

### 2. Related work: 0.7 pages

Organize by unresolved assumption, not by paper chronology:

1. Fixed-horizon future images/features and latent visual subgoals.
2. Long-horizon subgoals, milestones, and hierarchical control.
3. Interfaces that gate, align, or isolate future conditioning.
4. Evaluation work separating prediction quality from policy utility.

For each family, state whether target time is fixed, adaptive, implicit, or
planned. Do not criticize an external method for lacking an experiment unless
the original paper and released code have been verified.

### 3. Temporal contract: 0.8 pages

- Define `H`, `E`, `tau(t)`, both normalized displacements, target content,
  training window, and executed prefix.
- Show why a fixed endpoint is a boundary condition while a variable milestone
  is a high-level subgoal.
- State that a multi-chunk target can still help if timing or a hierarchical
  local target is supplied; do not present mismatch as intrinsic invalidity.
- Define the four evidence questions.

### 4. Systems and interventions: 1.3 pages

- LaWAM fixed-endpoint route.
- Recurrence milestone target and MINT/LMWM instantiations.
- Normal, shuffled, null, persistence, oracle, time-to-go, time-shuffled, and
  chunk-clipped conditions.
- Clearly separate fixed-checkpoint interventions from retrained arms.

### 5. Experimental design: 1.0 page

- Benchmarks and prespecified task strata.
- Matched initializations, training seeds, fixed steps, data and optimizer.
- Paired scenes and accepted-episode rules.
- Hierarchical uncertainty and Holm family.
- Primary gates, task-safety gate, and stop rules.

### 6. Results: 2.4 pages

Write claim-first in this order:

1. Targets differ in temporal relation to the action chunk.
2. `[LaWAM does/does not]` use correct chunk-end future content.
3. Matched training `[supports/rejects]` active fixed-endpoint utility.
4. Raw milestones `[differ/do not differ]` from fixed endpoint.
5. Temporal grounding `[repairs/does not repair]` the raw milestone.
6. Task-level heterogeneity and external replication bound the result.

Keep prior MINT, predictive-adapter, and outcome-weighting results to one
boundary paragraph plus an appendix table unless they directly support the new
causal chain.

### 7. Discussion and limitations: 0.8 pages

- Interpret the selected branch without restating every number.
- Separate target timing, target content, predictor quality, and training
  shaping.
- Explain why a distant milestone may require hierarchical conversion to a
  local endpoint.
- Limit scope by architecture, benchmark, task groups, simulation, and number
  of training seeds.
- State what the study does not establish about generative world models,
  planning, real robots, or other VLAs.

### 8. Conclusion: 0.25 pages

One result, one implication, one boundary. Do not end by marketing MINT.

## 10. Figures and tables

### Figure 1: temporal grounding, not method architecture

Four panels:

1. A model window with `H` actions, an executed prefix with `E` actions, and
   their last sampled indices.
2. The released LaWAM target at the coincident `H=E=36` endpoint and the
   historical local `H=50,E=36` distinction.
3. Recurrence milestone targets at variable `tau(t)`, with within- and
   multi-chunk examples.
4. The evidence ladder: prediction, package utility, content use, temporal use.

Before TG3, label temporal grounding as a hypothesis. After TG3, add only the
accepted effect and interval.

### Figure 2: target-horizon distributions

Show per-task distributions of `g_t^H` and `g_t^E`, not only means. Mark
`g=1`, report sample counts, and use task strata rather than outcome-selected
ordering.

### Figure 3: matched policy effects

Training seed is visible. Plot per-seed and per-task paired effects for
future-off, fixed, raw, and grounded arms with hierarchical intervals. Never
show only a macro bar.

### Figure 4: content and timing interventions

Show normal--shuffled, normal--null, oracle--normal, and correct-time--shuffled-
time contrasts. Distinguish fixed-checkpoint and retrained evidence visually.

### Table 1: protocol and component equality

Rows are arms; columns are initialization, target time, target content,
time-to-go, auxiliary gradient, inference route, parameters, seeds, and
episodes.

### Table 2: primary seed-by-task result

Include every task and training seed or use a compact main summary with the
complete matrix immediately adjacent in the appendix. Highlight regressions,
not only the best mean.

All figures should follow the recorded publication style: white background,
6--8 pt sans-serif text at final size, at least 0.5 pt strokes, colourblind-safe
colours plus shapes/line styles, no decorative gradients, and self-contained
captions defining `n`, seeds, resampling, intervals, and multiplicity control.

## 11. Result-to-claim decision table

| Evidence outcome | Allowed headline | Required manuscript branch |
|---|---|---|
| TG1 content gate and TG2 fixed utility pass; TG3 repair passes | Temporal grounding makes future conditions actionable | Preferred mechanism branch |
| TG2 fixed beats raw, TG3 fails | Fixed endpoint differs from milestones; cause unresolved | Bounded comparison, no mechanism claim |
| Fixed improves package utility but TG1 content fails | Training package helps; correct future content not identified | System-audit branch |
| Fixed and raw both fail future-off | Active future objectives do not improve matched downstream control | Controlled negative branch |
| Fixed and raw unresolved | No target-horizon effect established | Evaluation paper or stop |
| Positive macro with unsafe task regressions | Heterogeneous effect only | No general improvement claim |

## 12. What to retain, compress, or remove from the current draft

### Retain

- Claim--evidence separation.
- Matched training seeds and paired scene protocol.
- MINT prediction-versus-control boundary.
- P2-to-P3 fixed-baseline reversal.
- Correct-content, oracle, semantic, and spatial controls.

### Compress to appendix

- Operational scheduler and recovery history.
- Full R1/R4 narratives.
- Every legacy MINT variant.
- Efficiency details unless the selected method branch is positive.
- Historical LaWAM checkpoint screens that lack matched training seeds.

### Remove from the main narrative

- MINT-first title and abstract framing.
- Any implication that recurrence mining itself is the main contribution.
- Chronological accounts of failed variants.
- Claims that task duration, headroom, or milestone count already predicts
  utility.
- Any external-method critique not supported by a faithful reproduction or the
  original paper's own evidence.

## 13. Writing order after evidence closes

1. Freeze the accepted result-to-claim branch.
2. Build Tables 1--2 directly from canonical JSON.
3. Build Figures 1--4 and run visual preflight.
4. Write Results claim-first from the displays.
5. Write Temporal Contract, Systems, and Experimental Design.
6. Rewrite Introduction and Related Work around the resolved gap.
7. Write Discussion with mechanism boundary and counterexamples.
8. Write Abstract and title last.
9. Run claim-to-artifact audit, page-limit build, citation audit, and anonymous
   artifact-link check.

## 14. Current handoff

The CPU temporal-contract and milestone audit is complete. Its canonical
summary is `RESULTS_temporal_grounding_local_audit_v1.json`; the full
reproducible command is
`python lmvla/lmwm/scripts/audit_temporal_grounding_contract.py`. The active
`PAPER_TODO.md` now contains only GPU training and closed-loop evaluation.

Each GPU item still requires an immutable admission bundle containing its
intervention hook, source and checkpoint hashes, paired scene manifest, report
schema, analysis command, resource estimate, and stop rule. Preparing and
admitting that bundle does not authorize manual launch. Keep the current
claim-bearing manuscript unchanged until the GPU gates select an admissible
paper branch.
