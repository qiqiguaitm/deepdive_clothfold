# Preserving pi0.5 Pretraining While Adding Predictive World Knowledge

Updated: 2026-08-04 UTC

## 1. Decision

The next positive-method attempt should preserve the complete official pi0.5
initialization and add a training-time predictive adapter. It should not
replace the visual encoder, restart from PaliGemma, or inject another
observation-only milestone token.

The selected hypothesis is:

> An action-conditioned predictive adapter can transfer short-horizon latent
> dynamics into the pi0.5 action expert while leaving the pretrained visual-
> language path unchanged by the world-model objective.

This is a new hypothesis, not a repair that changes the interpretation of the
completed MINT-VLA experiment. The current paper remains a bounded negative
integration study until the new route independently passes its gates.

## 2. What must be preserved

"Using pi0.5 pretraining" means preserving all of the following, rather than
only loading the PaliGemma component:

1. Initialize the SigLIP image encoder, PaliGemma language model, action
   expert, state/action projections, and flow-matching head from the same
   official `pi05_base` checkpoint used by accepted A0.
2. Keep the original image, language, state, and action-token paths present and
   shape-compatible. The adapter is a residual branch with a zero-output
   initialization.
3. Stop the predictive loss at every pi0.5 feature used as an input or target.
   The world-model objective may update only the new adapter. Ordinary action
   fine-tuning follows the accepted A0 route and is not silently changed.
4. Remove the future-frame teacher at deployment. Inference receives only the
   observations, language, state, and action variables already available to
   pi0.5.
5. Audit the loaded parameter tree and checkpoint hash. A run initialized only
   from PaliGemma is a from-base VLA experiment, not a pi0.5-preserving one.

This boundary follows the motivation of Knowledge Insulation: action or
auxiliary objectives can damage pretrained vision-language knowledge when
their gradients enter the shared backbone. The official method separates the
continuous action expert from the VLM learning signal rather than treating a
frozen generic VLM as sufficient ([project page](https://www.pi.website/research/knowledge_insulation)).

## 3. Evidence already established locally

### 3.1 The failed route is specific

The corrected three-seed RoboTwin experiment gives A0 79.39%, A2-Abs 75.44%,
and A3 70.81%. Relative to A0, the paired hierarchical effects are -3.94
points (95% CI [-6.58, -0.97]) and -8.58 points ([-11.47, -5.75]). A3 loses on
all six tasks.

The latent predictor is nevertheless nontrivial: held-out cosine similarity is
0.8134 versus 0.7479 for persistence, and milestone retrieval reaches 80.8%
top-5. Correct predicted content also fails to beat current, zero, permuted,
cross-task, or within-task controls. A privileged spatial-token probe fails,
so the result cannot be attributed only to one-token pooling.

The oracle stage-transition follow-up changes A0 by only +0.53 points (95% CI
[-2.06, +3.00]) across three training seeds. It does not establish that better
milestone prediction would repair the route.

Therefore, the closed result is:

> Predictable observation-conditioned milestone features are not automatically
> action-relevant conditions for a pretrained pi0.5 policy.

It is not evidence that predictive training, action-conditioned dynamics, or a
world critic can never help a VLA.

### 3.2 The unresolved LaWAM attribution matters

The LaWAM-init/Future-off arm averages 90.83%, above every arm that keeps the
downstream world-model objective active. However, Future-off inherits LaWAM
pretraining and is not a clean VLA baseline. This leaves one plausible positive
mechanism unresolved: predictive training may improve initialization even when
online future conditioning is harmful.

LaWAM predicts future latent features conditioned on latent actions and then
uses latent visual subgoals ([paper](https://arxiv.org/abs/2606.15768)). Its
causal input is closer to dynamics than the observation-only MINT predictor.
The next experiment should isolate this distinction without discarding pi0.5.

## 4. Second-round method screening

The screening criteria are: complete pi0.5 inheritance, protection from WM
gradients, a direct action-relevant path, deployment cost, and whether the
local negative results already test the central mechanism.

| Candidate | pi0.5 inheritance | Control path | Decision |
|---|---|---|---|
| Observation-only milestone prefix (MINT/A2/A3) | High | Weak: future token must be interpreted implicitly | Stop; replicated negative result |
| LeWM/DINO encoder replacing SigLIP | Low | Direct perception replacement | Reject for this objective; removes a major pretrained component and changes tokenization |
| From-PaliGemma joint predictive pretraining | Low | Potentially direct after long retraining | Reject here; action expert and robot priors are re-learned rather than inherited |
| FutureVLA-style full visuomotor pretraining | Medium | Predictive representation is internalized | Scientifically relevant, but too much of pi0.5 is rewritten for the first test ([paper](https://arxiv.org/abs/2603.10712)) |
| Enfold-style predictive distillation | High if adapter-only | Current observation encoder imitates future-generator computation | Retain as the teacher-training mechanism ([paper](https://arxiv.org/abs/2607.26657)) |
| LaWAM-style action-conditioned latent dynamics | High if added residually | Future state depends on the candidate action | Retain as the policy-side mechanism |
| AHEAD future-token replacement | High | Compensates perception latency in dynamic scenes | Defer; current six tasks do not isolate moving-object latency ([paper](https://arxiv.org/abs/2606.02486)) |
| Semantic milestone through pi0.5 subtask prediction | High | High-level subtask directly conditions low-level action | Retain only as a later milestone-specific branch |
| World critic / VLA RL | High | Future latent estimates value and trains the policy | Valid but a separate RL project requiring rewards and failures ([paper](https://arxiv.org/abs/2607.29613)) |
| Latent Bridge | High | Predicts VLM cache between calls | Efficiency-only; not a control-success hypothesis ([paper](https://arxiv.org/abs/2605.02739)) |

The pi0.5 architecture already separates low-frequency semantic subtask
prediction from low-level action chunks. Its heterogeneous pretraining is the
source of much of its open-world ability, and the authors report that the same
model predicts a semantic subtask before the low-level action in long-horizon
operation ([pi0.5 paper](https://www.pi.website/download/pi05.pdf)). This makes
semantic subtasks the natural location for event milestones, but not for the
first latent-WM test: an event label changes the supervision and does not
resolve whether latent dynamics can improve the action expert.

## 5. Selected architecture

### 5.1 Training-time teacher

For a sample at time t:

- The accepted pi0.5 encoder produces stop-gradient current visual tokens.
- The same encoder processes the observed frame at a fixed physical horizon
  (initially +1.0 s) to produce stop-gradient target patch tokens.
- The adapter receives current tokens, language context, proprioception, and a
  summary of the ground-truth/noised action chunk.
- The adapter predicts a compact set of future latent tokens. Patch-level
  matching, not a single pooled feature, supplies the auxiliary target.
- Action-shuffled and action-masked inputs test whether prediction actually
  uses the proposed action consequence.

This combines the useful parts of Enfold-style training-time transfer and
LaWAM-style action conditioning. It does not require a generated RGB frame or
a world model at deployment.

### 5.2 Policy interface

The predicted latent must enter only the action expert. A zero-initialized,
layer-wise gate modulates action-expert hidden states or its existing AdaRMS
conditioning. It must not be appended to the PaliGemma prefix, and the same
vector must not simply be added to every action token.

The existing `LocalDynamicsAdapter` is useful scaffolding, but its current
route broadcasts one condition across action tokens. The claim-bearing
implementation should produce action-indexed or layer-indexed modulation and
publish an exact-zero invariance test before training.

### 5.3 Gradient routes

The required routes are:

| Loss | New adapter | Action expert | SigLIP/PaliGemma |
|---|---:|---:|---:|
| Predictive latent loss | update | no update | no update |
| Flow/action loss | update through gated route | accepted A0 update | accepted A0 update |

An optional stricter Knowledge-Insulated action route is a separate ablation,
not part of the first candidate. Changing both the WM route and A0 policy
gradient route in one arm would confound the attribution.

## 6. Minimal experiment and gates

### Offline gate

Train only the predictive adapter from the official pi0.5 checkpoint. Compare
action-conditioned, action-shuffled, and action-masked prediction on held-out
episodes. This gate only checks that the adapter models action-dependent future
variation; it cannot support a control claim.

Proceed only if action conditioning beats both controls with a paired 95% CI
excluding zero on the preregistered latent metric. Otherwise stop before policy
training.

### One-seed control gate

Train one candidate at seed 1000 with the accepted A0 data, normalization,
batch, update count, and scene manifest. Reuse accepted A0 only after an
initialization/config/source audit proves that all non-adapter fields match.

At the same candidate checkpoint evaluate normal, zero-gate, action-shuffled,
and action-masked interventions on identical scenes. Replication is authorized
only if:

1. normal candidate success exceeds A0;
2. normal exceeds zero-gate and both wrong-action controls;
3. no task regresses by more than 5 points; and
4. all comparisons and exclusions were fixed before rollout inspection.

### Replication gate

Only after the one-seed gate, train seeds 1001 and 1002. The final utility
claim requires the hierarchical 95% interval for candidate minus A0 to exclude
zero. Report all task and seed effects even if the aggregate passes.

## 7. Deferred milestone-specific route

If the predictive adapter passes, a separate experiment may map automatically
mined milestone events to concise semantic subtasks and train pi0.5's native
high-level prediction channel. The labels must encode action intent (for
example, which object, arm, contact, and placement relation), not generic stage
IDs. The low-level expert then executes the predicted subtask using the native
pi0.5 hierarchy.

This branch is deferred because the oracle stage-ID route already failed its
three-seed gate. Running it now would mix a new semantic-labeling hypothesis
with an unresolved latent-dynamics hypothesis.

## 8. Explicit exclusions

- Do not resume A2/A3 token, residual, pooling, or larger-predictor variants.
- Do not launch MT3--MT6 from the closed oracle-transition graph.
- Do not replace SigLIP with DINO/LeWM in a study whose goal is to preserve
  pi0.5 pretraining.
- Do not use the from-PaliGemma checkpoint as the main initialization.
- Do not select only Stack-2/Stack-3 after observing their positive MT1 means.
- Do not claim improvement from latent prediction metrics without closed-loop
  action-content interventions.

## 9. Paper boundary

The current manuscript should retain its completed negative result. A positive
adapter result, if obtained under the new preregistered protocol, would support
a separate contribution:

> Preserving a pretrained VLA requires future prediction to be distilled into
> an action-conditioned, gradient-insulated control interface; future-state
> predictability alone is insufficient.

Until the three-seed utility gate passes, this sentence is a hypothesis and
must not appear as a result claim.
