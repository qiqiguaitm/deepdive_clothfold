# pi0.5 Spatial Future-Condition Interface Plan

Updated: 2026-08-01 21:02 UTC

## Decision objective

Determine whether future visual information can reduce action error when pi0.5
receives it through a spatially structured native-vision interface. This test
precedes any larger predictor, additional training seeds, or six-task method
matrix.

The current evidence separates prediction from control:

- held-out future prediction improves latent cosine over persistence by 0.0655;
- a stronger state-dependent retrieval milestone changes success by only
  +0.34 pp (`p=0.738`);
- the current A2/A3 one-token interfaces show no correct-content advantage.

The next experiment therefore changes the policy interface, not predictor
capacity.

## Interface contract

1. Encode milestone images with the same current pi0.5 vision encoder used for
   policy observations.
2. Preserve two-dimensional token order. The first implementation may use a
   fixed 4x4 spatial pooling grid (16 tokens) to bound attention cost, but must
   not collapse the target to one global token.
3. Apply stop-gradient to target-image tokens before the policy adapter. The
   observation branch still receives the normal action gradient.
4. Train the spatial adapter, confidence gate, and action expert with the
   action objective. A detached target must not imply a detached adapter.
5. Match token count and trainable parameters in all non-content controls.
6. Record target availability, gate value, token norm, and action sensitivity
   for every evaluated sample.

## Stage S0: implementation and offline sufficiency probe

Use held-out successful demonstration transitions where the milestone frame is
available. Compare three parameter-matched conditions:

| Arm | Condition | Purpose |
|---|---|---|
| S0-N | learned no-goal tokens | capacity/token-presence control |
| S0-C | spatial tokens from the current observation | duplicated-state control |
| S0-P | privileged milestone spatial tokens | future-information condition |

Primary endpoint: held-out action flow loss under identical action noise and
sample IDs. Secondary endpoints: action-vector cosine and endpoint error,
reported separately for Stack-3 and Hammer.

The S0 training protocol was frozen before reading any arm result: seed 1000,
batch 16, 1,000 updates, the exact pi0.5 base initialization and optimizer,
and 320 training episodes (160 per task). The held-out set contains 80 disjoint
episodes (40 per task). Each arm uses the same 16-token adapter and parameter
initialization; only the condition source differs. Checkpoints are accepted
only at update 999. The deterministic held-out sample IDs, action noise keys,
and metric implementation must be hashed before the first checkpoint is
evaluated.

Go/no-go rule: S0-P must improve the prespecified Stack-3 action endpoint over
both S0-N and S0-C without a material Hammer regression. If it does not, stop
the spatial predictor program and retain the result as an interface limit.

Outcome (2026-08-01): S0-P does not pass. Stack-3 endpoint L2 is 0.3378 for
S0-P versus 0.3274/0.3580 for S0-N/S0-C, so the privileged condition does not
beat both controls. Hammer is 0.3080 versus 0.2355/0.2403; paired episode-level
bootstrap gives S0-P minus S0-N +0.0725 (95% CI [+0.0391, +0.1110]) and S0-P
minus S0-C +0.0677 ([+0.0332, +0.1054]). T3b is therefore closed. Canonical
statistics are in `logs/spatial_s0/s0_offline_verdict.json`; the positive gate
marker is intentionally absent.

## Stage S1: two-task closed-loop smoke

Run only after S0 passes. Train matched Stack-3/Hammer policies from identical
initialization and data using the same three conditions. Use fixed scene
manifests and at least two evaluation seeds. This is a development gate, not a
paper result.

For closed-loop S0-P, use a scene- and phase-matched demonstration milestone
image provider. Report it as a privileged demonstration condition, not a
future-peek oracle. The provider must be frozen before rollout evaluation.

Go/no-go rule: Stack-3 success improves over both controls and Hammer does not
lose more than 2 pp. Otherwise do not launch the six-task spatial matrix.

## Stage S2: predicted spatial condition

Run only after S1 passes.

- Replace privileged milestone patches with online predicted or
  state-retrieved patches while keeping the policy adapter fixed in shape.
- Train with a mixture of privileged, generated, no-goal, and corrupted
  conditions so deployment errors are represented during optimization.
- Condition prediction on the current observation and subtask instruction.
- Refresh on subtask change and compare against a fixed physical-time refresh.
- Gate on predictor confidence and measured information increment over the
  current-state condition.

The predicted condition must retain a prespecified fraction of the privileged
gain before training-seed replication. Correct/current/zero/shuffled controls
then repeat on identical scene IDs.

## Resource escalation

| Stage | Resource | Launch condition |
|---|---|---|
| S0 code/preflight | local CPU or 1 GPU | exact A0 may still be training |
| S0 offline probe | 1 GPU | shape, mask, gradient, and provenance tests pass |
| S1 two-task smoke | 3 matched arms | exact A0 gate accepted and S0 positive |
| S2 predicted smoke | 1 predictor + matched controls | S1 positive |
| Six-task/seeds | 4xA100 per arm | S2 retains privileged benefit |

No S1/S2 production job should be added to the resource-aware scheduler until
its preceding marker is present. The scheduler marker names are reserved as:

- `logs/resource_markers/pi05_spatial_s0_offline.ok`
- `logs/resource_markers/pi05_spatial_s1_closedloop.ok`
- `logs/resource_markers/pi05_spatial_s2_predicted.ok`

## Required implementation tests

- target images never enter the ordinary observation camera dictionary;
- stop-gradient blocks the target branch from updating the shared encoder;
- action gradients update the spatial adapter and gate;
- S0-N/S0-C/S0-P have identical token count and trainable parameter count;
- absent targets produce a deterministic no-goal condition;
- shuffled targets alter action outputs before any rollout is submitted;
- config, checkpoint, dataset, target provider, and scene manifest hashes are
  written beside every result.
