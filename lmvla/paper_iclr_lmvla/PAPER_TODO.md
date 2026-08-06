# pi0.5-Preserving Predictive and Outcome-Calibrated Control TODO

Updated: 2026-08-06 10:55 UTC

No unfinished training, evaluation, analysis, efficiency, or claim-bearing
figure task remains in the current preregistered experiment graph. Completed
evidence and execution provenance are preserved in
`PAPER_EVIDENCE_ARCHIVE_2026-08-01.md`; canonical JSON artifacts take
precedence over status prose.

## Final evidence boundary

- The main MINT-VLA confirmatory result remains negative. Across three matched
  training seeds, offline absolute conditioning is -3.94 points from no hint
  (95% CI `[-6.58,-0.97]`) and MINT-VLA is -8.58 points
  (`[-11.47,-5.75]`). The oracle-transition follow-up also fails its gate at
  +0.53 points (`[-2.06,+3.00]`).
- The separate policy-preserving predictive adapter passes P2. Candidate
  effects against the fixed matched A0 are +13.42, +9.08, and +12.33 points
  at seeds 1000--1002. Their equal-seed mean is +11.61 points with a
  hierarchical paired 95% CI of `[+8.31,+14.67]`.
- P2 establishes replicated candidate utility relative to one fixed matched
  A0 checkpoint. It does not establish content-specific causality or
  independent baseline-seed replication: the only normal-versus-shuffled
  intervention remains +1.25 points with Holm-adjusted `p=1.0` at seed 1000.
- The predictive adapter adds 16.82 million parameters (0.50%), 0.93 ms mean
  direct-model latency, 2.76 ms WebSocket round-trip latency, and 18 MiB
  four-GPU peak training memory. Compiled XLA cost analysis reports no FLOP
  increase (164.92 versus 172.29 GFLOPs for A0), which is reported as a
  compiler estimate rather than a theoretical operation count.
- R1 remains rejected: adding recurrence-derived CRAVE targets lowers success
  to 62.92%. R4 also remains rejected at replication: terminal-outcome
  weighting exceeds ordinary by +2.81 points (95% CI `[-0.94,+6.58]`) and
  outcome-free CRAVE by +3.81 points (`[-0.25,+7.97]`).

## Completion evidence

- P2 seed-1001 and seed-1002 evaluations each contain 24 valid cells and
  1,200 episodes with exact scene pairing and zero invalid cell.
- `RESULTS_pi05_predictive_adapter_p2_gate.json` records the accepted frozen
  three-seed gate.
- `RESULTS_pi05_predictive_adapter_p2_efficiency.json` records the completed
  matched parameter, compiled-FLOP, latency, throughput, and peak-memory
  comparison.
- The inference config-name repair changed only the A0 registered inference
  config and is frozen in
  `manifests/pi05_predictive_adapter_p2_efficiency_config_repair_v1.json`.
- No new claim-bearing figure was required; the final P2 evidence is reported
  in a full task-by-seed table and canonical JSON.

## Scheduler closure

The current graph has no runnable experiment node. The obsolete two-GPU P2
local accelerator is disabled automatically once the accepted final gate is
present. gf1 remains retired, robot-task submissions remain disabled, and no
new LeWM/DINO, from-PaliGemma, A2/A3, R1-replication, or closed MT3--MT6 task is
authorized by this TODO.

The anonymous archive URL in `main.tex` remains a submission-packaging field,
not an experiment or evidence task.
