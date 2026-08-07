# Temporal-Grounding Analysis Gate Audit

Date: 2026-08-07 UTC

## Finding

The frozen TG2 plan requires all of the following before TG3 is eligible:

1. `fixed_endpoint - future_off` is accepted;
2. `fixed_endpoint - raw_milestone` is accepted;
3. both comparisons satisfy their task-safety gates.

`lmvla/lmwm/scripts/analyze_temporal_grounding_tg2.py` currently computes
`stop_decision.tg3_authorized` from only the accepted
`fixed_endpoint - raw_milestone` comparison. It does not require accepted
fixed-endpoint utility against `future_off`.

## Reproduction

A synthetic, exactly paired matrix was constructed with:

- `future_off`: 100% success;
- `fixed_endpoint`: 50% success;
- `raw_milestone`: 0% success;
- all three training seeds, six tasks, four evaluation seeds, and 50 paired
  episodes per cell.

The current analyzer returned:

```text
fixed_utility=False
horizon_effect=True
task_safety_fixed_vs_off=False
task_safety_fixed_vs_raw=True
tg3_authorized=True
```

This contradicts Sections 7--9 of `PAPER_TODO.md`.

## Execution Impact

No scientific result is affected yet: TG2 evaluations are blocked behind the
nine-arm training-integrity dependency, and no TG1A or TG2 final analysis task
is registered in the resource-aware scheduler. The mismatch must be repaired
and regression-tested before the frozen TG2 analysis command is executed or any
TG3 work is admitted.

## Required Resolution

The protocol-preserving implementation repair is to require both accepted
comparisons when computing `tg3_authorized`, while retaining all frozen
bootstrap, task-safety, pairing, and result-selection rules. Because the active
authorization explicitly forbids source, gate, and dependency changes, this
audit does not modify the analyzer or scheduler. An explicit authorized runtime
amendment is required before implementing the repair and registering the final
CPU analysis tasks.

## Minimal Authorized Implementation Plan

After explicit authorization, the smallest protocol-preserving change is:

1. Change only `stop_decision.tg3_authorized` to the conjunction of accepted
   fixed-endpoint utility and accepted fixed-versus-raw horizon effect.
2. Add regression cases proving that horizon-only acceptance does not authorize
   TG3 and that both accepted comparisons do authorize it.
3. Pin the repaired analyzer and tests in a new amendment extending runtime v10;
   retain the existing LaWAM commit and exact two-file LaWAM runtime diff.
4. Register `temporal_grounding_tg1a_analysis` as a local zero-GPU task requiring
   all four TG1A evaluation tasks. Use the unchanged TG1A admission
   `analysis_command`, result JSON, and gate marker.
5. Register `temporal_grounding_tg2_analysis` as a local zero-GPU task requiring
   all nine TG2 evaluation tasks. Use the unchanged TG2 admission
   `analysis_command`, result JSON, and gate marker, plus the repaired analyzer
   hash.
6. Keep every TG3/TG4/TG5 task absent or disabled until the resulting canonical
   gate JSON is interpreted under the existing branch table.

Both frozen analysis command lines parse successfully. Their scene-manifest
inputs exist, and their result and marker parent directories already exist.
