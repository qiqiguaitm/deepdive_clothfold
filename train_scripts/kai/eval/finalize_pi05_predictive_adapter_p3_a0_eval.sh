#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
STAGE=${P3_EVAL_REPO:-$ROOT/.staging/pi05_p1_failover_20260804T1034Z}
VERIFY_REPO=${P345_VERIFY_REPO:-$ROOT}
SEED=${SEED:?set SEED to 1001 or 1002}

case "$SEED" in 1001|1002) ;; *) echo "P3 seed must be 1001 or 1002" >&2; exit 2;; esac

PY=$ROOT/kai0/.venv/bin/python
MANIFEST=$STAGE/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
RESULT_NAME=pi05_predictive_adapter_p3_a0_seed${SEED}
RESULT_ROOT=$STAGE/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$STAGE/lmvla/lmwm/docs/${RESULT_NAME}.json
MARKER=$STAGE/logs/resource_markers/${RESULT_NAME}.ok

"$PY" "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$VERIFY_REPO" \
  --manifest "$VERIFY_REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p3_protocol.json" \
  --phase p3
"$PY" "$STAGE/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" \
  --root "$RESULT_ROOT"

mkdir -p "$(dirname "$REPORT")" "$(dirname "$MARKER")"
"$PY" "$STAGE/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
printf 'validated=%s\ncondition=a0\ntraining_seed=%s\nreport=%s\nfinalizer=cpu-recovery\n' \
  "$(date -u +%FT%TZ)" "$SEED" "$REPORT" >"$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
