#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI05_EFFICIENCY_SCRIPT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-efficiency-latency.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PI05_EFFICIENCY_SCRIPT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=/vePFS/tim/workspace/deepdive_kai0
PY=$REPO/kai0/.venv/bin/python
BENCH=$REPO/train_scripts/kai/analysis/benchmark_pi05_policy_latency.py
OUT_DIR=$REPO/logs/efficiency/pi05_a0_a2_a3_latency
FINAL=$REPO/logs/efficiency/pi05_a0_a2_a3_latency.json
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=$REPO/openpi_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"

run_arm() {
  local arm=$1 config=$2 checkpoint=$3 hint_dim=$4
  local output=$OUT_DIR/$arm.json
  [[ -s "$output" ]] && { echo "skip completed arm=$arm"; return; }
  "$PY" "$BENCH" \
    --arm "$arm" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --hint-dim "$hint_dim" \
    --warmup 5 \
    --trials 30 \
    --output "$output"
}

run_arm \
  a0 \
  pi05_robotwin_a0_official_bj \
  "$REPO/kai0/checkpoints/pi05_robotwin_a0_official_bj/pi05_robotwin_a0_official/19999" \
  0
run_arm \
  a2 \
  pi05_robotwin_a2_prefix_official_eval_bj \
  "$REPO/kai0/checkpoints/pi05_robotwin_a2_prefix_official_bj/pi05_robotwin_a2_prefix_official/19999" \
  1152
run_arm \
  a3 \
  pi05_robotwin_a3_live_residual_prefix_official_eval \
  "$REPO/kai0/checkpoints/pi05_robotwin_a3_live_residual_prefix_official_east/pi05_robotwin_a3_live_residual_prefix_official/19999" \
  0

export OUT_DIR FINAL
"$PY" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
final = Path(os.environ["FINAL"])
arms = {name: json.loads((out_dir / f"{name}.json").read_text()) for name in ("a0", "a2", "a3")}
report = {
    "benchmark": "pi05_a0_a2_a3_latency",
    "comparison_scope": "architecture-level inference overhead; checkpoint weights do not alter graph shape",
    "arms": arms,
}
final.parent.mkdir(parents=True, exist_ok=True)
final.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(final)
PY
