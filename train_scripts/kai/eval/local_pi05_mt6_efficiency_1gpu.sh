#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI05_MT6_EFFICIENCY_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-mt6-efficiency.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PI05_MT6_EFFICIENCY_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PY=$REPO/kai0/.venv/bin/python
BENCH=$REPO/train_scripts/kai/analysis/benchmark_pi05_policy_latency.py
SELECTION=${MT3_SELECTION:-$REPO/logs/mt_stage_tracker/selection.json}
OUT_DIR=$REPO/logs/efficiency/pi05_mt6_selected
FINAL=$REPO/logs/efficiency/pi05_mt6_selected.json
mkdir -p "$OUT_DIR"

export OPENPI_DATA_HOME=$REPO/openpi_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"

candidate=$($PY -c "import json; print(json.load(open('$SELECTION'))['selected'])")
case "$candidate" in
  current_frame) history_steps=0 ;;
  history_proprio) history_steps=3 ;;
  *) echo "invalid selected tracker: $candidate" >&2; exit 2 ;;
esac

A0=$OUT_DIR/a0.json
MT3=$OUT_DIR/mt3_${candidate}.json
A0_CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/pi05_robotwin_a0_public_exact_seed1000/49999
MT3_CKPT=$REPO/kai0/checkpoints/pi05_robotwin_mt3_learned_exact/pi05_robotwin_mt3_learned_seed1000/49999

if [[ ! -s "$A0" ]]; then
  $PY "$BENCH" \
    --arm a0 \
    --config pi05_robotwin_a0_public_exact_bj \
    --checkpoint "$A0_CKPT" \
    --warmup 5 --trials 30 --output "$A0"
fi

if [[ ! -s "$MT3" ]]; then
  args=(
    --arm "mt3_${candidate}"
    --config "pi05_robotwin_mt3_learned_${candidate}_exact"
    --checkpoint "$MT3_CKPT"
    --transition-task-id 0
    --warmup 5 --trials 30 --output "$MT3"
  )
  if [[ "$history_steps" != 0 ]]; then
    args+=(--history-steps "$history_steps")
  fi
  LMWM_TRANSITION_INTERVENTION=predicted $PY "$BENCH" "${args[@]}"
fi

export A0 MT3 FINAL SELECTION candidate
$PY - <<'PY'
import hashlib
import json
import os
from pathlib import Path

a0 = json.loads(Path(os.environ["A0"]).read_text())
mt3 = json.loads(Path(os.environ["MT3"]).read_text())
selection = Path(os.environ["SELECTION"])
report = {
    "benchmark": "pi05_mt6_selected_efficiency",
    "selected_tracker": os.environ["candidate"],
    "selection": str(selection),
    "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "protocol": {
        "hardware_comparison": "same allocated 80-GiB GPU and process environment",
        "warmup": 5,
        "trials": 30,
        "flops_source": "compiled JAX executable cost_analysis",
    },
    "arms": {"a0": a0, "mt3": mt3},
    "delta": {
        "parameters": mt3["model_parameter_count"] - a0["model_parameter_count"],
        "flops": mt3["xla_cost_analysis"]["flops"] - a0["xla_cost_analysis"]["flops"],
        "warm_gpu_memory_mib": mt3["gpu_memory_mib"]["after_warmup"] - a0["gpu_memory_mib"]["after_warmup"],
        "direct_model_mean_ms": mt3["direct_model"]["mean_ms"] - a0["direct_model"]["mean_ms"],
        "websocket_roundtrip_mean_ms": mt3["websocket_roundtrip"]["mean_ms"] - a0["websocket_roundtrip"]["mean_ms"],
        "websocket_throughput_requests_per_second": mt3["websocket_throughput_requests_per_second"] - a0["websocket_throughput_requests_per_second"],
    },
}
final = Path(os.environ["FINAL"])
temporary = final.with_suffix(final.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(final)
print(final)
PY
