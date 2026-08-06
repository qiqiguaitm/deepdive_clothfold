#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PY=$REPO/kai0/.venv/bin/python
SELECTION=${MT3_SELECTION:-$REPO/logs/mt_stage_tracker/selection.json}
BASELINE=$REPO/logs/efficiency/pi05_train_memory_a0.json
FINAL=$REPO/logs/efficiency/pi05_mt6_train_memory_selected.json
TELEMETRY=$REPO/logs/efficiency/pi05_mt6_train_memory_selected.csv
LOG=$REPO/logs/efficiency/pi05_mt6_train_memory_selected.log
DATA=${PI05_EXACT_DATASET:-/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21}
INIT=${PI05_BASE_PARAMS:-$REPO/kai0/checkpoints/pi05_base/params}
PAIRS=${TRANSITION_PAIRS:-$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz}
ASSET_ID=robotwin2.0_absolute_meanstd
NORM=$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj/$ASSET_ID/norm_stats.json

mkdir -p "$(dirname "$FINAL")"
test -s "$SELECTION"
test -s "$BASELINE"
test -f "$INIT/_METADATA"
test -f "$PAIRS"
test -f "$NORM"
test -d "$DATA/meta"

candidate=$($PY -c "import json; print(json.load(open('$SELECTION'))['selected'])")
case "$candidate" in current_frame|history_proprio) ;; *) echo "invalid selected tracker: $candidate" >&2; exit 2 ;; esac
TRACKER=$REPO/logs/mt_stage_tracker/$candidate/tracker.pt
test -f "$TRACKER"

CONFIG=pi05_mt6_memory_${candidate}
EXP=pi05_mt6_memory_${candidate}
mkdir -p "$REPO/kai0/assets/$CONFIG/$ASSET_ID"
cp -f "$NORM" "$REPO/kai0/assets/$CONFIG/$ASSET_ID/norm_stats.json"

exec > >(tee "$LOG") 2>&1
(
  while true; do
    printf '%s,' "$(date -u +%s.%N)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 4 | paste -sd, -
    sleep 0.2
  done
) > "$TELEMETRY" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

export PATH=$REPO/kai0/.venv/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=$REPO/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax-mt6-memory}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd "$REPO/kai0"
"$PY" -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm mt3_learned \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed 1000 \
  --data-repo "$DATA" \
  --init-params "$INIT" \
  --asset-id "$ASSET_ID" \
  --transition-pairs "$PAIRS" \
  --tracker-checkpoint "$TRACKER" \
  --tracker-candidate "$candidate" \
  --num-train-steps 10 \
  --num-workers 8 \
  --save-interval 1000 \
  --log-interval 1 \
  --skip-final-checkpoint

cleanup
trap - EXIT
export BASELINE FINAL TELEMETRY LOG candidate CONFIG SELECTION TRACKER
"$PY" - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

rows = []
with Path(os.environ["TELEMETRY"]).open() as stream:
    for row in csv.reader(stream):
        if len(row) >= 5:
            rows.append([float(row[0]), *[int(value.strip()) for value in row[1:5]]])
if not rows:
    raise RuntimeError("no GPU memory telemetry")
per_gpu_peak = [max(row[index] for row in rows) for index in range(1, 5)]
baseline = json.loads(Path(os.environ["BASELINE"]).read_text())
selected_peak = max(per_gpu_peak)
report = {
    "benchmark": "pi05_mt6_selected_training_memory",
    "selected_tracker": os.environ["candidate"],
    "selection_sha256": hashlib.sha256(Path(os.environ["SELECTION"]).read_bytes()).hexdigest(),
    "tracker": os.environ["TRACKER"],
    "config": os.environ["CONFIG"],
    "hardware": "4x Shanghai A100 (ml.pni2.14xlarge)",
    "protocol": {
        "batch_size": 16,
        "fsdp_devices": 1,
        "steps": 10,
        "num_workers": 8,
        "sample_interval_seconds": 0.2,
        "final_checkpoint_saved": False,
    },
    "samples": len(rows),
    "per_gpu_peak_memory_mib": per_gpu_peak,
    "max_peak_memory_mib": selected_peak,
    "a0_max_peak_memory_mib": baseline["max_peak_memory_mib"],
    "delta_vs_a0_mib": selected_peak - baseline["max_peak_memory_mib"],
    "telemetry": os.environ["TELEMETRY"],
    "log": os.environ["LOG"],
}
final = Path(os.environ["FINAL"])
temporary = final.with_suffix(final.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(final)
print(json.dumps(report, indent=2, sort_keys=True))
PY
