#!/usr/bin/env bash
set -euo pipefail

if [[ "${PREDICTIVE_MEMORY_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-predictive-memory.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PREDICTIVE_MEMORY_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P2_VERIFY_REPO:-$REPO}
SOURCE_REPO=${TRAIN_SOURCE_REPO:-$VERIFY_REPO}
P2_GATE=${PREDICTIVE_P2_GATE:-$REPO/logs/predictive/p2_eval/p2_gate.accepted}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json
PY=$REPO/kai0/.venv/bin/python
OUT_DIR=$REPO/logs/efficiency/pi05_predictive_adapter_train_memory
FINAL=$REPO/logs/efficiency/pi05_predictive_adapter_train_memory.json
GPU_COUNT=${GPU_COUNT:-4}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}

test -f "$P2_GATE"
if [[ "$VERIFY_REPO" != "$REPO" || "$SOURCE_REPO" != "$REPO" ]]; then
  test -s "$VERIFY_REPO/REPLICATION_READY"
  test -s "$SOURCE_REPO/REPLICATION_READY"
fi
python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$VERIFY_REPO" --manifest "$PROTOCOL"
test "$GPU_COUNT" -eq 4
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/openpi_cache}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax-predictive-memory}
export PYTHONPATH="$SOURCE_REPO/kai0/src:$SOURCE_REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_arm() {
  local label=$1 train_arm=$2 config=$3
  local telemetry=$OUT_DIR/${label}.csv
  local log=$OUT_DIR/${label}.log
  local report=$OUT_DIR/${label}.json
  local exp=pi05_predictive_adapter_memory_${label}_${RUN_ID}
  local extra=()
  if [[ "$label" == candidate ]]; then
    extra+=(
      --adapter-checkpoint "$REPO/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999"
      --target-pairs "$REPO/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/pairs.npz"
      --frame-cache-root "$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
    )
  fi
  (
    while true; do
      printf '%s,' "$(date -u +%s.%N)"
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n "$GPU_COUNT" | paste -sd, -
      sleep 0.2
    done
  ) >"$telemetry" &
  local monitor_pid=$!
  set +e
  (
    cd "$REPO/kai0"
    "$PY" -u "$SOURCE_REPO/kai0/scripts/train_pi05_robotwin_confirmatory.py" \
      --arm "$train_arm" \
      --config-name "$config" \
      --exp-name "$exp" \
      --seed 1000 \
      --data-repo "$REPO/../VLANeXt-main/datasets/robotwin2.0_official_prompts_v21" \
      --init-params "$REPO/kai0/checkpoints/pi05_base/params" \
      --norm-assets-dir "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj" \
      --num-train-steps 10 \
      --save-interval 1000 \
      --log-interval 1 \
      --num-workers 8 \
      --fsdp-devices 1 \
      --skip-final-checkpoint \
      "${extra[@]}"
  ) >"$log" 2>&1
  local rc=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  [[ "$rc" -eq 0 ]] || { tail -100 "$log" >&2; return "$rc"; }
  ARM_LABEL=$label CONFIG_LABEL=$config TELEMETRY=$telemetry LOG=$log REPORT=$report \
    "$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

rows = []
with Path(os.environ["TELEMETRY"]).open() as stream:
    for row in csv.reader(stream):
        if len(row) == 5:
            rows.append([float(row[0]), *[int(value.strip()) for value in row[1:]]])
if not rows:
    raise RuntimeError("no four-GPU memory telemetry")
peaks = [max(row[index] for row in rows) for index in range(1, 5)]
report = {
    "arm": os.environ["ARM_LABEL"],
    "config": os.environ["CONFIG_LABEL"],
    "protocol": {"global_batch_size": 16, "fsdp_devices": 1, "steps": 10, "sample_interval_seconds": 0.2},
    "samples": len(rows),
    "per_gpu_peak_memory_mib": peaks,
    "max_peak_memory_mib": max(peaks),
    "telemetry": os.environ["TELEMETRY"],
    "log": os.environ["LOG"],
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
}

run_arm a0 a0 pi05_predictive_adapter_p1_a0_exact
run_arm candidate p1_predictive pi05_predictive_adapter_p1

export FINAL OUT_DIR P2_GATE
"$PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
a0 = json.loads((root / "a0.json").read_text())
candidate = json.loads((root / "candidate.json").read_text())
gate = Path(os.environ["P2_GATE"])
report = {
    "benchmark": "pi05_predictive_adapter_matched_training_memory",
    "p2_gate": str(gate),
    "p2_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
    "arms": {"a0": a0, "candidate": candidate},
    "delta_candidate_minus_a0_mib": candidate["max_peak_memory_mib"] - a0["max_peak_memory_mib"],
}
final = Path(os.environ["FINAL"])
temporary = final.with_suffix(final.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(final)
PY
