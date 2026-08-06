#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PROBE_GPUS=${PROBE_GPUS:?set PROBE_GPUS}
PROBE_EXP=${PROBE_EXP:?set PROBE_EXP}
RESULT=${RESULT:?set RESULT}
STEPS=${STEPS:-300}
WORKERS=${WORKERS:-8}
FSDP_DEVICES=${FSDP_DEVICES:-1}
LOG_INTERVAL=${LOG_INTERVAL:-10}
LOG_DIR=$REPO/logs/scaling
LOG=$LOG_DIR/${PROBE_EXP}.log

mkdir -p "$LOG_DIR" "$(dirname "$RESULT")"
if [[ -s "$RESULT" ]]; then
  echo "probe already complete: $RESULT"
  exit 0
fi

actual_gpus=$(nvidia-smi -L | wc -l)
if [[ "$actual_gpus" -ne "$PROBE_GPUS" ]]; then
  echo "expected $PROBE_GPUS GPUs, found $actual_gpus" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=$REPO/openpi_cache
export JAX_COMPILATION_CACHE_DIR=/tmp/jax-pi05-scaling-${PROBE_GPUS}g
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd "$REPO/kai0"
set +e
.venv/bin/python -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm mt1_oracle \
  --config-name pi05_robotwin_mt1_oracle_exact \
  --exp-name "$PROBE_EXP" \
  --seed 4242 \
  --data-repo /vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21 \
  --init-params "$REPO/kai0/checkpoints/pi05_base/params" \
  --asset-id robotwin2.0_absolute_meanstd \
  --transition-pairs "$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz" \
  --num-train-steps "$STEPS" \
  --num-workers "$WORKERS" \
  --fsdp-devices "$FSDP_DEVICES" \
  --save-interval 10000 \
  --log-interval "$LOG_INTERVAL" \
  --skip-final-checkpoint \
  2>&1 | tee "$LOG"
train_rc=${PIPESTATUS[0]}
set -e
if [[ "$train_rc" -ne 0 ]]; then
  exit "$train_rc"
fi

"$REPO/kai0/.venv/bin/python" - "$LOG" "$RESULT" "$PROBE_GPUS" "$STEPS" "$WORKERS" "$FSDP_DEVICES" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
gpus = int(sys.argv[3])
steps = int(sys.argv[4])
workers = int(sys.argv[5])
fsdp_devices = int(sys.argv[6])
text = log_path.read_text(errors="replace")
pattern = re.compile(
    r"Progress on:\s*([0-9.]+)(k?)it/.*?elapsed:(\d+):(\d+)(?::(\d+))?"
)
points = []
for match in pattern.finditer(text):
    count = float(match.group(1)) * (1000 if match.group(2) else 1)
    if match.group(5) is None:
        elapsed = int(match.group(3)) * 60 + int(match.group(4))
    else:
        elapsed = (
            int(match.group(3)) * 3600
            + int(match.group(4)) * 60
            + int(match.group(5))
        )
    points.append((count, elapsed))

if not points:
    raise SystemExit("no progress timing records found")
max_count = max(count for count, _ in points)
fit = [(count, elapsed) for count, elapsed in points if count >= max(50, max_count * 0.25)]
if len(fit) < 2 or max_count < steps * 0.9:
    raise SystemExit(f"insufficient steady-state progress records: max={max_count}, n={len(fit)}")
mean_x = sum(x for x, _ in fit) / len(fit)
mean_y = sum(y for _, y in fit) / len(fit)
denominator = sum((x - mean_x) ** 2 for x, _ in fit)
slope = sum((x - mean_x) * (y - mean_y) for x, y in fit) / denominator
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "config": "pi05_robotwin_mt1_oracle_exact",
    "seed": 4242,
    "gpus": gpus,
    "global_batch_size": 16,
    "per_device_batch_size": 16 / gpus,
    "num_workers": workers,
    "fsdp_devices": fsdp_devices,
    "requested_steps": steps,
    "observed_steps": max_count,
    "fit_points": len(fit),
    "steady_state_seconds_per_step": slope,
    "steady_state_steps_per_second": 1 / slope,
    "steady_state_samples_per_second": 16 / slope,
    "log": str(log_path),
}
temporary = result_path.with_suffix(result_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(result_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
