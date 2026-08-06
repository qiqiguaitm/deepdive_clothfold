#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LEROBOT_ROOT=${LEROBOT_ROOT:-/vePFS/tim/workspace/lerobot-main}
PYTHON=$LEROBOT_ROOT/.venv/bin/python
RUNTIME_DIR=$REPO/lmvla/lmwm/runtime/pi05_r4_training
TRAIN_ENTRYPOINT=$RUNTIME_DIR/train_entrypoint.py
ARM=${R4_ARM:?set R4_ARM}
SEED=${R4_SEED:?set R4_SEED to 1001 or 1002}
WORLD_SIZE=${WORLD_SIZE:-4}
STEPS=5000

case "$ARM" in ordinary|terminal_outcome|outcome_free_crave) ;; *) exit 2 ;; esac
case "$SEED" in 1001|1002) ;; *) echo "unsupported replication seed: $SEED" >&2; exit 2 ;; esac
test -x "$PYTHON"
test -f "$TRAIN_ENTRYPOINT"
test -f "$REPO/logs/r4/seed1000/r4_gate.accepted"
test -f "$REPO/logs/resource_markers/pi05_r4_training_runtime.ok"
test -f "$REPO/logs/resource_markers/pi05_r4_matched_runtime.ok"
test -f "$REPO/logs/resource_markers/pi05_r4_crave_sidecar.ok"

DATASET=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/lerobot_query_chunks
SIDECAR=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz
PUBLIC_MODEL=/vePFS/tim/hf_models/SidneyXie_pi05_robotwin
RUN_NAME=${ARM}-seed${SEED}
RUN_ROOT=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1
CONFIG=$REPO/logs/r4/training/configs/${RUN_NAME}.json
LOG=$REPO/logs/r4/training/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log
MARKER=$REPO/logs/resource_markers/pi05_r4_${RUN_NAME}.ok
if [[ -e "$RUN_ROOT/$RUN_NAME" ]]; then
  echo "refusing to overwrite an existing formal R4 run: $RUN_ROOT/$RUN_NAME" >&2
  exit 3
fi
mkdir -p "$(dirname "$CONFIG")" "$RUN_ROOT"
rm -f "$MARKER"
exec > >(tee -a "$LOG") 2>&1

"$REPO/kai0/.venv/bin/python" "$REPO/lmvla/lmwm/scripts/build_pi05_r4_replication_config.py" \
  --public-config "$PUBLIC_MODEL/train_config.json" --arm "$ARM" --seed "$SEED" \
  --world-size "$WORLD_SIZE" --steps "$STEPS" --output-dir "$RUN_ROOT/$RUN_NAME" \
  --dataset-root "$DATASET" --model-path "$PUBLIC_MODEL" --sidecar "$SIDECAR" \
  --output "$CONFIG"

"$PYTHON" - "$CONFIG" "$WORLD_SIZE" "$SEED" <<'PY'
import sys
import accelerate
import lerobot.policies
from lerobot.configs.train import TrainPipelineConfig

config = TrainPipelineConfig.from_pretrained(sys.argv[1])
config.validate()
world_size, seed = map(int, sys.argv[2:])
assert config.batch_size * world_size == 16
assert config.seed == seed
assert config.steps == 5000
assert accelerate.__version__ == "1.14.0"
PY

export PI05_R4_TRAINING_RUNTIME=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}
"$PYTHON" "$TRAIN_ENTRYPOINT" --check-binding
"$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes "$WORLD_SIZE" \
  --mixed_precision bf16 --main_process_port 0 "$TRAIN_ENTRYPOINT" --config_path "$CONFIG"

FINAL=$RUN_ROOT/$RUN_NAME/checkpoints/005000
test -s "$FINAL/pretrained_model/model.safetensors"
test -s "$FINAL/training_state/training_step.json"
"$PYTHON" - "$FINAL/training_state/training_step.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8")).get("step") == 5000
PY
printf 'completed=%s\narm=%s\nseed=%s\nworld_size=%s\nsteps=5000\nconfig=%s\nlog=%s\n' \
  "$(date -u +%FT%TZ)" "$ARM" "$SEED" "$WORLD_SIZE" "$CONFIG" "$LOG" > "$MARKER"
