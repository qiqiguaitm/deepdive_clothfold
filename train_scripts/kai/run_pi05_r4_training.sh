#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LEROBOT_ROOT=${LEROBOT_ROOT:-/vePFS/tim/workspace/lerobot-main}
PYTHON=$LEROBOT_ROOT/.venv/bin/python
ARM=${R4_ARM:?set R4_ARM to ordinary, terminal_outcome, or outcome_free_crave}
WORLD_SIZE=${WORLD_SIZE:-4}
STEPS=${R4_STEPS:-5000}
SMOKE=${R4_SMOKE:-0}
SEED=1000

case "$ARM" in
  ordinary|terminal_outcome|outcome_free_crave) ;;
  *) echo "unsupported R4 arm: $ARM" >&2; exit 2 ;;
esac
test -x "$PYTHON"
test -f "$REPO/logs/resource_markers/pi05_r4_training_runtime.ok"
test -f "$REPO/logs/resource_markers/pi05_r4_matched_runtime.ok"
test -f "$REPO/logs/resource_markers/pi05_r4_crave_sidecar.ok"

DATASET=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/lerobot_query_chunks
SIDECAR=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz
PUBLIC_MODEL=/vePFS/tim/hf_models/SidneyXie_pi05_robotwin
CONFIG_DIR=$REPO/logs/r4/training/configs
RUN_ROOT=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1
if [[ "$SMOKE" == 1 ]]; then
  RUN_NAME=smoke-${ARM}-${WORLD_SIZE}g
  rm -rf "$RUN_ROOT/$RUN_NAME"
else
  RUN_NAME=${ARM}-seed${SEED}
  if [[ -e "$RUN_ROOT/$RUN_NAME" ]]; then
    echo "refusing to overwrite an existing formal R4 run: $RUN_ROOT/$RUN_NAME" >&2
    exit 3
  fi
fi
mkdir -p "$CONFIG_DIR" "$RUN_ROOT"
CONFIG=$CONFIG_DIR/${RUN_NAME}.json
LOG=$REPO/logs/r4/training/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log
MARKER=$REPO/logs/resource_markers/pi05_r4_${RUN_NAME}.ok
rm -f "$MARKER"
exec > >(tee -a "$LOG") 2>&1

BUILD_ARGS=()
if [[ "$SMOKE" == 1 ]]; then BUILD_ARGS+=(--smoke); fi
"$REPO/kai0/.venv/bin/python" "$REPO/lmvla/lmwm/scripts/build_pi05_r4_train_config.py" \
  --public-config "$PUBLIC_MODEL/train_config.json" \
  --arm "$ARM" --world-size "$WORLD_SIZE" --steps "$STEPS" \
  --output-dir "$RUN_ROOT/$RUN_NAME" --dataset-root "$DATASET" \
  --model-path "$PUBLIC_MODEL" --sidecar "$SIDECAR" --output "$CONFIG" \
  "${BUILD_ARGS[@]}"

"$PYTHON" - "$CONFIG" "$WORLD_SIZE" <<'PY'
import sys
import accelerate
import lerobot.policies  # Registers the pi05 config with draccus.
from lerobot.configs.train import TrainPipelineConfig

config = TrainPipelineConfig.from_pretrained(sys.argv[1])
config.validate()
world_size = int(sys.argv[2])
if config.batch_size * world_size != 16:
    raise ValueError("R4 effective batch diverged from the frozen public batch 16")
if accelerate.__version__ != "1.14.0":
    raise ValueError(f"unexpected accelerate version: {accelerate.__version__}")
print(
    f"PREFLIGHT arm={config.job_name} world_size={world_size} "
    f"per_process_batch={config.batch_size} effective_batch=16 steps={config.steps}",
    flush=True,
)
PY

export PI05_R4_TRAINING_RUNTIME=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO/lmvla/lmwm/runtime/pi05_r4_training${PYTHONPATH:+:$PYTHONPATH}
"$PYTHON" -m accelerate.commands.launch \
  --multi_gpu --num_processes "$WORLD_SIZE" --mixed_precision bf16 \
  --main_process_port 0 -m lerobot.scripts.lerobot_train --config_path "$CONFIG"

if [[ "$SMOKE" != 1 ]]; then
  FINAL=$RUN_ROOT/$RUN_NAME/checkpoints/$(printf '%06d' "$STEPS")
  test -s "$FINAL/pretrained_model/model.safetensors"
  test -s "$FINAL/training_state/training_step.json"
  "$PYTHON" - "$FINAL/training_state/training_step.json" "$STEPS" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("step") != int(sys.argv[2]):
    raise ValueError(f"incomplete R4 final training state: {payload}")
PY
fi
printf 'completed=%s\narm=%s\nworld_size=%s\nsteps=%s\nconfig=%s\nlog=%s\n' \
  "$(date -u +%FT%TZ)" "$ARM" "$WORLD_SIZE" "$STEPS" "$CONFIG" "$LOG" > "$MARKER"
