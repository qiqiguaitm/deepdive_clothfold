#!/usr/bin/env bash
set -euo pipefail

REPO="${PUBLIC_PI05_REPO:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
MODEL="${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}"
TOKENIZER="${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}"
SERVER_PY="${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${PUBLIC_PI05_LOG_DIR:-$LAWAM/logs/volc_rteval}"
mkdir -p "$LOG_DIR"

test -f "$MODEL/model.safetensors"
test -f "$TOKENIZER/tokenizer.model"
test -x "$SERVER_PY"

export STAR_VLA_PYTHON="$SERVER_PY"
export PYTHONPATH="$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT="$REPO/train_scripts/kai/eval/serve_lerobot_pi05.py"
export PALIGEMMA_TOKENIZER_PATH="$TOKENIZER"
export KAI0_ROOT="$REPO"
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}"
export ROBOTWIN_TASKS="${ROBOTWIN_TASKS:-beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PORT_SEARCH_LIMIT=30

SEEDS="${SEEDS:-0 1 2 3}"
GPU_COUNT="${LOCAL_GPU_COUNT:-4}"
SEED_STAGGER_SECONDS="${SEED_STAGGER_SECONDS:-10}"
RESULT_NAME="${RESULT_NAME:-pi05_public_samebridge_4seed_v3}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-public-samebridge-v3}"

read -r -a seed_list <<<"$SEEDS"
if (( ${#seed_list[@]} > GPU_COUNT )); then
  echo "seed count ${#seed_list[@]} exceeds GPU_COUNT=$GPU_COUNT" >&2
  exit 2
fi

pids=()
status=0
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="$index"
    export GPU_IDS="$index"
    export SEED="$seed"
    export PORT_BASE=$((11200 + index * 80))
    export ROBOTWIN_EVAL_ROOT="$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed"
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$LAWAM"
    bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
      "$MODEL" "$TASK_CONFIG" "$RUN_TAG_PREFIX-seed$seed"
  ) >"$LOG_DIR/${RESULT_NAME}_seed${seed}_${STAMP}.log" 2>&1 &
  pids+=("$!")
  sleep "$SEED_STAGGER_SECONDS"
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

summary_count="$(find "$LAWAM/results/eval_runs/robotwin/$RESULT_NAME" -name summary.json -type f | wc -l)"
echo "public pi0.5 same-bridge eval finished status=$status summaries=$summary_count stamp=$STAMP"
exit "$status"
