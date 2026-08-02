#!/usr/bin/env bash
set -euo pipefail

# Keep a running evaluation stable when the repository copy is updated.
if [[ "${A2_EVAL_SCRIPT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-a2-official-eval.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  A2_EVAL_SCRIPT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=/vePFS/tim/workspace/deepdive_kai0
LAWAM=$REPO/lmvla/lawam
PY=$REPO/kai0/.venv/bin/python
ROBOTWIN_PY=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
TASKS="${ROBOTWIN_TASKS:-beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three}"
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=$LAWAM/logs/local_rteval
mkdir -p "$LOG_DIR"

export STAR_VLA_PYTHON=$PY
export OPENPI_DATA_HOME="$REPO/openpi_cache"
export OPENPI_SERVE_SCRIPT=$REPO/kai0/scripts/serve_policy.py
export KAI0_ROOT=$REPO/kai0
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_OPENPI_CONFIG=pi05_robotwin_a2_prefix_official_eval_bj
export ROBOTWIN_HINT_ENCODER=so400m
export OPENPI_SERVER_HINT_ENCODER=so400m
export EVAL_HINT_RESIDUAL=0
export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
export ROBOTWIN_PYTHON=$ROBOTWIN_PY
export ROBOTWIN_EXTRA_SITE=/vePFS/HuanQian/conda_envs/abot-physworld/lib/python3.10/site-packages
export PYTHONPATH="$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export PORT_SEARCH_LIMIT=30
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export TORCH_CUDA_ARCH_LIST=8.0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export OMP_NUM_THREADS=4

CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a2_prefix_official_bj/pi05_robotwin_a2_prefix_official/19999
test -d "$CKPT/params"
test -f "$CKPT/assets/robotwin2.0/norm_stats.json"

SEEDS="${SEEDS:-0 1 2 3}"
SEED_STAGGER_SECONDS="${SEED_STAGGER_SECONDS:-20}"
LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-2}"
EVAL_NAME="${EVAL_NAME:-pi05_rt_a2_prefix_official_local2g}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-local-unseen-a2}"
pids=""
status=0

for seed in $SEEDS; do
  gpu=$((seed % LOCAL_GPU_COUNT))
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export GPU_IDS=$gpu
    export SEED=$seed
    export PORT_BASE=$((10000 + seed * 40))
    export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$EVAL_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$LAWAM"
    bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
      "$CKPT" "$TASK_CONFIG" "${RUN_TAG_PREFIX}-seed$seed"
  ) > "$LOG_DIR/A2_seed${seed}_${STAMP}.log" 2>&1 &
  pids="$pids $!"
  sleep "$SEED_STAGGER_SECONDS"
done

for pid in $pids; do
  wait "$pid" || status=1
done
echo "local Robotwin A2 eval finished status=$status stamp=$STAMP"
exit "$status"
