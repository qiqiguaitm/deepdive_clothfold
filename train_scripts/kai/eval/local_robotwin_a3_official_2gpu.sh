#!/usr/bin/env bash
set -euo pipefail

# Keep the running body stable even when the shared repository is updated.
if [[ "${A3_EVAL_SCRIPT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-a3-official-eval.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  A3_EVAL_SCRIPT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
PY=$REPO/kai0/.venv/bin/python
ROBOTWIN_PY=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
TASKS="${ROBOTWIN_TASKS:-beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three}"
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=${LOCAL_EVAL_LOG_DIR:-$LAWAM/logs/local_rteval}
mkdir -p "$LOG_DIR"

A1_RECOVERY_MARKER="${A1_RECOVERY_MARKER:-/tmp/robotwin-local-a1-recovery.running}"
while [[ -e "$A1_RECOVERY_MARKER" ]]; do
  echo "waiting for local A1 recovery eval: $A1_RECOVERY_MARKER"
  sleep 60
done

export STAR_VLA_PYTHON=$PY
export OPENPI_DATA_HOME="$REPO/openpi_cache"
export OPENPI_SERVE_SCRIPT=$REPO/kai0/scripts/serve_policy.py
export KAI0_ROOT=$REPO/kai0
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_OPENPI_CONFIG="${PI05_EVAL_CONFIG_NAME:-pi05_robotwin_a3_live_residual_prefix_official_eval}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$ROBOTWIN_PY}"
export PYTHONPATH="$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}"
EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-${NUM_WORKERS:-1}}"
if ! [[ "$EVAL_WORKERS_PER_GPU" =~ ^[12]$ ]]; then
  echo "EVAL_WORKERS_PER_GPU must be 1 or 2, got: $EVAL_WORKERS_PER_GPU" >&2
  exit 2
fi
export NUM_WORKERS="$EVAL_WORKERS_PER_GPU"
if (( EVAL_WORKERS_PER_GPU > 1 )); then
  export ALLOW_GPU_OVERSUBSCRIBE=1
fi
export PORT_SEARCH_LIMIT=30
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export OMP_NUM_THREADS=4

CKPT="${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_a3_live_residual_prefix_official_east/pi05_robotwin_a3_live_residual_prefix_official/19999}"
ASSET_ID="${PI05_ASSET_ID:-robotwin2.0}"
test -d "$CKPT/params"
test -f "$CKPT/assets/$ASSET_ID/norm_stats.json"

SEEDS="${SEEDS:-0 1 2 3}"
SEED_STAGGER_SECONDS="${SEED_STAGGER_SECONDS:-20}"
GPUS_PER_SEED="${GPUS_PER_SEED:-1}"
LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-$((4 * GPUS_PER_SEED))}"
GPU_INDEX_OFFSET="${GPU_INDEX_OFFSET:-0}"
MAX_PARALLEL_SEEDS="${MAX_PARALLEL_SEEDS:-0}"
RESULT_NAME="${RESULT_NAME:-pi05_rt_a3_live_residual_official_local2g}"
ROBOTWIN_EVAL_RUN_TAG_PREFIX="${ROBOTWIN_EVAL_RUN_TAG_PREFIX:-local-unseen-a3-seed}"
gpu_slots=$((LOCAL_GPU_COUNT / GPUS_PER_SEED))
(( gpu_slots > 0 )) || { echo "LOCAL_GPU_COUNT must be >= GPUS_PER_SEED" >&2; exit 2; }
pids=""
active=0
seed_index=0
status=0

for seed in $SEEDS; do
  gpu_slot=$((seed_index % gpu_slots))
  gpu_start=$((GPU_INDEX_OFFSET + gpu_slot * GPUS_PER_SEED))
  gpu_ids=$(seq -s, "$gpu_start" $((gpu_start + GPUS_PER_SEED - 1)))
  (
    export CUDA_VISIBLE_DEVICES=$gpu_ids
    export GPU_IDS=$gpu_ids
    export SEED=$seed
    export PORT_BASE=$((${PORT_BASE_OFFSET:-9600} + seed * 40))
    export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$LAWAM"
    run_tag="${ROBOTWIN_EVAL_RUN_TAG_PREFIX}${seed}"
    shopt -s nullglob
    scheduler_paths=("$ROBOTWIN_EVAL_ROOT"/*/"$run_tag"/.task_scheduler.json)
    shopt -u nullglob
    if [ "${#scheduler_paths[@]}" -eq 1 ]; then
      run_dir=$(dirname "${scheduler_paths[0]}")
      ROBOTWIN_ATTACH_SCHEDULER=1 \
        bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$CKPT" "$TASK_CONFIG" "$run_tag" "$run_dir"
    elif [ "${#scheduler_paths[@]}" -eq 0 ]; then
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$CKPT" "$TASK_CONFIG" "$run_tag"
    else
      echo "ambiguous schedulers for $run_tag: ${scheduler_paths[*]}" >&2
      exit 12
    fi
  ) > "$LOG_DIR/A3_seed${seed}_${STAMP}.log" 2>&1 &
  pids="$pids $!"
  active=$((active + 1))
  seed_index=$((seed_index + 1))
  sleep "$SEED_STAGGER_SECONDS"
  if (( MAX_PARALLEL_SEEDS > 0 && active >= MAX_PARALLEL_SEEDS )); then
    for pid in $pids; do
      wait "$pid" || status=1
    done
    pids=""
    active=0
  fi
done

for pid in $pids; do
  wait "$pid" || status=1
done
echo "local Robotwin A3 eval finished status=$status stamp=$STAMP"
exit "$status"
