#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
LAWAM=$REPO/lmvla/lawam
PY=$REPO/kai0/.venv/bin/python
ROBOTWIN_PY=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
TASKS="beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=$LAWAM/logs/local_rteval
mkdir -p "$LOG_DIR"

export STAR_VLA_PYTHON=$PY
export OPENPI_SERVE_SCRIPT=$REPO/kai0/scripts/serve_policy.py
export KAI0_ROOT=$REPO/kai0
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
export ROBOTWIN_PYTHON=$ROBOTWIN_PY
export PYTHONPATH="$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=50
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
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

SEEDS="${SEEDS:-0 1 2 3}"
SEED_STAGGER_SECONDS="${SEED_STAGGER_SECONDS:-20}"
ARMS="${ARMS:-A0 A1}"
RESULT_SUFFIX="${RESULT_SUFFIX:-local2g_v2}"

run_arm() {
  local arm=$1
  local gpu=$2
  local config=$3
  local ckpt=$4
  local result_name=$5
  local hint_encoder=$6
  local port_base=$7
  local pids=""
  local status=0

  test -d "$ckpt/params"
  test -f "$ckpt/assets/robotwin2.0/norm_stats.json"
  for seed in $SEEDS; do
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      export GPU_IDS=$gpu
      export SEED=$seed
      export PORT_BASE=$((port_base + seed * 40))
      export ROBOTWIN_OPENPI_CONFIG=$config
      export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$result_name/seed$seed
      if [[ -n "$hint_encoder" ]]; then
        export ROBOTWIN_HINT_ENCODER=$hint_encoder
      else
        unset ROBOTWIN_HINT_ENCODER
      fi
      mkdir -p "$ROBOTWIN_EVAL_ROOT"
      cd "$LAWAM"
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$ckpt" "$TASK_CONFIG" "local-unseen-$arm-seed$seed"
    ) > "$LOG_DIR/${arm}_seed${seed}_${STAMP}.log" 2>&1 &
    pids="$pids $!"
    sleep "$SEED_STAGGER_SECONDS"
  done

  for pid in $pids; do
    wait "$pid" || status=1
  done
  return "$status"
}

A0_CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a0_official_bj/pi05_robotwin_a0_official/19999
A1_CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a1_prefix_official_bj/pi05_robotwin_a1_prefix_official/19999

status=0
arm_pids=""
if [[ " $ARMS " == *" A0 "* ]]; then
  run_arm A0 0 pi05_robotwin_a0_official_bj \
    "$A0_CKPT" "pi05_rt_a0_official_$RESULT_SUFFIX" "" 8400 &
  arm_pids="$arm_pids $!"
fi
if [[ " $ARMS " == *" A1 "* ]]; then
  run_arm A1 1 pi05_robotwin_a1_prefix_official_eval_bj \
    "$A1_CKPT" "pi05_rt_a1_prefix_official_$RESULT_SUFFIX" dinov3-base 8800 &
  arm_pids="$arm_pids $!"
fi
for pid in $arm_pids; do
  wait "$pid" || status=1
done
echo "local Robotwin A0/A1 eval finished status=$status stamp=$STAMP"
exit "$status"
