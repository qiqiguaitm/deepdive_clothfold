#!/usr/bin/env bash
set -euo pipefail
umask 0000

REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM=$REPO/lmvla/lawam
VARIANT="${ALL6_EVAL_VARIANT:-combo}"
RUN_BASENAME="${ALL6_EVAL_RUN_BASENAME:-20260730_152020+robotwin_all6_v2_combo_seed2026}"
TRAIN_SEED="${ALL6_TRAIN_SEED:-2026}"
if [ "$RUN_BASENAME" = latest ]; then
  CKPT_GLOB="$LAWAM/results/Checkpoints/robotwin"/*+"robotwin_all6_v2_${VARIANT}_seed${TRAIN_SEED}"/final_model/pytorch_model.pt
  CKPT=$(ls -1t $CKPT_GLOB 2>/dev/null | head -1)
  test -n "$CKPT" || { echo "No completed checkpoint for $VARIANT seed$TRAIN_SEED" >&2; exit 3; }
  RUN=$(dirname "$(dirname "$CKPT")")
  RUN_BASENAME=$(basename "$RUN")
else
  RUN=$LAWAM/results/Checkpoints/robotwin/$RUN_BASENAME
  CKPT=$RUN/final_model/pytorch_model.pt
fi
MS=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_v2
PY=$REPO/kai0/.venv/bin/python
SERVER_PY=$REPO/train_scripts/kai/eval/lawam_server_python.sh
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=$LAWAM/logs/local_rteval
mkdir -p "$LOG_DIR"

test -f "$CKPT"
test -f "$RUN/config.yaml"
test -f "$RUN/dataset_statistics.json"
test -f "$MS/lmwm.pt"

export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-$SERVER_PY}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}"
export ROBOTWIN_TASKS="${ROBOTWIN_TASKS:-beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export ROBOTWIN_SAVE_VIDEO="${ROBOTWIN_SAVE_VIDEO:-0}"
export ROBOTWIN_INSTRUCTION_TYPE="${ROBOTWIN_INSTRUCTION_TYPE:-unseen}"
export PORT_SEARCH_LIMIT=30
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
unset ROBOTWIN_EXTRA_SITE

export LMWM_CKPT=$MS/lmwm.pt
export LMWM_ADAPTER_DIR=$REPO/lmvla/lmwam/adapter
export LMWM_SWAP_TEACHER=1
export LMWM_FEAT_STRIDE=1
export LMWM_HINT_DROPOUT=0.15
export LMWM_DUAL=1
export LMWM_DUAL_2Q=1
unset LMWM_MS_RESIDUAL
unset LMWM_MS_DETACH_BACKBONE
case "$VARIANT" in
  combo)
    export LMWM_MS_RESIDUAL=1
    export LMWM_MS_DETACH_BACKBONE=1
    ;;
  residual)
    export LMWM_MS_RESIDUAL=1
    ;;
  isolation)
    export LMWM_MS_DETACH_BACKBONE=1
    ;;
  absolute)
    ;;
  local|nowm|nowm_resetflow|neverwm)
    unset LMWM_CKPT LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
    unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_RESIDUAL
    unset LMWM_MS_DETACH_BACKBONE
    ;;
  *)
    echo "Unsupported ALL6_EVAL_VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

SEEDS="${SEEDS:-0 1}"
GPUS_PER_SEED="${GPUS_PER_SEED:-1}"
LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-2}"
GPU_INDEX_OFFSET="${GPU_INDEX_OFFSET:-0}"
RESULT_NAME="${RESULT_NAME:-rt_all6_v2_${VARIANT}_seed${TRAIN_SEED}}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-local-${VARIANT}-s${TRAIN_SEED}}"
pids=""
status=0
seed_index=0
for seed in $SEEDS; do
  gpu_start=$((GPU_INDEX_OFFSET + (seed_index * GPUS_PER_SEED) % LOCAL_GPU_COUNT))
  gpu_ids=$(seq -s, "$gpu_start" $((gpu_start + GPUS_PER_SEED - 1)))
  (
    export CUDA_VISIBLE_DEVICES=$gpu_ids
    export GPU_IDS=$gpu_ids
    export SEED=$seed
    export PORT_BASE=$((12000 + seed * 100))
    export ROBOTWIN_CKPT_ALIAS=robotwin_all6_v2_${VARIANT}_seed${TRAIN_SEED}
    export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    chmod -R a+rwX "$ROBOTWIN_EVAL_ROOT" || true
    if [ "${ROBOTWIN_ATTACH_EXISTING:-0}" = 1 ]; then
      if [ -n "${ROBOTWIN_ATTACH_RUN_TAG:-}" ]; then
        mapfile -t scheduler_files < <(
          find "$ROBOTWIN_EVAL_ROOT" -type f \
            -path "*/${ROBOTWIN_ATTACH_RUN_TAG}/.task_scheduler.json"
        )
      else
        mapfile -t scheduler_files < <(
          find "$ROBOTWIN_EVAL_ROOT" -type f -name .task_scheduler.json
        )
      fi
      if [ "${#scheduler_files[@]}" -ne 1 ]; then
        echo "attach requires exactly one matching scheduler under $ROBOTWIN_EVAL_ROOT; found ${#scheduler_files[@]}" >&2
        exit 4
      fi
      export ROBOTWIN_RESUME_RUN_DIR
      ROBOTWIN_RESUME_RUN_DIR=$(dirname "${scheduler_files[0]}")
      export ROBOTWIN_ATTACH_SCHEDULER=1
      export ROBOTWIN_WORKER_INDEX_OFFSET="${ROBOTWIN_WORKER_INDEX_OFFSET:-1000}"
      pending_count=$(
        "$PY" -c 'import json,os,sys; d=json.load(open(sys.argv[1])); retry=os.getenv("ROBOTWIN_ATTACH_REQUEUE_FAILED", "0").lower() in {"1","true","yes","on"}; print(len(d.get("pending", [])) + (len(d.get("failed", {})) if retry else 0))' \
          "$ROBOTWIN_RESUME_RUN_DIR/.task_scheduler.json"
      )
      if [ "$pending_count" -eq 0 ]; then
        echo "attach seed=$seed has no pending tasks; skipping model load"
        exit 0
      fi
    fi
    cd "$LAWAM"
    bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
      "$CKPT" "$TASK_CONFIG" "${RUN_TAG_PREFIX}-seed$seed"
  ) > "$LOG_DIR/all6_${VARIANT}_s${TRAIN_SEED}_seed${seed}_${STAMP}.log" 2>&1 &
  pids="$pids $!"
  seed_index=$((seed_index + 1))
  sleep 20
done

for pid in $pids; do
  wait "$pid" || status=1
done
echo "local all6 $VARIANT seed${TRAIN_SEED} eval finished status=$status stamp=$STAMP"
exit "$status"
