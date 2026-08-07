#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
STAGE=${P3_EVAL_REPO:-$ROOT/.staging/pi05_p1_failover_20260804T1034Z}
VERIFY_REPO=${P345_VERIFY_REPO:-$ROOT}
MANIFEST=$STAGE/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
MARKER=$STAGE/logs/resource_markers/pi05_predictive_adapter_p3_north_accelerator.ok
LOG_DIR=$STAGE/logs/predictive/p3_north_accelerator
STAMP=$(date -u +%Y%m%d_%H%M%SZ)
WAIT_SECONDS=${P3_ACCELERATOR_WAIT_SECONDS:-7200}

mkdir -p "$LOG_DIR" "$(dirname "$MARKER")"
exec >>"$LOG_DIR/launcher_${STAMP}.log" 2>&1
set -x

export OPENPI_DATA_HOME=$ROOT/openpi_cache
export ROBOTWIN_PATH=/vePFS-North-E/vis_robot/huanqian/RoboTwin
export ROBOTWIN_PYTHON=$ROOT/lmvla/lawam/robotwin_python_wrapper_northe.sh
export PYTHONPATH=$STAGE/kai0/src:$ROOT/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/vePFS-North-E/vis_robot/tim/runtime/torch_extensions/h20_sm90_py310
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export ROBOTWIN_NUM_SLOTS=1

python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$VERIFY_REPO" \
  --manifest "$VERIFY_REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p3_protocol.json" \
  --phase p3

wait_for_scheduler() {
  local path=$1
  local canonical_marker=$2
  local waited=0
  while [[ ! -f "$path" ]]; do
    if [[ -f "$canonical_marker" ]]; then
      return 1
    fi
    if (( waited >= WAIT_SECONDS )); then
      echo "timed out waiting for P3 scheduler: $path" >&2
      return 2
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

run_lane() {
  local training_seed=$1
  local eval_seed=$2
  local gpu=$3
  local result_name=pi05_predictive_adapter_p3_a0_seed${training_seed}
  local checkpoint=$STAGE/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed${training_seed}/49999
  local canonical_marker=$STAGE/logs/resource_markers/${result_name}.ok
  local run_tag=local-unseen-a3-seed${eval_seed}
  local scheduler=$STAGE/lmvla/lawam/results/eval_runs/robotwin/$result_name/seed${eval_seed}/pi05_predictive_adapter_p1_a0_exact__demo_clean/$run_tag/.task_scheduler.json

  if ! wait_for_scheduler "$scheduler" "$canonical_marker"; then
    [[ -f "$canonical_marker" ]] && return 0
    return 1
  fi
  [[ -f "$canonical_marker" ]] && return 0
  test -f "$checkpoint/params/_METADATA"

  env \
    REPO="$STAGE" \
    SEED="$eval_seed" \
    GPU_INDEX="$gpu" \
    WORKER_INDEX_OFFSET="$((32000 + training_seed * 20 + eval_seed))" \
    PORT_BASE_OFFSET="$((30000 + gpu * 100))" \
    RESULT_NAME="$result_name" \
    ROBOTWIN_ATTACH_RUN_TAG="$run_tag" \
    PI05_EVAL_CONFIG_NAME=pi05_robotwin_a0_public_exact_bj \
    PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
    CKPT="$checkpoint" \
    EVAL_WORKERS_PER_GPU=1 \
    ATTACH_MARKER_NAME="pi05_p3_north_accelerator_train${training_seed}_eval${eval_seed}" \
    bash "$STAGE/train_scripts/kai/eval/attach_pi05_a0_confirmatory_local.sh"
}

pids=()
status=0
gpu=0
for training_seed in 1001 1002; do
  for eval_seed in 0 1 2 3; do
    run_lane "$training_seed" "$eval_seed" "$gpu" \
      >"$LOG_DIR/train${training_seed}_eval${eval_seed}_${STAMP}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
    sleep 5
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
test "$status" -eq 0

printf 'completed=%s\ntraining_seeds=1001,1002\nevaluation_seeds=0,1,2,3\nhelper_workers=8\n' \
  "$(date -u +%FT%TZ)" >"$MARKER"
