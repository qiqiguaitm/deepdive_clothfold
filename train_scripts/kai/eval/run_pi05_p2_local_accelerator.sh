#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P2_VERIFY_REPO:-$REPO/logs/frozen_source_overlays/pi05_replication_v1}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
MARKER=$REPO/logs/resource_markers/pi05_predictive_adapter_p2_local_accelerator.ok
LOG_DIR=$REPO/logs/predictive/p2_local_accelerator
STAMP=$(date -u +%Y%m%d_%H%M%SZ)

mkdir -p "$LOG_DIR" "$(dirname "$MARKER")"
exec >>"$LOG_DIR/launcher_${STAMP}.log" 2>&1
set -x

export REPO
export OPENPI_DATA_HOME=$REPO/openpi_cache
export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
export ROBOTWIN_PYTHON=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
export PYTHONPATH=$VERIFY_REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR=/vePFS/tim/runtime/torch_extensions/a100_sm80_py310
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export PREDICTIVE_ACTION_INTERVENTION=normal
export ROBOTWIN_NUM_SLOTS=1
export ROBOTWIN_ATTACH_REQUEUE_FAILED=1

test -s "$VERIFY_REPO/REPLICATION_READY"
test -x "$ROBOTWIN_PYTHON"
test -f "$MANIFEST"
test -d "$TORCH_EXTENSIONS_DIR"

"$REPO/kai0/.venv/bin/python" - "$VERIFY_REPO" <<'PY'
import pathlib
import sys

import openpi

expected = pathlib.Path(sys.argv[1]).resolve() / "kai0/src/openpi"
actual = pathlib.Path(openpi.__file__).resolve()
if expected not in actual.parents:
    raise SystemExit(f"frozen openpi overlay not active: {actual}")
print(f"frozen_openpi={actual}")
PY

run_lane() {
  local training_seed=$1
  local gpu=$2
  local worker_offset=$3
  local port_offset=$4
  shift 4

  local result_name=pi05_predictive_adapter_p2_seed${training_seed}_normal
  local checkpoint=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed${training_seed}/49999
  local canonical_marker=$REPO/logs/resource_markers/${result_name}.ok
  test -f "$checkpoint/params/_METADATA"
  test -f "$checkpoint/assets/robotwin2.0_absolute_meanstd/norm_stats.json"

  local eval_seed
  for eval_seed in "$@"; do
    if [[ -f "$canonical_marker" ]]; then
      return 0
    fi
    local scheduler=$REPO/lmvla/lawam/results/eval_runs/robotwin/$result_name/seed${eval_seed}/pi05_predictive_adapter_p1__demo_clean/local-unseen-a3-seed${eval_seed}/.task_scheduler.json
    test -f "$scheduler"
    env \
      SEED="$eval_seed" \
      GPU_INDEX="$gpu" \
      WORKER_INDEX_OFFSET="$worker_offset" \
      PORT_BASE_OFFSET="$port_offset" \
      RESULT_NAME="$result_name" \
      ROBOTWIN_ATTACH_RUN_TAG="local-unseen-a3-seed${eval_seed}" \
      PI05_EVAL_CONFIG_NAME=pi05_predictive_adapter_p1_eval \
      PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
      CKPT="$checkpoint" \
      EVAL_WORKERS_PER_GPU=1 \
      ATTACH_MARKER_NAME="pi05_p2_local_accelerator_train${training_seed}_eval${eval_seed}" \
      bash "$REPO/train_scripts/kai/eval/attach_pi05_a0_confirmatory_local.sh"
  done
}

pids=()
status=0
run_lane 1001 0 6000 28000 0 2 >"$LOG_DIR/train1001_lane0_${STAMP}.log" 2>&1 &
pids+=("$!")
sleep 10
run_lane 1001 0 6100 28400 1 3 >"$LOG_DIR/train1001_lane1_${STAMP}.log" 2>&1 &
pids+=("$!")
sleep 10
run_lane 1002 1 7000 30000 0 2 >"$LOG_DIR/train1002_lane0_${STAMP}.log" 2>&1 &
pids+=("$!")
sleep 10
run_lane 1002 1 7100 30400 1 3 >"$LOG_DIR/train1002_lane1_${STAMP}.log" 2>&1 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
test "$status" -eq 0

printf 'completed=%s\ntraining_seeds=1001,1002\nworkers_per_gpu=2\nslots_per_worker=1\n' \
  "$(date -u +%FT%TZ)" >"$MARKER"
