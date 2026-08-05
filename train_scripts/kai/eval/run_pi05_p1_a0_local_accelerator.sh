#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P1_VERIFY_REPO:-$REPO/logs/frozen_source_overlays/pi05_r1_v1}
RESULT_NAME=pi05_predictive_adapter_p1_seed1000_a0
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
MARKER=$REPO/logs/resource_markers/${RESULT_NAME}.ok
ACCELERATOR_MARKER=$REPO/logs/resource_markers/pi05_p1_a0_local_accelerator.ok
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999}
STAMP=$(date -u +%Y%m%d_%H%M%SZ)
LOG_DIR=$REPO/logs/predictive/p1_a0_local_accelerator

mkdir -p "$LOG_DIR" "$(dirname "$ACCELERATOR_MARKER")"
exec >>"$LOG_DIR/launcher_${STAMP}.log" 2>&1
set -x

export REPO
export OPENPI_DATA_HOME=$REPO/openpi_cache
export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
export ROBOTWIN_PYTHON=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
export PYTHONPATH=$VERIFY_REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export TORCH_CUDA_ARCH_LIST=8.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22

test -s "$VERIFY_REPO/READY"
test -x "$ROBOTWIN_PYTHON"
test -f "$CKPT/params/_METADATA"
test -f "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json"
for seed in 2 3; do
  test -f "$RESULT_ROOT/seed${seed}/pi05_predictive_adapter_p1_a0_exact__demo_clean/local-unseen-a3-seed${seed}/.task_scheduler.json"
done

"$REPO/kai0/.venv/bin/python" - <<'PY'
import pathlib
import openpi

expected = pathlib.Path(
    "/vePFS/tim/workspace/deepdive_kai0/"
    "logs/frozen_source_overlays/pi05_r1_v1/kai0/src/openpi"
).resolve()
actual = pathlib.Path(openpi.__file__).resolve()
if expected not in actual.parents:
    raise SystemExit(f"frozen openpi overlay not active: {actual}")
print(f"frozen_openpi={actual}")
PY

pids=()
status=0
worker_specs=(
  "2 0 4000 26400"
  "3 1 5000 26800"
)
for spec in "${worker_specs[@]}"; do
  read -r seed gpu worker port <<<"$spec"
  env SEED="$seed" GPU_INDEX="$gpu" WORKER_INDEX_OFFSET="$worker" \
    PORT_BASE_OFFSET="$port" RESULT_NAME="$RESULT_NAME" \
    ROBOTWIN_ATTACH_RUN_TAG="local-unseen-a3-seed${seed}" \
    PI05_EVAL_CONFIG_NAME=pi05_robotwin_a0_public_exact_bj \
    PI05_ASSET_ID=robotwin2.0_absolute_meanstd CKPT="$CKPT" \
    EVAL_WORKERS_PER_GPU=2 \
    ATTACH_MARKER_NAME="pi05_p1_a0_local_accelerator_s${seed}" \
    bash "$REPO/train_scripts/kai/eval/attach_pi05_a0_confirmatory_local.sh" \
    >"$LOG_DIR/seed${seed}_workers${worker}_$((worker + 1))_${STAMP}.log" 2>&1 &
  pids+=("$!")
  sleep 15
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
test "$status" -eq 0

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$RESULT_ROOT"
tmp=$REPORT.local-accelerator.tmp.$$
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$tmp"
mv "$tmp" "$REPORT"
printf 'validated=%s\ncondition=a0\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CKPT" "$REPORT" >"$MARKER"
printf 'completed=%s\nseeds=2,3\nworkers_per_gpu=2\ncanonical_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$MARKER" >"$ACCELERATOR_MARKER"
