#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
VERIFY_REPO=$REPO/logs/frozen_source_overlays/pi05_r1_v1
CKPT=${CKPT:?missing CKPT}
RESULT_NAME=pi05_predictive_adapter_p1_seed1000_a0
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
MARKER=$REPO/logs/resource_markers/${RESULT_NAME}.ok
ACCELERATOR_MARKER=$REPO/logs/resource_markers/pi05_p1_a0_north_tail_accelerator.ok
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
LOG_DIR=$REPO/logs/predictive/p1_platform_north
STAMP=$(date -u +%Y%m%d_%H%M%SZ)

mkdir -p "$LOG_DIR" "$(dirname "$ACCELERATOR_MARKER")" /home/tim/workspace
exec >>"$LOG_DIR/a0_tail_accelerator_${STAMP}.log" 2>&1
set -x

export REPO
export OPENPI_DATA_HOME=$REPO/openpi_cache
export ROBOTWIN_PATH=/vePFS-North-E/vis_robot/huanqian/RoboTwin
export ROBOTWIN_PYTHON=$REPO/lmvla/lawam/robotwin_python_wrapper_northe.sh
export PYTHONPATH=$VERIFY_REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/vePFS-North-E/vis_robot/tim/runtime/torch_extensions/h20_sm90_py310
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export ROBOTWIN_WORKER_INDEX_OFFSET=3000
mkdir -p "$TORCH_EXTENSIONS_DIR"
ln -sfn "$REPO" /home/tim/workspace/deepdive_kai0

test -s "$REPO/logs/resource_markers/pi05_p1_north_eval_stage.ok"
test -s "$VERIFY_REPO/READY"
test -x "$ROBOTWIN_PYTHON"
test -f "$CKPT/params/_METADATA"
test -f "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json"
test "$(find "$RESULT_ROOT" -name .task_scheduler.json -type f | wc -l)" -eq 4

pending=$(python3 - "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
schedulers = sorted(root.glob("seed*/*/local-unseen-a3-seed*/.task_scheduler.json"))
assert len(schedulers) == 4
print(sum(len(json.loads(path.read_text()).get("pending", [])) for path in schedulers))
PY
)
if (( pending == 0 )); then
  printf 'completed=%s\ncondition=a0\nworker_offset=3000\nskipped=no_pending_cells\n' \
    "$(date -u +%FT%TZ)" >"$ACCELERATOR_MARKER"
  exit 0
fi
if (( pending < 0 || pending > 4 )); then
  echo "unexpected pending cell count: $pending" >&2
  exit 13
fi
echo "pending_cells=$pending"

"$REPO/kai0/.venv/bin/python" - <<'PY'
import pathlib
import openpi

expected = pathlib.Path(
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/"
    "logs/frozen_source_overlays/pi05_r1_v1/kai0/src/openpi"
).resolve()
actual = pathlib.Path(openpi.__file__).resolve()
if expected not in actual.parents:
    raise SystemExit(f"frozen openpi overlay not active: {actual}")
print(f"frozen_openpi={actual}")
PY

env \
  PI05_EVAL_CONFIG_NAME=pi05_robotwin_a0_public_exact_bj \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  CKPT="$CKPT" RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TEST_NUM=50 ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 SEEDS="0 1 2 3" \
  LOCAL_GPU_COUNT=4 GPU_INDEX_OFFSET=0 MAX_PARALLEL_SEEDS=4 \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-26600} \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$RESULT_ROOT"
tmp=$REPORT.tail.tmp.$$
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$tmp"
mv "$tmp" "$REPORT"
printf 'validated=%s\ncondition=a0\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CKPT" "$REPORT" >"$MARKER"
printf 'completed=%s\ncondition=a0\nworker_offset=3000\ncanonical_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$MARKER" >"$ACCELERATOR_MARKER"
