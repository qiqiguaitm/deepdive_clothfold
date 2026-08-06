#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
CKPT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1001/49999
MANIFEST=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_east_h20_abi_preflight_seeds_v1.json
RESULT_NAME=pi05_predictive_adapter_p2_east_h20_abi_preflight_v1
RESULT_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME
MARKER=$REPO/logs/resource_markers/$RESULT_NAME.ok

test "${TORCH_CUDA_ARCH_LIST:?set TORCH_CUDA_ARCH_LIST}" = "9.0"
test "${TORCH_EXTENSIONS_DIR:?set TORCH_EXTENSIONS_DIR}" = \
  /vePFS/tim/runtime/torch_extensions/h20_sm90_py310
for extension in kinematics_fused_cu geom_cu tensor_step_cu lbfgs_step_cu line_search_cu; do
  test -s "$TORCH_EXTENSIONS_DIR/$extension/$extension.so"
done
test -s "$CKPT/params/_METADATA"
test -s "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json"
test -s "$MANIFEST"

env \
  PI05_EVAL_CONFIG_NAME=pi05_predictive_adapter_p1_eval \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  PREDICTIVE_ACTION_INTERVENTION=normal \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TASKS=beat_block_hammer \
  ROBOTWIN_TEST_NUM=1 \
  ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=5 \
  ROBOTWIN_ATTACH_REQUEUE_FAILED=1 \
  ROBOTWIN_EVAL_RUN_TAG_PREFIX=p2-east-h20-abi-seed \
  SEEDS=0 \
  LOCAL_GPU_COUNT=1 \
  MAX_PARALLEL_SEEDS=1 \
  PORT_BASE_OFFSET=22320 \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

mapfile -t summaries < <(find "$RESULT_ROOT" -type f -name summary.json | sort)
test "${#summaries[@]}" -eq 1
python3 - "${summaries[0]}" "$MARKER" "$CKPT" <<'PY'
import json
import os
from pathlib import Path
import sys

summary_path, marker_path, checkpoint = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text())
assert summary["task_name"] == "beat_block_hammer", summary
assert summary["n_episodes"] == 1, summary
assert len(summary["episodes"]) == 1, summary
assert summary["episodes"][0]["seed"] == 100000, summary
assert summary["model_queries"] > 0, summary
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
temporary.write_text(
    "validated=true\n"
    f"summary={summary_path.resolve()}\n"
    f"checkpoint={checkpoint.resolve()}\n"
    "torch_cuda_arch_list=9.0\n"
    "torch_extensions_dir=/vePFS/tim/runtime/torch_extensions/h20_sm90_py310\n"
)
temporary.replace(marker_path)
PY
