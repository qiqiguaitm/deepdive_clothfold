#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P345_VERIFY_REPO:-$REPO}
EVAL_REPO=${P3_EVAL_REPO:-$REPO}
SEED=${SEED:?set SEED to 1001 or 1002}
PROTOCOL=$VERIFY_REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p3_protocol.json

case "$SEED" in 1001|1002) ;; *) echo "P3 seed must be 1001 or 1002" >&2; exit 2;; esac
python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$VERIFY_REPO" --manifest "$PROTOCOL" --phase p3

CKPT=${CKPT:-$EVAL_REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed${SEED}/49999}
RESULT_NAME=pi05_predictive_adapter_p3_a0_seed${SEED}
exec env REPO="$EVAL_REPO" PREDICTIVE_P1_CONDITION=a0 CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" PORT_BASE_OFFSET="${PORT_BASE_OFFSET:-22400}" \
  LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-4}" MAX_PARALLEL_SEEDS="${MAX_PARALLEL_SEEDS:-4}" \
  bash "$EVAL_REPO/train_scripts/kai/eval/run_pi05_predictive_adapter_p1_formal.sh"
