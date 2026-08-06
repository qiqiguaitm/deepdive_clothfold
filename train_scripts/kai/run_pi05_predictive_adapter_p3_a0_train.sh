#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P345_VERIFY_REPO:-$REPO}
TRAIN_REPO=${P3_TRAIN_REPO:-$REPO}
SEED=${SEED:?set SEED to 1001 or 1002}
PROTOCOL=$VERIFY_REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p3_protocol.json

case "$SEED" in 1001|1002) ;; *) echo "P3 seed must be 1001 or 1002" >&2; exit 2;; esac
python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$VERIFY_REPO" --manifest "$PROTOCOL" --phase p3

exec env \
  REPO="$TRAIN_REPO" \
  TRAIN_VERIFY_REPO="$TRAIN_REPO" \
  TRAIN_SOURCE_REPO="$TRAIN_REPO" \
  PYTHONPATH="$TRAIN_REPO/kai0/src:$TRAIN_REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}" \
  DATA_REPO="$TRAIN_REPO/datasets/robotwin2.0_official_prompts_v21" \
  PYTHON_BIN="${PYTHON_BIN:-$REPO/kai0/.venv/bin/python}" \
  ARM=a0 SEED="$SEED" STEPS=50000 SAVE_INTERVAL=5000 WORKERS="${WORKERS:-16}" \
  bash "$TRAIN_REPO/train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
