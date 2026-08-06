#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p5_protocol.json
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
RESULT_NAME=pi05_predictive_adapter_p5_public_paired
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json

python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$REPO" --manifest "$PROTOCOL" --phase p5
for seeds in "0 1" "2 3"; do
  env PUBLIC_PI05_REPO="$REPO" PUBLIC_PI05_LOG_DIR="$REPO/logs/predictive/p5_runtime" \
    SEEDS="$seeds" LOCAL_GPU_COUNT=2 \
    RESULT_NAME="$RESULT_NAME" RUN_TAG_PREFIX=p5-public-paired \
    ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
    bash "$REPO/train_scripts/kai/eval/run_pi05_public_samebridge_multiseed.sh"
done
python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$RESULT_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
exec bash "$REPO/train_scripts/kai/analysis/finalize_pi05_predictive_adapter_p5.sh"
