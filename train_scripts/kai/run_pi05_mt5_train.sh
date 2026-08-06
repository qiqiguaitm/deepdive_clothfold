#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${ARM:?set ARM to mt5_local or mt5_combined}
SEED=${SEED:-1000}
STEPS=${STEPS:-50000}
WORKERS=${WORKERS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
DATA_REPO=${PI05_EXACT_DATASET:-/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21}
INIT_PARAMS=${PI05_BASE_PARAMS:-$REPO/kai0/checkpoints/pi05_base/params}
TARGET_PAIRS=${MT5_TARGET_PAIRS:-$REPO/lmvla/lmwm/data/robotwin_fixed_horizon_1s_v1/pairs.npz}
FRAME_CACHE=${MT5_FRAME_CACHE:-$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256}
TRANSITION_PAIRS=${TRANSITION_PAIRS:-$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz}

case "$ARM" in
  mt5_local)
    CONFIG=${CONFIG:-pi05_robotwin_mt5_local_exact}
    EXP=${EXP:-pi05_robotwin_mt5_local_seed${SEED}}
    ;;
  mt5_combined)
    CONFIG=${CONFIG:-pi05_robotwin_mt5_combined_exact}
    EXP=${EXP:-pi05_robotwin_mt5_combined_seed${SEED}}
    SELECTION=${MT3_SELECTION:-$REPO/logs/mt_stage_tracker/selection.json}
    TRACKER_CANDIDATE=${TRACKER_CANDIDATE:-$($REPO/kai0/.venv/bin/python -c "import json; print(json.load(open('$SELECTION'))['selected'])")}
    TRACKER_CHECKPOINT=${TRACKER_CHECKPOINT:-$REPO/logs/mt_stage_tracker/$TRACKER_CANDIDATE/tracker.pt}
    test -f "$TRACKER_CHECKPOINT"
    ;;
  *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac

ASSET_ID=robotwin2.0_absolute_meanstd
NORM=$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj/$ASSET_ID/norm_stats.json
mkdir -p "$REPO/kai0/assets/$CONFIG/$ASSET_ID"
cp -f "$NORM" "$REPO/kai0/assets/$CONFIG/$ASSET_ID/norm_stats.json"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-mt5-${USER:-tim}}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CKPT_ROOT=$REPO/kai0/checkpoints/$CONFIG/$EXP
RESUME_ARGS=()
if find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null | grep -q .; then
  RESUME_ARGS=(--resume)
fi

EXTRA_ARGS=()
if [[ "$ARM" == mt5_combined ]]; then
  EXTRA_ARGS+=(
    --transition-pairs "$TRANSITION_PAIRS"
    --tracker-candidate "$TRACKER_CANDIDATE"
    --tracker-checkpoint "$TRACKER_CHECKPOINT"
  )
fi

cd "$REPO/kai0"
exec .venv/bin/python -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm "$ARM" \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed "$SEED" \
  --data-repo "$DATA_REPO" \
  --init-params "$INIT_PARAMS" \
  --asset-id "$ASSET_ID" \
  --target-pairs "$TARGET_PAIRS" \
  --frame-cache-root "$FRAME_CACHE" \
  --num-train-steps "$STEPS" \
  --num-workers "$WORKERS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 100 \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
