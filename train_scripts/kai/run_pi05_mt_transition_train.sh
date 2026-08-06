#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${ARM:?set ARM to mt1_oracle, mt2_null, or mt3_learned}
SEED=${SEED:-1000}
STEPS=${STEPS:-50000}
WORKERS=${WORKERS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
CONFIG=${CONFIG:-pi05_robotwin_${ARM}_exact}
EXP=${EXP:-pi05_robotwin_${ARM}_seed${SEED}}
TRANSITION_PAIRS=${TRANSITION_PAIRS:-$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz}
DATA_REPO=${PI05_EXACT_DATASET:-/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21}
INIT_PARAMS=${PI05_BASE_PARAMS:-$REPO/kai0/checkpoints/pi05_base/params}

case "$ARM" in
  mt1_oracle|mt2_null) ;;
  mt3_learned)
    TRACKER_CHECKPOINT=${TRACKER_CHECKPOINT:?set TRACKER_CHECKPOINT for mt3_learned}
    TRACKER_CANDIDATE=${TRACKER_CANDIDATE:?set TRACKER_CANDIDATE for mt3_learned}
    case "$TRACKER_CANDIDATE" in current_frame|history_proprio) ;; *) exit 2 ;; esac
    ;;
  *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac

mkdir -p "$REPO/kai0/assets/$CONFIG/robotwin2.0_absolute_meanstd"
cp -f \
  "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj/robotwin2.0_absolute_meanstd/norm_stats.json" \
  "$REPO/kai0/assets/$CONFIG/robotwin2.0_absolute_meanstd/norm_stats.json"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-cache-${USER:-tim}}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CKPT_ROOT="$REPO/kai0/checkpoints/$CONFIG/$EXP"
RESUME_ARGS=()
if find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null | grep -q .; then
  RESUME_ARGS=(--resume)
fi

cd "$REPO/kai0"
EXTRA_ARGS=()
if [[ "$ARM" == mt3_learned ]]; then
  EXTRA_ARGS+=(--tracker-checkpoint "$TRACKER_CHECKPOINT" --tracker-candidate "$TRACKER_CANDIDATE")
fi
exec .venv/bin/python -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm "$ARM" \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed "$SEED" \
  --data-repo "$DATA_REPO" \
  --init-params "$INIT_PARAMS" \
  --asset-id robotwin2.0_absolute_meanstd \
  --transition-pairs "$TRANSITION_PAIRS" \
  --num-train-steps "$STEPS" \
  --num-workers "$WORKERS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 100 \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
