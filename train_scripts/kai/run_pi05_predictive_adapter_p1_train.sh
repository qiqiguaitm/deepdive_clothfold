#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
source "$REPO/train_scripts/kai/lib/checkpoint_resume.sh"
TRAIN_VERIFY_REPO=${TRAIN_VERIFY_REPO:-$REPO}
TRAIN_SOURCE_REPO=${TRAIN_SOURCE_REPO:-$REPO}
ARM=${ARM:?set ARM to a0 or candidate}
SEED=${SEED:-1000}
STEPS=${STEPS:-50000}
WORKERS=${WORKERS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
GATE=${PREDICTIVE_P0_GATE:-$REPO/logs/predictive/p0_eval/p0_gate.accepted}
DATA_REPO=${DATA_REPO:-$REPO/../VLANeXt-main/datasets/robotwin2.0_official_prompts_v21}
PYTHON_BIN=${PYTHON_BIN:-$REPO/kai0/.venv/bin/python}

test -f "$GATE"
if [[ "$TRAIN_VERIFY_REPO" != "$REPO" || "$TRAIN_SOURCE_REPO" != "$REPO" ]]; then
  test -s "$TRAIN_VERIFY_REPO/REPLICATION_READY"
  test -s "$TRAIN_SOURCE_REPO/REPLICATION_READY"
fi
SOURCE_AUDIT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p1_baseline_audit.json
test -f "$SOURCE_AUDIT"
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_source_freeze.py" \
  --repo "$TRAIN_VERIFY_REPO" \
  --audit "$SOURCE_AUDIT" \
  --output "$REPO/logs/predictive/p1_preflight/source_freeze_${ARM}_seed${SEED}.json"
case "$ARM" in
  a0)
    TRAIN_ARM=a0
    CONFIG=pi05_predictive_adapter_p1_a0_exact
    EXP=pi05_predictive_adapter_p1_a0_seed${SEED}
    EXTRA_ARGS=()
    ;;
  candidate)
    TRAIN_ARM=p1_predictive
    CONFIG=pi05_predictive_adapter_p1
    EXP=pi05_predictive_adapter_p1_seed${SEED}
    EXTRA_ARGS=(
      --adapter-checkpoint "$REPO/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999"
      --target-pairs "$REPO/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/pairs.npz"
      --frame-cache-root "$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
    )
    ;;
  *)
    echo "unsupported P1 training arm: $ARM" >&2
    exit 2
    ;;
esac

CHECKPOINT_ROOT=$REPO/kai0/checkpoints/$CONFIG/$EXP
RESUME_ARGS=()
checkpoint_resume_args "$CHECKPOINT_ROOT" RESUME_ARGS

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/openpi_cache}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax-predictive-p1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Keep relative checkpoint paths in the canonical repository while the script
# and imported Python modules continue to come from the frozen source overlay.
cd "$REPO/kai0"
exec "$PYTHON_BIN" -u "$TRAIN_SOURCE_REPO/kai0/scripts/train_pi05_robotwin_confirmatory.py" \
  --arm "$TRAIN_ARM" \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed "$SEED" \
  --data-repo "$DATA_REPO" \
  --init-params "$REPO/kai0/checkpoints/pi05_base/params" \
  --norm-assets-dir "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj" \
  --num-train-steps "$STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --num-workers "$WORKERS" \
  --fsdp-devices 1 \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
