#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${R1_ARM:?set R1_ARM to crave or combined}
SEED=${SEED:-1000}
STEPS=${STEPS:-50000}
GPU_COUNT=${GPU_COUNT:-4}
WORKERS=${WORKERS:-$([[ "$GPU_COUNT" -ge 8 ]] && echo 16 || echo 8)}
P0_GATE=${PREDICTIVE_P0_GATE:-$REPO/logs/predictive/p0_eval/p0_gate.accepted}
R0_GATE=${CRAVE_R0_GATE:-$REPO/logs/crave_r0/probe_gate/r0_gate.accepted}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json
CRAVE_TARGETS=$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/r1_dense_targets.npz
CRAVE_TARGETS_MANIFEST=$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/r1_dense_targets_manifest.json
P0_CHECKPOINT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999

for required in "$P0_GATE" "$R0_GATE" "$PROTOCOL" "$CRAVE_TARGETS" "$CRAVE_TARGETS_MANIFEST"; do
  test -s "$required"
done
mkdir -p "$REPO/logs/r1"
python3 "$REPO/lmvla/lmwm/scripts/verify_pi05_r1_protocol.py" \
  --repo "$REPO" --protocol "$PROTOCOL" --output "$REPO/logs/r1/protocol_${ARM}_s${SEED}.json"

case "$ARM" in
  crave)
    TRAIN_ARM=r1_crave
    CONFIG=pi05_r1_crave
    EXP=pi05_r1_crave_seed${SEED}
    EXTRA_ARGS=()
    ;;
  combined)
    TRAIN_ARM=r1_combined
    CONFIG=pi05_r1_combined
    EXP=pi05_r1_combined_seed${SEED}
    EXTRA_ARGS=(
      --adapter-checkpoint "$P0_CHECKPOINT"
      --target-pairs "$REPO/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/pairs.npz"
      --frame-cache-root "$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
    )
    ;;
  *)
    echo "unsupported R1 arm: $ARM" >&2
    exit 2
    ;;
esac

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/openpi_cache}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax-r1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd "$REPO/kai0"
exec .venv/bin/python -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm "$TRAIN_ARM" \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed "$SEED" \
  --data-repo "$REPO/../VLANeXt-main/datasets/robotwin2.0_official_prompts_v21" \
  --init-params "$REPO/kai0/checkpoints/pi05_base/params" \
  --norm-assets-dir "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj" \
  --crave-targets "$CRAVE_TARGETS" \
  --num-train-steps "$STEPS" \
  --save-interval 5000 \
  --num-workers "$WORKERS" \
  --fsdp-devices 1 \
  "${EXTRA_ARGS[@]}"
