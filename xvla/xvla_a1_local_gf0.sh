#!/usr/bin/env bash
# Task_A1 (细长夹爪叠衣) 本地 gf0 XVLA 训练：14D joint → 20D EE6D
# (continuous gripper alpha, 0.07m=open/0m=close) → 2×A100 DDP。
#
# Usage:
#   ./xvla/xvla_a1_local_gf0.sh prepare
#   ./xvla/xvla_a1_local_gf0.sh smoke
#   ./xvla/xvla_a1_local_gf0.sh full
#
# Overrides: GPUS=0,1 BS=16 WORKERS=8 STEPS_SMOKE=10
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in prepare|smoke|full) ;; *) echo "usage: $0 prepare|smoke|full" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/kai0/.venv_xvla/bin/python"
RAW="$REPO_ROOT/kai0/data/Task_A1/self_built/A1_base_dagger_awbc_enc"
EE6D="$REPO_ROOT/xvla/data/self_built/A1_base_dagger_awbc_enc_ee6d_alpha"
TRAIN="$REPO_ROOT/xvla/launch/xvla_train.py"
CONVERT="$REPO_ROOT/xvla/data/joint_to_ee6d.py"
VALIDATE="$REPO_ROOT/xvla/data/validate_ee6d_dataset.py"
GPUS="${GPUS:-0,1}"
BS="${BS:-16}"
WORKERS="${WORKERS:-8}"
STEPS_SMOKE="${STEPS_SMOKE:-10}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/kai0/checkpoints/xvla_local}"

export XVLA_SB="$REPO_ROOT/xvla/data/self_built"
export XVLA_CKPT_INIT="${XVLA_CKPT_INIT:-$REPO_ROOT/xvla/xvla_ckpts}"
export XVLA_FOLD_INIT="${XVLA_FOLD_INIT:-$REPO_ROOT/xvla/ckpts/xvla_e0_v1_official_fixedcam/step_final/state_dict.pt}"
export XVLA_BART_TOK="$REPO_ROOT/xvla/assets/bart-large-tokenizer"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

for p in "$VENV_PY" "$RAW/meta/info.json" "$XVLA_CKPT_INIT/config.json" \
         "$XVLA_FOLD_INIT" \
         "$XVLA_CKPT_INIT/model.safetensors" "$XVLA_BART_TOK/tokenizer.json"; do
  [ -e "$p" ] || { echo "ERROR: missing $p" >&2; exit 3; }
done

if [ ! -f "$EE6D/.conversion_complete" ]; then
  echo "[A1] converting 14D joint dataset → 20D EE6D (continuous gripper: 0.07m=open, 0m=close)"
  "$VENV_PY" "$CONVERT" --in_dir "$RAW" --out_dir "$EE6D" --workers 32 \
    --continuous --g_open_bound 0.07 --g_close_bound 0.0
  touch "$EE6D/.conversion_complete"
else
  echo "[A1] EE6D dataset already ready: $EE6D"
fi
"$VENV_PY" "$VALIDATE" "$EE6D" --continuous --open-m 0.07 --close-m 0.0
[ "$MODE" = prepare ] && exit 0

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NGPU="${#GPU_LIST[@]}"
if [ "$MODE" = smoke ]; then
  OUT="$OUT_ROOT/xvla_a1_awbc_smoke"
  EXTRA=(--max_steps "$STEPS_SMOKE")
else
  OUT="$OUT_ROOT/xvla_a1_awbc"
  EXTRA=()
fi
mkdir -p "$OUT"

echo "[A1] mode=$MODE gpus=$GPUS world=$NGPU per_gpu_bs=$BS out=$OUT"
if [ "$NGPU" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$VENV_PY" "$TRAIN" \
    --config A1_local_awbc --output_dir "$OUT" --batch_size "$BS" --workers "$WORKERS" \
    "${EXTRA[@]}"
else
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_PY" -m torch.distributed.run \
    --standalone --nproc_per_node="$NGPU" "$TRAIN" \
    --config A1_local_awbc --output_dir "$OUT" --batch_size "$BS" --workers "$WORKERS" \
    "${EXTRA[@]}"
fi
