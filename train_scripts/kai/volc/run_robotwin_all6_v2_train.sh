#!/usr/bin/env bash
set -euo pipefail

VARIANT="${ROBOTWIN_ALL6_VARIANT:?set ROBOTWIN_ALL6_VARIANT}"
TRAIN_SEED="${ROBOTWIN_ALL6_SEED:-2026}"
REPO="${ROBOTWIN_ALL6_REPO:?set ROBOTWIN_ALL6_REPO}"
MAX_STEPS="${ROBOTWIN_ALL6_MAX_STEPS:-20000}"
SAVE_INTERVAL="${ROBOTWIN_ALL6_SAVE_INTERVAL:-5000}"
PER_DEVICE_BATCH_SIZE="${ROBOTWIN_ALL6_PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${ROBOTWIN_ALL6_GRADIENT_ACCUMULATION_STEPS:-2}"
FUSED_ADAMW="${ROBOTWIN_ALL6_FUSED_ADAMW:-false}"
AUTO_RESUME="${ROBOTWIN_ALL6_AUTO_RESUME:-true}"
DATA_MIX="robotwin2_lmwm_all6_v2"
MS="$REPO/lmvla/lmwm/data/robotwin_milestone_all6_v2"
DS="$REPO/lmvla/lawam/dataset/robotwin2_lmwm_all6_v2_v30"
PY="$REPO/kai0/.venv/bin/python"

mkdir -p /home/tim/workspace
ln -sfn "$REPO" /home/tim/workspace/deepdive_kai0
export PYTHON="$PY"
export PATH="$REPO/kai0/.venv/bin:$PATH"

for f in \
  "$DS/meta/modality.json" \
  "$DS/meta/tasks.parquet" \
  "$DS/meta/episodes/chunk-000/file-000.parquet" \
  "$DS/meta/info.json" \
  "$MS/READY" \
  "$REPO/lmvla/lawam/results/Checkpoints/qwen3_weights/config.json" \
  "$REPO/lmvla/lawam/latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
do
  test -e "$f" || { echo "FATAL missing $f" >&2; exit 13; }
done
if [ "$VARIANT" != neverwm ]; then
  test -f "$REPO/lmvla/lawam/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt" || {
    echo "FATAL missing LaWAM pretrain checkpoint" >&2
    exit 13
  }
fi

cd "$REPO/lmvla/lawam"
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE

EXTRA_ARGS=()
case "$VARIANT" in
  nowm|nowm_resetflow|neverwm)
    EXTRA_ARGS+=(
      --framework.action_model.future_prediction=false
      --framework.action_model.enable_loss_distill=false
      --trainer.ddp_find_unused_parameters=true
    )
    if [ "$VARIANT" = nowm_resetflow ]; then
      EXTRA_ARGS+=(--trainer.load_pretrained_policy_flow=false)
    elif [ "$VARIANT" = neverwm ]; then
      EXTRA_ARGS+=(
        --trainer.pretrained_checkpoint=null
        --trainer.load_pretrained_policy_flow=false
        --trainer.freeze.unfreeze_lam_decoder=false
      )
    fi
    ;;
  local)
    ;;
  absolute|residual|isolation|combo)
    for f in "$MS/pairs.npz" "$MS/target_compact.npz" "$MS/lmwm.pt"; do
      test -f "$f" || { echo "FATAL missing $f" >&2; exit 13; }
    done
    export LMWM_CKPT="$MS/lmwm.pt"
    export LMWM_MILESTONE_TARGET="$MS/pairs.npz"
    export LMWM_TARGET_COMPACT="$MS/target_compact.npz"
    export LMWM_ADAPTER_DIR="$REPO/lmvla/lmwam/adapter"
    test -f "$LMWM_ADAPTER_DIR/lmwm_adapter.py"
    test -f "$LMWM_ADAPTER_DIR/lmwm_milestone_target.py"
    "$PY" -c "import sys; sys.path.insert(0, '$LMWM_ADAPTER_DIR'); import lmwm_adapter, lmwm_milestone_target"
    export LMWM_SWAP_TEACHER=1
    export LMWM_FEAT_STRIDE=1
    export LMWM_HINT_DROPOUT=0.15
    export LMWM_DUAL=1
    export LMWM_DUAL_2Q=1
    test "$VARIANT" = residual && export LMWM_MS_RESIDUAL=1
    test "$VARIANT" = isolation && export LMWM_MS_DETACH_BACKBONE=1
    if test "$VARIANT" = combo; then
      export LMWM_MS_RESIDUAL=1
      export LMWM_MS_DETACH_BACKBONE=1
    fi
    ;;
  *)
    echo "FATAL unknown variant $VARIANT" >&2
    exit 14
    ;;
esac

RUN_ID="robotwin_all6_v2_${VARIANT}_seed${TRAIN_SEED}"
STAMP=$(date -u +%Y%m%d_%H%M%S)
export LAWAM_RUN_TIMESTAMP="$STAMP"
echo "variant=$VARIANT seed=$TRAIN_SEED run_id=$RUN_ID data_mix=$DATA_MIX"
echo "steps=$MAX_STEPS batch_per_device=$PER_DEVICE_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS fused_adamw=$FUSED_ADAMW auto_resume=$AUTO_RESUME"
if [ "$VARIANT" = neverwm ]; then
  echo "initializer=pure_qwen3vl policy_flow=random lawam_pretrain=disabled wm_objective=disabled"
elif [ "$VARIANT" = nowm_resetflow ]; then
  echo "initializer=lawam_pretrain policy_flow=random wm_objective=disabled"
fi

bash train_lawam_distributed.sh \
  --config_yaml starVLA/config/training/train_robotwin.yaml \
  --run_id="$RUN_ID" \
  --seed="$TRAIN_SEED" \
  --datasets.vla_data.data_mix="$DATA_MIX" \
  --datasets.vla_data.per_device_batch_size="$PER_DEVICE_BATCH_SIZE" \
  --trainer.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --trainer.optimizer.fused="$FUSED_ADAMW" \
  --trainer.auto_resume="$AUTO_RESUME" \
  --trainer.max_train_steps="$MAX_STEPS" \
  --trainer.save_interval="$SAVE_INTERVAL" \
  "${EXTRA_ARGS[@]}"
