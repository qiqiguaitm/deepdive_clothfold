#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:?REPO_ROOT must point to the frozen repository tree}"
LAWAM="$REPO/lmvla/lawam"
ARM="${TG4_ARM:?TG4_ARM is required}"
TRAIN_SEED="${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
PY="$REPO/kai0/.venv/bin/python"
DATASET="$LAWAM/dataset/robotwin2_lmwm_all6_v2_v30"
MANIFEST="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_source_decomposition_v1.json"

case "$ARM" in
  clean_base|future_off|auxiliary_only|conditioning_only|parameter_matched_null|full) ;;
  *) echo "unsupported TG4_ARM=$ARM" >&2; exit 2 ;;
esac
case "$TRAIN_SEED" in
  1100|1101|1102) ;;
  *) echo "unsupported TG4_TRAIN_SEED=$TRAIN_SEED" >&2; exit 2 ;;
esac

"$PY" "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_bundle.py" \
  --repo "$REPO" --manifest "$MANIFEST"

for path in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/tasks.parquet" \
  "$DATASET/meta/episodes/chunk-000/file-000.parquet" \
  "$LAWAM/results/Checkpoints/qwen3_weights/config.json" \
  "$LAWAM/latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
do
  test -e "$path" || { echo "missing frozen TG4 input: $path" >&2; exit 13; }
done
if [[ "$ARM" != clean_base ]]; then
  test -f "$LAWAM/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt" || {
    echo "missing LaWAM pretraining checkpoint" >&2
    exit 13
  }
fi

unset LAWAM_FUTURE_OFF LAWAM_AUXILIARY_OFF LAWAM_CONDITIONING_OFF
unset LAWAM_FUTURE_INTERVENTION LAWAM_FUTURE_CAPTURE_ROOT LAWAM_FUTURE_SHUFFLE_MANIFEST
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_ADAPTER_DIR
unset LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_RESIDUAL LMWM_MS_DETACH_BACKBONE

EXTRA_ARGS=(
  --framework.action_model.future_prediction=true
  --framework.action_model.enable_loss_distill=true
)
case "$ARM" in
  clean_base)
    EXTRA_ARGS=(
      --framework.action_model.future_prediction=false
      --framework.action_model.enable_loss_distill=false
      --trainer.pretrained_checkpoint=null
      --trainer.load_pretrained_policy_flow=false
      --trainer.ddp_find_unused_parameters=true
    )
    ;;
  future_off)
    EXTRA_ARGS=(
      --framework.action_model.future_prediction=false
      --framework.action_model.enable_loss_distill=false
      --trainer.ddp_find_unused_parameters=true
    )
    ;;
  auxiliary_only)
    export LAWAM_CONDITIONING_OFF=1
    ;;
  conditioning_only)
    export LAWAM_AUXILIARY_OFF=1
    ;;
  parameter_matched_null)
    export LAWAM_FUTURE_OFF=1
    ;;
  full) ;;
esac

RUN_ID="temporal_grounding_tg4_${ARM}_seed${TRAIN_SEED}"
RUN_GLOB="$LAWAM/results/Checkpoints/robotwin/"*+"$RUN_ID"
if compgen -G "$RUN_GLOB" >/dev/null; then
  echo "refusing to resume or overwrite frozen TG4 arm: $RUN_GLOB" >&2
  exit 3
fi

export LAWAM_RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)_${ARM}_s${TRAIN_SEED}"
export LAWAM_DATA_ORDER_AUDIT_DIR="$REPO/logs/temporal_grounding/tg4/data_order/$RUN_ID"
export LAWAM_INITIALIZATION_AUDIT_PATH="$REPO/logs/temporal_grounding/tg4/initialization/$RUN_ID.json"
export LAWAM_DETERMINISTIC_MODEL_INIT=1
export PYTHON="$PY"
export PATH="$REPO/kai0/.venv/bin:$PATH"

cd "$LAWAM"
echo "TG4 arm=$ARM seed=$TRAIN_SEED global_batch=128 updates=20000 in_order=true workers=8"
bash train_lawam_distributed.sh \
  --config_yaml starVLA/config/training/train_robotwin.yaml \
  --run_id="$RUN_ID" \
  --seed="$TRAIN_SEED" \
  --datasets.vla_data.data_mix=robotwin2_lmwm_all6_v2 \
  --datasets.vla_data.sec_chunk=1.0 \
  --datasets.vla_data.num_frames=2 \
  --datasets.vla_data.per_device_batch_size=16 \
  --datasets.vla_data.num_workers=8 \
  --datasets.vla_data.val_num_workers=2 \
  --datasets.vla_data.in_order=true \
  --datasets.vla_data.enable_video_frame_cache=false \
  --trainer.gradient_accumulation_steps=2 \
  --trainer.optimizer.fused=false \
  --trainer.auto_resume=false \
  --trainer.max_train_steps=20000 \
  --trainer.save_interval=20000 \
  --framework.action_model.future_action_window_size=49 \
  --framework.action_model.past_action_window_size=0 \
  --framework.action_model.action_horizon=50 \
  --framework.action_model.flow_cfg.horizon_sec=1.0 \
  "${EXTRA_ARGS[@]}"
