#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:?REPO_ROOT must point to the frozen North repository tree}"
LAWAM="$REPO/lmvla/lawam"
ARM="${TG2R_ARM:?TG2R_ARM must be future_off, fixed_endpoint, or raw_milestone}"
TRAIN_SEED="${TG2R_TRAIN_SEED:?TG2R_TRAIN_SEED must be 1000, 1001, or 1002}"
DATA_MIX=robotwin2_lmwm_all6_v2
DATASET="$LAWAM/dataset/robotwin2_lmwm_all6_v2_v30"
MILESTONES="$REPO/lmvla/lmwm/data/robotwin_milestone_all6_v2"
PY="$REPO/kai0/.venv/bin/python"
PARENT="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_admission_v1.json"
RECOVERY="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_v1.json"

"$PY" "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$REPO" --manifest "$PARENT" --bundle TG2
"$PY" "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg2_recovery_bundle.py" \
  --repo "$REPO" --manifest "$RECOVERY"

case "$ARM" in
  future_off|fixed_endpoint|raw_milestone) ;;
  *) echo "unsupported TG2R_ARM=$ARM" >&2; exit 2 ;;
esac
case "$TRAIN_SEED" in
  1000|1001|1002) ;;
  *) echo "unsupported TG2R_TRAIN_SEED=$TRAIN_SEED" >&2; exit 2 ;;
esac

for path in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/tasks.parquet" \
  "$DATASET/meta/episodes/chunk-000/file-000.parquet" \
  "$LAWAM/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt" \
  "$LAWAM/results/Checkpoints/qwen3_weights/config.json" \
  "$LAWAM/latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
do
  test -e "$path" || { echo "missing frozen TG2R input: $path" >&2; exit 13; }
done
printf '%s  %s\n' \
  9fe985abf0ef3d0868f9cc330f7248ade01aaf5891f94528a593eb17d6ba14cc \
  "$LAWAM/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt" \
  | sha256sum --check --strict
printf '%s  %s\n' \
  265da39a37405213f1ca0501de96b434672afdb2b618d44df9656429626f465d \
  "$DATASET/meta/info.json" \
  | sha256sum --check --strict

unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE
unset LAWAM_FUTURE_INTERVENTION LAWAM_FUTURE_CAPTURE_ROOT LAWAM_FUTURE_SHUFFLE_MANIFEST
unset LAWAM_FUTURE_OFF LMWM_REQUIRE_FULL_TARGET_COVERAGE

if [[ "$ARM" == future_off ]]; then
  export LAWAM_FUTURE_OFF=1
elif [[ "$ARM" == raw_milestone ]]; then
  for path in "$MILESTONES/pairs.npz" "$MILESTONES/target_compact.npz"; do
    test -f "$path" || { echo "missing frozen raw-milestone input: $path" >&2; exit 13; }
  done
  export LMWM_MILESTONE_TARGET="$MILESTONES/pairs.npz"
  export LMWM_TARGET_COMPACT="$MILESTONES/target_compact.npz"
  export LMWM_FEAT_STRIDE=1
  export LMWM_REQUIRE_FULL_TARGET_COVERAGE=1
  export LMWM_ADAPTER_DIR="$REPO/lmvla/lmwam/adapter"
  "$PY" -c "import sys; sys.path.insert(0, '$LMWM_ADAPTER_DIR'); import lmwm_milestone_target"
fi

RUN_ID="temporal_grounding_tg2r_${ARM}_seed${TRAIN_SEED}"
RUN_GLOB="$LAWAM/results/Checkpoints/robotwin/"*+"$RUN_ID"
if compgen -G "$RUN_GLOB" >/dev/null; then
  echo "refusing to resume or overwrite frozen TG2R arm: $RUN_GLOB" >&2
  exit 3
fi

export LAWAM_RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)_${ARM}_s${TRAIN_SEED}"
export LAWAM_DATA_ORDER_AUDIT_DIR="$REPO/logs/temporal_grounding/tg2r/data_order/$RUN_ID"
export LAWAM_INITIALIZATION_AUDIT_PATH="$REPO/logs/temporal_grounding/tg2r/initialization/$RUN_ID.json"
export LAWAM_DETERMINISTIC_MODEL_INIT=1
export PYTHON="$PY"
export PATH="$REPO/kai0/.venv/bin:$PATH"
cd "$LAWAM"
echo "TG2R arm=$ARM seed=$TRAIN_SEED H=50 E=50 global_batch=128 updates=20000 in_order=true"

bash train_lawam_distributed.sh \
  --config_yaml starVLA/config/training/train_robotwin.yaml \
  --run_id="$RUN_ID" \
  --seed="$TRAIN_SEED" \
  --datasets.vla_data.data_mix="$DATA_MIX" \
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
  --framework.action_model.future_prediction=true \
  --framework.action_model.enable_loss_distill=true \
  --framework.action_model.future_action_window_size=49 \
  --framework.action_model.past_action_window_size=0 \
  --framework.action_model.action_horizon=50 \
  --framework.action_model.flow_cfg.horizon_sec=1.0
