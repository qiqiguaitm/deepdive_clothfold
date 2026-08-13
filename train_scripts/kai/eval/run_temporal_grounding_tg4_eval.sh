#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
ARM="${TG4_ARM:?TG4_ARM is required}"
TRAIN_SEED="${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
CONDITION="${TG4_CONDITION:?TG4_CONDITION is required}"
GPU_COUNT="${LOCAL_GPU_COUNT:-4}"
VISIBLE_GPUS="${TG4_VISIBLE_GPUS:-}"
PORT_OFFSET="${TG4_PORT_OFFSET:-0}"
SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
SHUFFLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_shuffle_v1.json"
INTEGRITY="$REPO/logs/resource_markers/temporal_grounding_tg4_training_integrity.ok"
EVAL_MANIFEST="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json"
RUN_ID="temporal_grounding_tg4_${ARM}_seed${TRAIN_SEED}"
RESULT_ROOT="$LAWAM/results/eval_runs/robotwin/${RUN_ID}_${CONDITION}"
FEATURE_ROOT="$REPO/logs/temporal_grounding/tg4/features/${RUN_ID}"
CAPTURE_MARKER="$REPO/logs/resource_markers/${RUN_ID}_normal_capture_complete.json"
LOG_DIR="$REPO/logs/temporal_grounding/tg4/eval"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

case "$ARM" in
  clean_base|future_off|auxiliary_only|conditioning_only|parameter_matched_null|full) ;;
  *) echo "unsupported TG4_ARM=$ARM" >&2; exit 2 ;;
esac
case "$TRAIN_SEED" in
  1100|1101|1102) ;;
  *) echo "unsupported TG4_TRAIN_SEED=$TRAIN_SEED" >&2; exit 2 ;;
esac
case "$CONDITION" in
  normal) ;;
  shuffled)
    [[ "$ARM" == full ]] || {
      echo "shuffled TG4 evaluation is only defined for full" >&2
      exit 2
    }
    ;;
  *) echo "unsupported TG4_CONDITION=$CONDITION" >&2; exit 2 ;;
esac
(( GPU_COUNT >= 1 && GPU_COUNT <= 4 )) || {
  echo "LOCAL_GPU_COUNT must be in [1,4]" >&2
  exit 2
}
[[ "$PORT_OFFSET" =~ ^[0-9]+$ ]] || {
  echo "TG4_PORT_OFFSET must be a non-negative integer" >&2
  exit 2
}
gpu_devices=()
if [[ -n "$VISIBLE_GPUS" ]]; then
  IFS=, read -r -a gpu_devices <<<"$VISIBLE_GPUS"
  (( ${#gpu_devices[@]} >= GPU_COUNT )) || {
    echo "TG4_VISIBLE_GPUS provides fewer devices than LOCAL_GPU_COUNT" >&2
    exit 2
  }
else
  for ((gpu_index=0; gpu_index<GPU_COUNT; gpu_index++)); do
    gpu_devices+=("$gpu_index")
  done
fi

test -f "$INTEGRITY"
test -f "$SCENES"
test -f "$SHUFFLE"
test -f "$EVAL_MANIFEST"
"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_evaluation.py" \
  --repo "$REPO" --manifest "$EVAL_MANIFEST"
mapfile -t RUN_DIRS < <(find "$LAWAM/results/Checkpoints/robotwin" -maxdepth 1 \
  -type d -name "*+${RUN_ID}" -print | sort)
[[ ${#RUN_DIRS[@]} -eq 1 ]] || {
  echo "expected exactly one checkpoint run for $RUN_ID, found ${#RUN_DIRS[@]}" >&2
  printf '%s\n' "${RUN_DIRS[@]}" >&2
  exit 13
}
CKPT="${RUN_DIRS[0]}/final_model/pytorch_model.pt"
test -f "$CKPT"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to mix TG4 evaluation results with existing path: $RESULT_ROOT" >&2
  exit 3
fi
if [[ "$ARM" == full && "$CONDITION" == normal && -e "$FEATURE_ROOT" ]]; then
  echo "refusing to overwrite TG4 feature capture: $FEATURE_ROOT" >&2
  exit 3
fi
if [[ "$CONDITION" == shuffled ]]; then
  test -f "$CAPTURE_MARKER"
  test -d "$FEATURE_ROOT"
fi
mkdir -p "$RESULT_ROOT" "$LOG_DIR"
if [[ "$ARM" == full && "$CONDITION" == normal ]]; then
  mkdir -p "$FEATURE_ROOT"
fi

export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-$REPO/kai0/.venv/bin/python}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}"
export ROBOTWIN_TASKS="beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=50
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_REPLAN_STEPS=36
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_EPISODE_SEED_MANIFEST="$SCENES"
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export ROBOTWIN_TASK_SCOPED_SERVER=1
export LAWAM_FUTURE_INTERVENTION="$CONDITION"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export USE_BF16=1
unset LAWAM_FUTURE_OFF LAWAM_AUXILIARY_OFF LAWAM_CONDITIONING_OFF
unset LAWAM_FUTURE_CAPTURE_ROOT LAWAM_FUTURE_SHUFFLE_MANIFEST
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE

case "$ARM" in
  auxiliary_only) export LAWAM_CONDITIONING_OFF=1 ;;
  conditioning_only) export LAWAM_AUXILIARY_OFF=1 ;;
  parameter_matched_null) export LAWAM_FUTURE_OFF=1 ;;
esac
if [[ "$ARM" == full ]]; then
  export LAWAM_FUTURE_CAPTURE_ROOT="$FEATURE_ROOT"
fi
if [[ "$CONDITION" == shuffled ]]; then
  export LAWAM_FUTURE_SHUFFLE_MANIFEST="$SHUFFLE"
fi

cd "$LAWAM"
status=0
for ((batch_start=0; batch_start<4; batch_start+=GPU_COUNT)); do
  pids=()
  for ((offset=0; offset<GPU_COUNT && batch_start+offset<4; offset++)); do
    eval_seed=$((batch_start + offset))
    (
      export CUDA_VISIBLE_DEVICES="${gpu_devices[$offset]}"
      export GPU_IDS="$offset"
      export SEED="$eval_seed"
      export PORT_BASE=$((14000 + PORT_OFFSET + eval_seed * 100))
      export ROBOTWIN_CKPT_ALIAS="tg4_${ARM}_s${TRAIN_SEED}"
      export ROBOTWIN_EVAL_ROOT="$RESULT_ROOT/seed${eval_seed}"
      mkdir -p "$ROBOTWIN_EVAL_ROOT"
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$CKPT" "$TASK_CONFIG" "tg4-${ARM}-s${TRAIN_SEED}-${CONDITION}-e${eval_seed}"
    ) >"$LOG_DIR/${ARM}_s${TRAIN_SEED}_${CONDITION}_e${eval_seed}_${STAMP}.log" 2>&1 &
    pids+=("$!")
    sleep 20
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
done

if [[ "$status" == 0 && "$ARM" == full && "$CONDITION" == normal ]]; then
  "$REPO/kai0/.venv/bin/python" \
    "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_feature_capture.py" \
    --feature-root "$FEATURE_ROOT" \
    --scene-manifest "$SCENES" \
    --output "$CAPTURE_MARKER"
fi
echo "TG4 evaluation arm=$ARM seed=$TRAIN_SEED condition=$CONDITION status=$status at $(date -u --iso-8601=seconds)"
exit "$status"
