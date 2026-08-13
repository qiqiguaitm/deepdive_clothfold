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
CHECKPOINT_ROOT="${TG4_CHECKPOINT_ROOT:-$LAWAM/results/Checkpoints/robotwin}"
RESULT_BASE="${TG4_RESULT_BASE:-$LAWAM/results/eval_runs/robotwin}"
FEATURE_BASE="${TG4_FEATURE_BASE:-$REPO/logs/temporal_grounding/tg4/features}"
MARKER_ROOT="${TG4_MARKER_ROOT:-$REPO/logs/resource_markers}"
LOG_DIR="${TG4_LOG_DIR:-$REPO/logs/temporal_grounding/tg4/eval}"
CONTROL_PYTHON="${TG4_CONTROL_PYTHON:-$REPO/kai0/.venv/bin/python}"
RESUME_HELPER="$REPO/lmvla/lmwm/scripts/prepare_temporal_grounding_tg4_eval_resume.py"
STRICT_RESULT_VERIFIER="$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_eval_results.py"
RESULT_ROOT="$RESULT_BASE/${RUN_ID}_${CONDITION}"
FEATURE_ROOT="$FEATURE_BASE/${RUN_ID}"
CAPTURE_MARKER="$MARKER_ROOT/${RUN_ID}_normal_capture_complete.json"
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
test -f "$RESUME_HELPER"
test -f "$STRICT_RESULT_VERIFIER"
"$CONTROL_PYTHON" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_evaluation.py" \
  --repo "$REPO" --manifest "$EVAL_MANIFEST"
mapfile -t RUN_DIRS < <(find "$CHECKPOINT_ROOT" -maxdepth 1 \
  -type d -name "*+${RUN_ID}" -print | sort)
[[ ${#RUN_DIRS[@]} -eq 1 ]] || {
  echo "expected exactly one checkpoint run for $RUN_ID, found ${#RUN_DIRS[@]}" >&2
  printf '%s\n' "${RUN_DIRS[@]}" >&2
  exit 13
}
CKPT="${RUN_DIRS[0]}/final_model/pytorch_model.pt"
test -f "$CKPT"
if [[ -e "$RESULT_ROOT" && ! -d "$RESULT_ROOT" ]]; then
  echo "TG4 evaluation result root is not a directory: $RESULT_ROOT" >&2
  exit 3
fi
result_root_preexisting=0
[[ ! -d "$RESULT_ROOT" ]] || result_root_preexisting=1
if [[ "$ARM" == full && "$CONDITION" == normal ]]; then
  if [[ -e "$FEATURE_ROOT" && ! -d "$FEATURE_ROOT" ]]; then
    echo "TG4 feature capture root is not a directory: $FEATURE_ROOT" >&2
    exit 3
  fi
  if [[ -d "$FEATURE_ROOT" && "$result_root_preexisting" == 0 ]]; then
    echo "refusing orphaned TG4 feature capture without a result root: $FEATURE_ROOT" >&2
    exit 3
  fi
  if [[ "$result_root_preexisting" == 1 && ! -d "$FEATURE_ROOT" ]] && \
      find "$RESULT_ROOT" -type f -name summary.json -print -quit | grep -q .; then
    echo "refusing TG4 full resume with summaries but no feature capture: $RESULT_ROOT" >&2
    exit 3
  fi
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
      # batched_eval_runner assigns CUDA_VISIBLE_DEVICES from GPU_IDS for each
      # model server, so pass the host-visible physical index through directly.
      unset CUDA_VISIBLE_DEVICES
      export GPU_IDS="${gpu_devices[$offset]}"
      export SEED="$eval_seed"
      export PORT_BASE=$((14000 + PORT_OFFSET + eval_seed * 100))
      export ROBOTWIN_CKPT_ALIAS="tg4_${ARM}_s${TRAIN_SEED}"
      export ROBOTWIN_EVAL_ROOT="$RESULT_ROOT/seed${eval_seed}"
      mkdir -p "$ROBOTWIN_EVAL_ROOT"
      resume_run_dir="$(
        "$CONTROL_PYTHON" "$RESUME_HELPER" \
          --result-root "$RESULT_ROOT" \
          --checkpoint "$CKPT" \
          --arm "$ARM" \
          --train-seed "$TRAIN_SEED" \
          --condition "$CONDITION" \
          --eval-seed "$eval_seed"
      )"
      eval_args=(
        "$CKPT"
        "$TASK_CONFIG"
        "tg4-${ARM}-s${TRAIN_SEED}-${CONDITION}-e${eval_seed}"
      )
      if [[ -n "$resume_run_dir" ]]; then
        eval_args+=("$resume_run_dir")
      fi
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "${eval_args[@]}"
    ) >"$LOG_DIR/${ARM}_s${TRAIN_SEED}_${CONDITION}_e${eval_seed}_${STAMP}.log" 2>&1 &
    pids+=("$!")
    sleep 20
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
done

if [[ "$status" == 0 ]] && ! "$CONTROL_PYTHON" "$STRICT_RESULT_VERIFIER" \
    --manifest "$SCENES" --root "$RESULT_ROOT"; then
  status=1
fi
if [[ "$status" == 0 && "$ARM" == full && "$CONDITION" == normal ]]; then
  "$CONTROL_PYTHON" \
    "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_feature_capture.py" \
    --feature-root "$FEATURE_ROOT" \
    --scene-manifest "$SCENES" \
    --output "$CAPTURE_MARKER"
fi
echo "TG4 evaluation arm=$ARM seed=$TRAIN_SEED condition=$CONDITION status=$status at $(date -u --iso-8601=seconds)"
exit "$status"
