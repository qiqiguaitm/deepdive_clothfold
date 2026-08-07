#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:?REPO_ROOT must point to the frozen repository tree}"
LAWAM="$REPO/lmvla/lawam"
ARM="${TG2_ARM:?TG2_ARM must be future_off, fixed_endpoint, or raw_milestone}"
TRAIN_SEED="${TG2_TRAIN_SEED:?TG2_TRAIN_SEED must be 1000, 1001, or 1002}"
SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
case "$ARM" in future_off|fixed_endpoint|raw_milestone) ;; *) exit 2 ;; esac
case "$TRAIN_SEED" in 1000|1001|1002) ;; *) exit 2 ;; esac

RUN_ID="temporal_grounding_tg2_${ARM}_seed${TRAIN_SEED}"
mapfile -t checkpoints < <(find "$LAWAM/results/Checkpoints/robotwin" -maxdepth 3 -type f \
  -path "*+${RUN_ID}/final_model/pytorch_model.pt" | sort)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "expected exactly one final TG2 checkpoint for $RUN_ID, found ${#checkpoints[@]}" >&2
  exit 3
fi
CKPT="${checkpoints[0]}"
RUN="$(dirname "$(dirname "$CKPT")")"
RUN_BASENAME="$(basename "$RUN")"
RESULT_NAME="temporal_grounding_tg2_${ARM}_seed${TRAIN_SEED}"
RESULT_ROOT="$LAWAM/results/eval_runs/robotwin/$RESULT_NAME"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to mix TG2 eval results with existing path: $RESULT_ROOT" >&2
  exit 3
fi

export ROBOTWIN_EPISODE_SEED_MANIFEST="$SCENES"
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=3
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_TEST_NUM=50
export ROBOTWIN_NUM_SLOTS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_TASK_SCOPED_SERVER=1
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE LMWM_REQUIRE_FULL_TARGET_COVERAGE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE
unset LAWAM_FUTURE_INTERVENTION LAWAM_FUTURE_CAPTURE_ROOT LAWAM_FUTURE_SHUFFLE_MANIFEST
unset LAWAM_FUTURE_OFF
if [[ "$ARM" == future_off ]]; then
  export LAWAM_FUTURE_OFF=1
fi

ALL6_EVAL_VARIANT=local \
ALL6_EVAL_RUN_BASENAME="$RUN_BASENAME" \
ALL6_TRAIN_SEED="$TRAIN_SEED" \
SEEDS="0 1 2 3" \
GPUS_PER_SEED=1 \
LOCAL_GPU_COUNT=4 \
NUM_WORKERS=1 \
RESULT_NAME="$RESULT_NAME" \
RUN_TAG_PREFIX="tg2-${ARM}-s${TRAIN_SEED}" \
REPO_ROOT="$REPO" \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh"
