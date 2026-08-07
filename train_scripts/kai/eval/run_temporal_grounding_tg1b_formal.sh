#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
CHECKPOINT_ARM="${TG1B_CHECKPOINT_ARM:?TG1B_CHECKPOINT_ARM must be future_off or local_wm}"
CADENCE="${TG1B_EXECUTION_CADENCE:?TG1B_EXECUTION_CADENCE must be 36 or 50}"
SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
BUNDLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1b_admission_v1.json"

"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$REPO" --manifest "$BUNDLE" --bundle TG1B

case "$CHECKPOINT_ARM" in
  future_off)
    VARIANT=nowm
    RUN_BASENAME=20260731_172204+robotwin_all6_v2_nowm_seed2027
    ;;
  local_wm)
    VARIANT=local
    RUN_BASENAME=20260730_234942+robotwin_all6_v2_local_seed2027
    ;;
  *) echo "unsupported TG1B_CHECKPOINT_ARM=$CHECKPOINT_ARM" >&2; exit 2 ;;
esac
case "$CADENCE" in
  36|50) ;;
  *) echo "unsupported TG1B_EXECUTION_CADENCE=$CADENCE" >&2; exit 2 ;;
esac

CKPT="$LAWAM/results/Checkpoints/robotwin/$RUN_BASENAME/final_model/pytorch_model.pt"
RESULT_NAME="temporal_grounding_tg1b_${CHECKPOINT_ARM}_e${CADENCE}"
RESULT_ROOT="$LAWAM/results/eval_runs/robotwin/$RESULT_NAME"
test -f "$CKPT"
test -f "$SCENES"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to mix TG1B results with existing path: $RESULT_ROOT" >&2
  exit 3
fi

export ROBOTWIN_EPISODE_SEED_MANIFEST="$SCENES"
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=3
export ROBOTWIN_REPLAN_STEPS="$CADENCE"
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_TEST_NUM=50
export ROBOTWIN_NUM_SLOTS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_TASK_SCOPED_SERVER=1
unset LAWAM_FUTURE_INTERVENTION LAWAM_FUTURE_CAPTURE_ROOT LAWAM_FUTURE_SHUFFLE_MANIFEST

ALL6_EVAL_VARIANT="$VARIANT" \
ALL6_EVAL_RUN_BASENAME="$RUN_BASENAME" \
ALL6_TRAIN_SEED=2027 \
SEEDS="0 1 2 3" \
GPUS_PER_SEED=1 \
LOCAL_GPU_COUNT=4 \
NUM_WORKERS=1 \
RESULT_NAME="$RESULT_NAME" \
RUN_TAG_PREFIX="tg1b-${CHECKPOINT_ARM}-e${CADENCE}" \
REPO_ROOT="$REPO" \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh"
