#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
CONDITION="${TG1A_CONDITION:?TG1A_CONDITION must be normal, shuffled, null, or persistence}"
CKPT="$LAWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
SHUFFLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_shuffle_v1.json"
FEATURE_ROOT="$REPO/logs/tg1_retry500/predicted_endpoint_features"
CAPTURE_MARKER="$REPO/logs/resource_markers/temporal_grounding_tg1a_retry500_normal_capture_complete.json"
RESULT_ROOT="$LAWAM/results/eval_runs/robotwin/temporal_grounding_tg1a_${CONDITION}"
LOG_DIR="$REPO/logs/temporal_grounding/tg1a/platform"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
BUNDLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json"
AMENDMENT="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json"

"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$REPO" --manifest "$BUNDLE" --bundle TG1A
"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg1_retry500.py" \
  --repo "$REPO" --manifest "$AMENDMENT" --bundle TG1A

case "$CONDITION" in
  normal|shuffled|null|persistence) ;;
  *) echo "unsupported TG1A_CONDITION=$CONDITION" >&2; exit 2 ;;
esac

test -f "$CKPT"
test -f "$SCENES"
test -f "$SHUFFLE"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to mix TG1A retry500 results with existing path: $RESULT_ROOT" >&2
  exit 3
fi
if [[ "$CONDITION" == normal && -e "$FEATURE_ROOT" ]]; then
  echo "refusing to overwrite TG1A retry500 normal feature capture: $FEATURE_ROOT" >&2
  exit 3
fi
mkdir -p "$RESULT_ROOT" "$LOG_DIR"
if [[ "$CONDITION" == normal ]]; then
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
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE LMWM_REQUIRE_FULL_TARGET_COVERAGE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE
unset LAWAM_FUTURE_OFF

if [[ "$CONDITION" == normal || "$CONDITION" == shuffled ]]; then
  export LAWAM_FUTURE_CAPTURE_ROOT="$FEATURE_ROOT"
else
  unset LAWAM_FUTURE_CAPTURE_ROOT
fi
if [[ "$CONDITION" == shuffled ]]; then
  export LAWAM_FUTURE_SHUFFLE_MANIFEST="$SHUFFLE"
  test -f "$CAPTURE_MARKER"
else
  unset LAWAM_FUTURE_SHUFFLE_MANIFEST
fi

cd "$LAWAM"
pids=()
status=0
seed_index=0
for seed in 0 1 2 3; do
  gpu_index=$((seed_index % ${LOCAL_GPU_COUNT:-4}))
  (
    export CUDA_VISIBLE_DEVICES="$gpu_index"
    export GPU_IDS="$gpu_index"
    export SEED="$seed"
    export PORT_BASE=$((14000 + seed * 100))
    export ROBOTWIN_CKPT_ALIAS=lawam_robotwin_sft_release
    export ROBOTWIN_EVAL_ROOT="$RESULT_ROOT/seed$seed"
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
      "$CKPT" "$TASK_CONFIG" "tg1a-${CONDITION}-seed${seed}"
  ) >"$LOG_DIR/tg1a_retry500_${CONDITION}_seed${seed}_${STAMP}.log" 2>&1 &
  pids+=("$!")
  seed_index=$((seed_index + 1))
  sleep 20
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" == 0 && "$CONDITION" == normal ]]; then
  "$REPO/kai0/.venv/bin/python" \
    "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_feature_capture.py" \
    --feature-root "$FEATURE_ROOT" \
    --scene-manifest "$SCENES" \
    --output "$CAPTURE_MARKER"
fi
echo "TG1A retry500 condition=$CONDITION finished status=$status at $(date -u --iso-8601=seconds)"
exit "$status"
