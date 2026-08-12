#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
CKPT="$LAWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
SHUFFLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_shuffle_v1.json"
FEATURE_ROOT="$REPO/logs/tg1_retry500/predicted_endpoint_features"
CAPTURE_MARKER="$REPO/logs/resource_markers/temporal_grounding_tg1a_retry500_normal_capture_complete.json"
RESULT_ROOT="$LAWAM/results/eval_runs/robotwin/temporal_grounding_tg1a_shuffled"
MARKER="$REPO/logs/resource_markers/temporal_grounding_tg1a_shuffled_tail_east4g.ok"
LOG_DIR="$REPO/logs/temporal_grounding/tg1a/tail"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
BUNDLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json"
AMENDMENT="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json"

"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$REPO" --manifest "$BUNDLE" --bundle TG1A
"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg1_retry500.py" \
  --repo "$REPO" --manifest "$AMENDMENT" --bundle TG1A

test -f "$CKPT"
test -f "$SCENES"
test -f "$SHUFFLE"
test -f "$CAPTURE_MARKER"
mkdir -p "$LOG_DIR"

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
export LAWAM_FUTURE_INTERVENTION=shuffled
export LAWAM_FUTURE_CAPTURE_ROOT="$FEATURE_ROOT"
export LAWAM_FUTURE_SHUFFLE_MANIFEST="$SHUFFLE"
export ROBOTWIN_ATTACH_SCHEDULER=1
export ROBOTWIN_ATTACH_REQUEUE_FAILED=0
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export USE_BF16=1
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE LMWM_REQUIRE_FULL_TARGET_COVERAGE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE
unset LAWAM_FUTURE_OFF

cd "$LAWAM"
pids=()
status=0
for seed in 0 1 2 3; do
  run_dir="$RESULT_ROOT/seed${seed}/lawam_robotwin_sft_release__demo_clean/tg1a-shuffled-seed${seed}"
  test -f "$run_dir/.task_scheduler.json"
  (
    export CUDA_VISIBLE_DEVICES="$seed"
    export GPU_IDS=0
    export SEED="$seed"
    export PORT_BASE=$((24000 + seed * 200))
    export ROBOTWIN_WORKER_INDEX_OFFSET=$((1000 + seed * 100))
    export ROBOTWIN_CKPT_ALIAS=lawam_robotwin_sft_release
    bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
      "$CKPT" "$TASK_CONFIG" "tg1a-shuffled-seed${seed}" "$run_dir"
  ) >"$LOG_DIR/tg1a_shuffled_tail_seed${seed}_${STAMP}.log" 2>&1 &
  pids+=("$!")
  sleep 10
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" == 0 ]]; then
  printf 'completed=%s\nseeds=0 1 2 3\nmode=shared_scheduler_attach\n' \
    "$(date -u +%FT%TZ)" > "$MARKER"
fi
exit "$status"
