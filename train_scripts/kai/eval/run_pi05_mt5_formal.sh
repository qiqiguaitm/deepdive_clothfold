#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${MT5_ARM:?set MT5_ARM to local or combined}
SEED=${SEED:?set training seed}
SELECTION=${MT3_SELECTION:-$REPO/logs/mt_stage_tracker/selection.json}
candidate=$($REPO/kai0/.venv/bin/python -c "import json; print(json.load(open('$SELECTION'))['selected'])")
case "$candidate" in current_frame|history_proprio) ;; *) exit 2 ;; esac

case "$ARM" in
  local)
    CONFIG=pi05_robotwin_mt5_local_exact
    CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_mt5_local_exact/pi05_robotwin_mt5_local_seed${SEED}/49999}
    ;;
  combined)
    CONFIG=pi05_robotwin_mt5_combined_${candidate}_exact
    CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_mt5_combined_exact/pi05_robotwin_mt5_combined_seed${SEED}/49999}
    ;;
  *) echo "unsupported MT5_ARM=$ARM" >&2; exit 2 ;;
esac

RESULT_NAME=${RESULT_NAME:-pi05_mt5_${ARM}_seed${SEED}}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
ARTIFACT=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1
history=0
[[ "$candidate" == history_proprio ]] && history=1

COMMON_ENV=(
  PI05_EVAL_CONFIG_NAME="$CONFIG"
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd
  CKPT="$CKPT"
  RESULT_NAME="$RESULT_NAME"
  ROBOTWIN_TEST_NUM=50
  ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST"
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
  SEEDS="0 1 2 3"
  LOCAL_GPU_COUNT=4
  GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
  MAX_PARALLEL_SEEDS=4
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-19400}
)
if [[ "$ARM" == combined ]]; then
  COMMON_ENV+=(
    LMWM_TRANSITION_INTERVENTION=predicted
    ROBOTWIN_TRANSITION_INTERVENTION=predicted
    ROBOTWIN_TRANSITION_ORACLE=0
    ROBOTWIN_TRANSITION_HISTORY="$history"
    ROBOTWIN_TRANSITION_PAIRS=$ARTIFACT/pairs.npz
    ROBOTWIN_TRANSITION_TASK_MAP=$ARTIFACT/eval_task_id.json
    ROBOTWIN_TRANSITION_EPISODES=/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21/meta/episodes.jsonl
  )
fi
env "${COMMON_ENV[@]}" bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" \
  --root "$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME"
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME" \
  --expected-cells 24 > "$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncheckpoint=%s\narm=%s\ntracker=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CKPT" "$ARM" "$candidate" "$REPORT" > "$MARKER"
