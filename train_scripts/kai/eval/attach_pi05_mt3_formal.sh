#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
INTERVENTION=${MT3_INTERVENTION:?set MT3_INTERVENTION}
case "$INTERVENTION" in
  predicted|within_task|null|oracle) ;;
  *) echo "unsupported MT3_INTERVENTION=$INTERVENTION" >&2; exit 2 ;;
esac

SELECTION=${MT3_SELECTION:-$REPO/logs/mt_stage_tracker/selection.json}
candidate=$($REPO/kai0/.venv/bin/python -c \
  "import json; print(json.load(open('$SELECTION'))['selected'])")
case "$candidate" in current_frame|history_proprio) ;; *) exit 2 ;; esac

config=pi05_robotwin_mt3_learned_${candidate}_exact
history=0
[[ "$candidate" == history_proprio ]] && history=1
oracle=0
[[ "$INTERVENTION" == oracle ]] && oracle=1

ARTIFACT=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1
export PI05_EVAL_CONFIG_NAME=$config
export PI05_ASSET_ID=robotwin2.0_absolute_meanstd
export ROBOTWIN_RUN_GROUP=${config}__demo_clean
export LMWM_TRANSITION_INTERVENTION=$INTERVENTION
export ROBOTWIN_TRANSITION_INTERVENTION=$INTERVENTION
export ROBOTWIN_TRANSITION_ORACLE=$oracle
export ROBOTWIN_TRANSITION_HISTORY=$history
export ROBOTWIN_TRANSITION_PAIRS=$ARTIFACT/pairs.npz
export ROBOTWIN_TRANSITION_TASK_MAP=$ARTIFACT/eval_task_id.json
export ROBOTWIN_TRANSITION_EPISODES=${ROBOTWIN_TRANSITION_EPISODES:-/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21/meta/episodes.jsonl}
export ATTACH_RUN_TAG_PREFIX=${ATTACH_RUN_TAG_PREFIX:-local-unseen-a3-seed}

if [[ "${MT3_ATTACH_DRY_RUN:-0}" == 1 ]]; then
  printf 'candidate=%s\nconfig=%s\nintervention=%s\noracle=%s\nhistory=%s\nrun_group=%s\n' \
    "$candidate" "$config" "$INTERVENTION" "$oracle" "$history" "$ROBOTWIN_RUN_GROUP"
  exit 0
fi

exec bash "$REPO/train_scripts/kai/eval/attach_pi05_a0_confirmatory_platform.sh"
