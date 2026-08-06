#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI05_R3_SCREEN_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-r3-semantic-screen.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PI05_R3_SCREEN_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
ROBOTWIN_PY_DEFAULT=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
ARTIFACT=${R3_ARTIFACT:-$REPO/lmvla/lmwm/data/pi05_r3_semantic_vocabulary_v1}
MANIFEST=${R3_SCENE_MANIFEST:-$REPO/lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json}
CONDITION=${R3_CONDITION:?set R3_CONDITION}
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1 2 3"}
GPU_COUNT=${LOCAL_GPU_COUNT:-4}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-24200}

case "$CONDITION" in
  semantic_next) MODE=semantic-next; INTERVENTION=correct ;;
  generic_stage) MODE=generic-stage; INTERVENTION=correct ;;
  semantic_current) MODE=semantic-current; INTERVENTION=correct ;;
  shuffled_semantic) MODE=semantic-next; INTERVENTION=within-task ;;
  no_subtask) MODE=none; INTERVENTION=correct ;;
  *) echo "unknown R3_CONDITION=$CONDITION" >&2; exit 2 ;;
esac

RESULT_NAME=${RESULT_NAME:-pi05_r3_${CONDITION}_screen_v1}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
LOG_DIR=${R3_LOG_DIR:-$LAWAM/logs/r3_semantic_screen}
STAMP=$(date -u +%Y%m%d_%H%M%S)

for required in \
  "$MODEL/model.safetensors" "$TOKENIZER/tokenizer.model" "$ARTIFACT/READY" \
  "$ARTIFACT/vocabulary.json" "$ARTIFACT/task_map.json" \
  "$ARTIFACT/semantic_profile_pairs.npz" "$ARTIFACT/semantic_profile_episodes.jsonl" \
  "$MANIFEST" "$REPO/train_scripts/kai/eval/serve_lerobot_pi05_r3.py"; do
  test -s "$required"
done
mkdir -p "$LOG_DIR" "$(dirname "$MARKER")"

$SERVER_PY $REPO/lmvla/lmwm/scripts/verify_pi05_r3_screen_protocol.py \
  --repo "$REPO" \
  --protocol "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r3_semantic_screen_protocol_v1.json" \
  --artifact "$ARTIFACT" \
  --condition "$CONDITION" \
  --model-root "$MODEL" \
  --tokenizer-root "$TOKENIZER" \
  --output "$LOG_DIR/${RESULT_NAME}_protocol_audit.json"

export STAR_VLA_PYTHON=$SERVER_PY
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT=$REPO/train_scripts/kai/eval/serve_lerobot_pi05_r3.py
export PALIGEMMA_TOKENIZER_PATH=$TOKENIZER
export KAI0_ROOT=$REPO
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$ROBOTWIN_PY_DEFAULT}
export PYTHONPATH=$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM:-10}
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export ROBOTWIN_TRANSITION_ORACLE=1
export ROBOTWIN_TRANSITION_INTERVENTION=$INTERVENTION
export ROBOTWIN_TRANSITION_PAIRS=$ARTIFACT/semantic_profile_pairs.npz
export ROBOTWIN_TRANSITION_EPISODES=$ARTIFACT/semantic_profile_episodes.jsonl
export ROBOTWIN_TRANSITION_TASK_MAP=$ARTIFACT/task_map.json
export R3_SEMANTIC_VOCABULARY=$ARTIFACT/vocabulary.json
export R3_SEMANTIC_TASK_MAP=$ARTIFACT/task_map.json
export R3_SEMANTIC_MODE=$MODE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PORT_SEARCH_LIMIT=30 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}

read -r -a seed_list <<<"$SEEDS"
if (( ${#seed_list[@]} > GPU_COUNT )); then
  echo "seed count ${#seed_list[@]} exceeds GPU_COUNT=$GPU_COUNT" >&2
  exit 2
fi
pids=()
status=0
for index in "${!seed_list[@]}"; do
  seed=${seed_list[$index]}
  gpu=$((GPU_INDEX_OFFSET + index))
  (
    export CUDA_VISIBLE_DEVICES=$gpu GPU_IDS=$gpu SEED=$seed
    export PORT_BASE=$((PORT_BASE_OFFSET + index * 80))
    export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$LAWAM"
    run_tag=r3-$CONDITION-seed$seed
    shopt -s nullglob
    schedulers=("$ROBOTWIN_EVAL_ROOT"/*/"$run_tag"/.task_scheduler.json)
    shopt -u nullglob
    if [[ ${#schedulers[@]} -eq 1 ]]; then
      ROBOTWIN_ATTACH_SCHEDULER=1 bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$MODEL" "$TASK_CONFIG" "$run_tag" "$(dirname "${schedulers[0]}")"
    elif [[ ${#schedulers[@]} -eq 0 ]]; then
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh "$MODEL" "$TASK_CONFIG" "$run_tag"
    else
      echo "ambiguous scheduler count ${#schedulers[@]} for $run_tag" >&2
      exit 12
    fi
  ) >"$LOG_DIR/${RESULT_NAME}_seed${seed}_${STAMP}.log" 2>&1 &
  pids+=("$!")
  sleep 10
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME
$SERVER_PY $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py --manifest "$MANIFEST" --root "$ROOT"
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
$SERVER_PY $REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py "$ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
$SERVER_PY - "$REPORT" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d['summary_count']==24 and d['task_count']==6 and d['total_episodes']==240, d
PY
printf 'validated=%s\ncondition=%s\nmode=%s\nintervention=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CONDITION" "$MODE" "$INTERVENTION" "$REPORT" >"$MARKER"
