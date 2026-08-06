#!/usr/bin/env bash
set -euo pipefail

# This diagnostic intentionally uses a separate entrypoint so the frozen P1/P2
# evaluator source snapshot remains unchanged.
if [[ "${CRAVE_R0_ROLLOUT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-crave-r0-rollouts.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  CRAVE_R0_ROLLOUT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
ROBOTWIN_PY_DEFAULT=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1"}
TEST_NUM=${ROBOTWIN_TEST_NUM:-10}
GPU_COUNT=${LOCAL_GPU_COUNT:-2}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
RESULT_NAME=${RESULT_NAME:-pi05_crave_r0_public_rollouts_v1}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-crave-r0-public}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-23600}
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_crave_r0_rollout_collection.ok}
AUDIT_JSON=${AUDIT_JSON:-$REPO/logs/crave_r0/rollouts/artifact_audit.json}
AUDIT_MD=${AUDIT_MD:-$REPO/logs/crave_r0/rollouts/artifact_audit.md}
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=$LAWAM/logs/crave_r0_rollouts

mkdir -p "$LOG_DIR" "$(dirname "$MARKER")" "$(dirname "$AUDIT_JSON")"
test -s "$MODEL/model.safetensors"
test -s "$TOKENIZER/tokenizer.model"
test -x "$SERVER_PY"
command -v ffmpeg >/dev/null

export STAR_VLA_PYTHON=$SERVER_PY
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT=$REPO/train_scripts/kai/eval/serve_lerobot_pi05.py
export PALIGEMMA_TOKENIZER_PATH=$TOKENIZER
export KAI0_ROOT=$REPO
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$ROBOTWIN_PY_DEFAULT}
export PYTHONPATH=$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=$TEST_NUM
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=1
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PORT_SEARCH_LIMIT=30
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}

read -r -a seed_list <<<"$SEEDS"
read -r -a task_list <<<"$TASKS"
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
    export CUDA_VISIBLE_DEVICES=$gpu
    export GPU_IDS=$gpu
    export SEED=$seed
    export PORT_BASE=$((PORT_BASE_OFFSET + index * 80))
    export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$LAWAM"
    run_tag=$RUN_TAG_PREFIX-seed$seed
    shopt -s nullglob
    scheduler_paths=("$ROBOTWIN_EVAL_ROOT"/*/"$run_tag"/.task_scheduler.json)
    shopt -u nullglob
    if [[ ${#scheduler_paths[@]} -eq 1 ]]; then
      run_dir=$(dirname "${scheduler_paths[0]}")
      ROBOTWIN_ATTACH_SCHEDULER=1 \
        bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$MODEL" "$TASK_CONFIG" "$run_tag" "$run_dir"
    elif [[ ${#scheduler_paths[@]} -eq 0 ]]; then
      bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
        "$MODEL" "$TASK_CONFIG" "$run_tag"
    else
      echo "ambiguous schedulers for $run_tag: ${scheduler_paths[*]}" >&2
      exit 12
    fi
  ) >"$LOG_DIR/${RESULT_NAME}_seed${seed}_${STAMP}.log" 2>&1 &
  pids+=("$!")
  sleep 10
done

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

root=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME
expected_summaries=$((${#seed_list[@]} * ${#task_list[@]}))
expected_videos=$((expected_summaries * TEST_NUM))
summary_count=$(find "$root" -name summary.json -type f | wc -l)
video_count=$(find "$root" -name 'episode*.mp4' -type f | wc -l)
[[ $summary_count -eq $expected_summaries ]]
[[ $video_count -eq $expected_videos ]]

python "$REPO/lmvla/lmwm/scripts/audit_robotwin_rollout_artifacts.py" \
  --root "$root" --json-out "$AUDIT_JSON" --markdown-out "$AUDIT_MD"
python - "$AUDIT_JSON" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["successes"] > 0, audit
assert audit["failures"] > 0, audit
assert audit["trajectory_file_count"] > 0, audit
assert audit["supports_crave_rollout_metrics"], audit
PY

printf 'completed=%s\nroot=%s\nsummaries=%s\nvideos=%s\naudit=%s\n' \
  "$(date -u +%FT%TZ)" "$root" "$summary_count" "$video_count" "$AUDIT_JSON" >"$MARKER"
