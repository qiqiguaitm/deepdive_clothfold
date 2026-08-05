#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
COLLECTOR_LAWAM=${R4_COLLECTOR_LAWAM:-$REPO/logs/frozen_source_overlays/pi05_r4_collector_v1/lawam}
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
FFMPEG_DIR=${R4_FFMPEG_DIR:-/vePFS/tim/workspace/miniconda3_portable/envs/vlanext/bin}
ROBOTWIN_PY_DEFAULT=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
SCENE_MANIFEST=${R4_SCENE_MANIFEST:-$REPO/lmvla/lmwm/data/pi05_r4_outcome_scene_seeds_v1.json}
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1 2 3"}
TEST_NUM=${ROBOTWIN_TEST_NUM:-10}
GPU_COUNT=${LOCAL_GPU_COUNT:-4}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
RESULT_NAME=${RESULT_NAME:-pi05_r4_outcomes_public_v1}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-r4-outcomes-public}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-24800}
FINALIZE=${R4_FINALIZE_DATASET:-1}
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_r4_outcome_collection.ok}
AUDIT_JSON=${AUDIT_JSON:-$REPO/logs/r4/outcomes/dataset_audit.json}
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR=$REPO/logs/r4/outcomes/collector

mkdir -p "$LOG_DIR" "$(dirname "$MARKER")" "$(dirname "$AUDIT_JSON")"
export PATH="$FFMPEG_DIR:$PATH"

require_file() {
  local path=$1
  [[ -s "$path" ]] || { echo "required file missing or empty: $path" >&2; exit 2; }
}

require_file "$COLLECTOR_LAWAM/COLLECTOR_READY"
require_file "$MODEL/model.safetensors"
require_file "$TOKENIZER/tokenizer.model"
[[ -x "$SERVER_PY" ]] || { echo "server Python is not executable: $SERVER_PY" >&2; exit 2; }
require_file "$SCENE_MANIFEST"
command -v ffmpeg >/dev/null || { echo "ffmpeg unavailable under PATH=$PATH" >&2; exit 2; }

python3 - "$COLLECTOR_LAWAM/COLLECTOR_READY" "$SCENE_MANIFEST" "$TEST_NUM" <<'PY'
import json, sys
ready=json.load(open(sys.argv[1])); scenes=json.load(open(sys.argv[2]))
assert ready["protocol"] == "pi05_r4_trajectory_collector_overlay_v1", ready
assert ready["lawam_commit"] == "865e0b631c67cc5463feab04e34056a5538186c5", ready
assert ready["patch_sha256"] == "3a39b0c77077561a85922a9aef1a9828900626557e0f7e14e48be9a58f77058b", ready
assert scenes["protocol"] == "pi05_r4_outcome_scene_seeds_v1", scenes
assert scenes["episodes_per_cell"] == int(sys.argv[3]), scenes
PY

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
export ROBOTWIN_SAVE_TRAJECTORY=1
export ROBOTWIN_EPISODE_SEED_MANIFEST=$SCENE_MANIFEST
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
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
    export ROBOTWIN_EVAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
    mkdir -p "$ROBOTWIN_EVAL_ROOT"
    cd "$COLLECTOR_LAWAM"
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

root=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
expected=$((${#seed_list[@]} * ${#task_list[@]} * TEST_NUM))
summary_count=$(find "$root" -name summary.json -type f | wc -l)
video_count=$(find "$root" -name 'episode*.mp4' -type f | wc -l)
trajectory_count=$(find "$root" -name 'episode*.npz' -type f | wc -l)
[[ $summary_count -eq $((${#seed_list[@]} * ${#task_list[@]})) ]]
[[ $video_count -eq $expected ]]
[[ $trajectory_count -eq $expected ]]

if [[ "$FINALIZE" == 1 ]]; then
  dataset_manifest=$root/dataset_manifest.json
  python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_outcome_manifest.py" \
    --result-root "$root" --scene-manifest "$SCENE_MANIFEST" \
    --behavior-policy "$MODEL/model.safetensors" --output "$dataset_manifest"
  python3 "$REPO/lmvla/lmwm/scripts/audit_pi05_r4_outcome_dataset.py" \
    --manifest "$dataset_manifest" --output "$AUDIT_JSON"
fi

printf 'completed=%s\nroot=%s\nsummaries=%s\nvideos=%s\ntrajectories=%s\naudit=%s\n' \
  "$(date -u +%FT%TZ)" "$root" "$summary_count" "$video_count" \
  "$trajectory_count" "$AUDIT_JSON" >"$MARKER"
