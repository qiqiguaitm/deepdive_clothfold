#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI05_R4_EVAL_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-r4-formal-eval.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PI05_R4_EVAL_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
ARM=${R4_ARM:?set R4_ARM to ordinary, terminal_outcome, or outcome_free_crave}
SEED=1000
STEPS=5000
case "$ARM" in
  ordinary|terminal_outcome|outcome_free_crave) ;;
  *) echo "unsupported R4 arm: $ARM" >&2; exit 2 ;;
esac

RUN_NAME=${ARM}-seed${SEED}
TRAIN_ROOT=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/$RUN_NAME
MODEL=$TRAIN_ROOT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model
TRAIN_MARKER=$REPO/logs/resource_markers/pi05_r4_${RUN_NAME}.ok
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
ROBOTWIN_PY_DEFAULT=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_formal_eval_protocol_v1.json
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1 2 3"}
GPU_COUNT=${LOCAL_GPU_COUNT:-4}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-24600}
RESULT_NAME=${RESULT_NAME:-pi05_r4_${ARM}_seed1000}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
LOG_DIR=${R4_EVAL_LOG_DIR:-$LAWAM/logs/r4_formal_eval}
STAMP=$(date -u +%Y%m%d_%H%M%S)

for required in \
  "$TRAIN_MARKER" "$MODEL/model.safetensors" "$MODEL/config.json" \
  "$TOKENIZER/tokenizer.model" "$MANIFEST" "$PROTOCOL" \
  "$REPO/train_scripts/kai/eval/serve_lerobot_pi05.py"; do
  test -s "$required"
done
mkdir -p "$LOG_DIR" "$(dirname "$MARKER")"

$SERVER_PY - "$PROTOCOL" "$ARM" "$MODEL" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
arm = sys.argv[2]
model = pathlib.Path(sys.argv[3]).resolve()
assert protocol["status"] == "preregistered"
assert arm in protocol["arms"]
assert protocol["training_seed"] == 1000
assert protocol["checkpoint_step"] == 5000
assert model.name == "pretrained_model" and model.parent.name == "005000"
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
export ROBOTWIN_TEST_NUM=50
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=$RESULT_NAME
export ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PORT_SEARCH_LIMIT=30 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}

read -r -a seed_list <<<"$SEEDS"
(( GPU_COUNT > 0 )) || { echo "LOCAL_GPU_COUNT must be positive" >&2; exit 2; }
status=0
for ((offset=0; offset<${#seed_list[@]}; offset+=GPU_COUNT)); do
  pids=()
  for ((slot=0; slot<GPU_COUNT && offset+slot<${#seed_list[@]}; slot++)); do
    seed=${seed_list[$((offset + slot))]}
    gpu=$((GPU_INDEX_OFFSET + slot))
    (
      export CUDA_VISIBLE_DEVICES=$gpu GPU_IDS=$gpu SEED=$seed
      export PORT_BASE=$((PORT_BASE_OFFSET + seed * 80))
      export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed$seed
      mkdir -p "$ROBOTWIN_EVAL_ROOT"
      cd "$LAWAM"
      run_tag=r4-$ARM-seed$seed
      shopt -s nullglob
      schedulers=("$ROBOTWIN_EVAL_ROOT"/*/"$run_tag"/.task_scheduler.json)
      shopt -u nullglob
      if [[ ${#schedulers[@]} -eq 1 ]]; then
        ROBOTWIN_ATTACH_SCHEDULER=1 bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
          "$MODEL" "$TASK_CONFIG" "$run_tag" "$(dirname "${schedulers[0]}")"
      elif [[ ${#schedulers[@]} -eq 0 ]]; then
        bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
          "$MODEL" "$TASK_CONFIG" "$run_tag"
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
done

ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
$SERVER_PY $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest "$MANIFEST" --root "$ROOT"
$SERVER_PY $REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py \
  "$ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
$SERVER_PY - "$REPORT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["summary_count"] == 24
assert result["task_count"] == 6
assert result["total_episodes"] == 1200
PY
printf 'validated=%s\narm=%s\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$ARM" "$MODEL" "$REPORT" >"$MARKER"
