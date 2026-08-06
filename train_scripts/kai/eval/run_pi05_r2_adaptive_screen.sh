#!/usr/bin/env bash
set -euo pipefail

if [[ "${PI05_R2_SCREEN_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-r2-adaptive-screen.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PI05_R2_SCREEN_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
ROBOTWIN_PY_DEFAULT=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
READOUT=${ROBOTWIN_R2_READOUT:-$REPO/lmvla/lmwm/data/pi05_r2_causal_readout_v1/readout.npz}
MANIFEST=${R2_SCENE_MANIFEST:-$REPO/lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json}
CONDITION=${R2_CONDITION:?set R2_CONDITION to fixed4 or adaptive}
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1 2 3"}
GPU_COUNT=${LOCAL_GPU_COUNT:-4}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-24800}

case "$CONDITION" in
  fixed4|adaptive) ;;
  *) echo "unknown R2_CONDITION=$CONDITION" >&2; exit 2 ;;
esac

RESULT_NAME=${RESULT_NAME:-pi05_r2_${CONDITION}_screen_v1}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
LOG_DIR=$LAWAM/logs/r2_adaptive_screen
DIAGNOSTICS_ROOT=$REPO/logs/r2_adaptive_screen/diagnostics
STAMP=$(date -u +%Y%m%d_%H%M%S)

for required in \
  "$MODEL/model.safetensors" "$TOKENIZER/tokenizer.model" "$READOUT" \
  "$(dirname "$READOUT")/readout_manifest.json" \
  "$(dirname "$READOUT")/r2_readout.accepted" "$MANIFEST" \
  "$REPO/train_scripts/kai/eval/pi05_r2_python_wrapper.sh" \
  "$REPO/train_scripts/kai/eval/run_pi05_r2_batched_eval_runner.py" \
  "$LAWAM/examples/Robotwin/eval_files/model2robotwin_openpi_r2.py"; do
  test -s "$required"
done
mkdir -p "$LOG_DIR" "$DIAGNOSTICS_ROOT" "$(dirname "$MARKER")"

$SERVER_PY $REPO/lmvla/lmwm/scripts/verify_pi05_r2_protocol.py \
  --repo "$REPO" \
  --protocol "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r2_adaptive_execution_protocol_v1.json" \
  --condition "$CONDITION" \
  --readout "$READOUT" \
  --model-root "$MODEL" \
  --tokenizer-root "$TOKENIZER" \
  --output "$LOG_DIR/${RESULT_NAME}_protocol_audit.json"

export R2_REAL_PYTHON=$SERVER_PY
export STAR_VLA_PYTHON=$REPO/train_scripts/kai/eval/pi05_r2_python_wrapper.sh
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT=$REPO/train_scripts/kai/eval/serve_lerobot_pi05.py
export PALIGEMMA_TOKENIZER_PATH=$TOKENIZER
export KAI0_ROOT=$REPO RT_REPO=$REPO
export ROBOTWIN_MODEL_INTERFACE=openpi_r2
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$ROBOTWIN_PY_DEFAULT}
export PYTHONPATH=$REPO/kai0/packages/openpi-client/src:$REPO/lmvla/crave/src:${PYTHONPATH:-}
export ROBOTWIN_R2_CONDITION=$CONDITION
export ROBOTWIN_R2_READOUT=$READOUT
export ROBOTWIN_R2_DIAGNOSTICS_ROOT=$DIAGNOSTICS_ROOT
export ROBOTWIN_TASKS=$TASKS
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=10
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=0
export ROBOTWIN_REPLAN_STEPS=4
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
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
    run_tag=r2-$CONDITION-seed$seed
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

ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME
$SERVER_PY $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest "$MANIFEST" --root "$ROOT"
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
$SERVER_PY $REPO/lmvla/lmwm/scripts/summarize_pi05_r2_eval.py \
  "$ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
$SERVER_PY - "$REPORT" "$DIAGNOSTICS_ROOT/$CONDITION" <<'PY'
import json, pathlib, sys
d=json.load(open(sys.argv[1]))
assert d['summary_count']==24 and d['task_count']==6 and d['total_episodes']==240, d
assert len(d['efficiency_cells'])==24 and d['total_model_queries'] > 0, d
diagnostics=list(pathlib.Path(sys.argv[2]).glob('seed*/*.json'))
assert len(diagnostics)==24, len(diagnostics)
PY
printf 'validated=%s\ncondition=%s\nreport=%s\nroot=%s\ndiagnostics=%s\n' \
  "$(date -u +%FT%TZ)" "$CONDITION" "$REPORT" "$ROOT" "$DIAGNOSTICS_ROOT/$CONDITION" >"$MARKER"
