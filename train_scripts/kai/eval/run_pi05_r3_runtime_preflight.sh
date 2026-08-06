#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
TOKENIZER=${PALIGEMMA_TOKENIZER_PATH:-/vePFS/tim/hf_models/paligemma_tokenizer}
SERVER_PY=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
ARTIFACT=$REPO/lmvla/lmwm/data/pi05_r3_semantic_vocabulary_v1
SCENES=$REPO/lmvla/lmwm/data/pi05_r3_preflight_scene_seed_v1.json
RESULT_NAME=pi05_r3_semantic_next_runtime_preflight_v5
ROOT=$LAWAM/results/eval_runs/robotwin/$RESULT_NAME/seed0
RUN_TAG=r3-semantic-next-preflight-v5-seed0
MARKER=${MARKER:-$REPO/logs/resource_markers/$RESULT_NAME.ok}
AUDIT=$REPO/logs/r3_preflight_v5/protocol_audit.json

mkdir -p "$ROOT" "$(dirname "$MARKER")" "$(dirname "$AUDIT")"
for required in \
  "$MODEL/model.safetensors" "$TOKENIZER/tokenizer.model" "$ARTIFACT/READY" \
  "$SCENES" "$REPO/train_scripts/kai/eval/serve_lerobot_pi05_r3.py"; do
  test -s "$required"
done

"$SERVER_PY" "$REPO/lmvla/lmwm/scripts/verify_pi05_r3_screen_protocol.py" \
  --repo "$REPO" \
  --protocol "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r3_semantic_screen_protocol_v1.json" \
  --artifact "$ARTIFACT" \
  --condition semantic_next \
  --model-root "$MODEL" \
  --tokenizer-root "$TOKENIZER" \
  --output "$AUDIT"

export STAR_VLA_PYTHON=$SERVER_PY
export ROBOTWIN_SERVER_BACKEND=openpi ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT=$REPO/train_scripts/kai/eval/serve_lerobot_pi05_r3.py
export PALIGEMMA_TOKENIZER_PATH=$TOKENIZER KAI0_ROOT=$REPO
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwm/scripts/robotwin_python_wrapper.sh}
export PYTHONPATH=$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
export ROBOTWIN_TASKS=beat_block_hammer TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=1 ROBOTWIN_NUM_SLOTS=1 NUM_WORKERS=1 ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50 ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export ROBOTWIN_EPISODE_SEED_MANIFEST=$SCENES ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export ROBOTWIN_TRANSITION_ORACLE=1 ROBOTWIN_TRANSITION_INTERVENTION=correct
export ROBOTWIN_TRANSITION_PAIRS=$ARTIFACT/semantic_profile_pairs.npz
export ROBOTWIN_TRANSITION_EPISODES=$ARTIFACT/semantic_profile_episodes.jsonl
export ROBOTWIN_TRANSITION_TASK_MAP=$ARTIFACT/task_map.json
export R3_SEMANTIC_VOCABULARY=$ARTIFACT/vocabulary.json
export R3_SEMANTIC_TASK_MAP=$ARTIFACT/task_map.json R3_SEMANTIC_MODE=semantic-next
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PORT_SEARCH_LIMIT=30 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}
export CUDA_VISIBLE_DEVICES=0 GPU_IDS=0 SEED=0 PORT_BASE=${PORT_BASE:-24580}
export ROBOTWIN_EVAL_ROOT=$ROOT

cd "$LAWAM"
shopt -s nullglob
schedulers=("$ROOT"/*/"$RUN_TAG"/.task_scheduler.json)
shopt -u nullglob
if [[ ${#schedulers[@]} -eq 1 ]]; then
  ROBOTWIN_ATTACH_SCHEDULER=1 bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
    "$MODEL" demo_clean "$RUN_TAG" "$(dirname "${schedulers[0]}")"
elif [[ ${#schedulers[@]} -eq 0 ]]; then
  bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
    "$MODEL" demo_clean "$RUN_TAG"
else
  echo "ambiguous preflight scheduler count: ${#schedulers[@]}" >&2
  exit 12
fi

summary=$(find "$ROOT" -name summary.json -type f -print -quit)
test -n "$summary"
"$SERVER_PY" - "$summary" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["task_name"] == "beat_block_hammer", d
assert d["n_episodes"] == 1 and len(d["episodes"]) == 1, d
assert d["episodes"][0]["seed"] == 100000, d
assert d["model_queries"] > 0, d
PY
printf 'validated=%s\nresult=%s\nsummary=%s\n' \
  "$(date -u +%FT%TZ)" "$RESULT_NAME" "$summary" >"$MARKER"
