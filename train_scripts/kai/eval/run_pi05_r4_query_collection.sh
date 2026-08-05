#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
RESULT_NAME=${RESULT_NAME:?RESULT_NAME is required}
TASKS=${ROBOTWIN_TASKS:-"beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three"}
SEEDS=${SEEDS:-"0 1"}
TEST_NUM=${ROBOTWIN_TEST_NUM:-10}
FINAL_MARKER=${MARKER:?MARKER is required}
INNER_MARKER=${FINAL_MARKER}.outcomes
ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST_DIR=${R4_QUERY_MANIFEST_DIR:-$REPO/logs/r4/outcomes/query_manifests}
OUTCOME_MANIFEST=$MANIFEST_DIR/${RESULT_NAME}_outcome.json
QUERY_MANIFEST=$MANIFEST_DIR/${RESULT_NAME}_query.json
HOOK_DIR=$REPO/lmvla/lmwm/runtime/pi05_r4_query_capture
QUERY_ROBOTWIN_PY=$HOOK_DIR/robotwin_python_wrapper.sh

test -s "$HOOK_DIR/sitecustomize.py"
test -s "$HOOK_DIR/hook.py"
test -x "$QUERY_ROBOTWIN_PY"
mkdir -p "$MANIFEST_DIR"
rm -f "$FINAL_MARKER" "$INNER_MARKER"

export R4_CAPTURE_QUERY_OBSERVATIONS=1
export PYTHONPATH=$HOOK_DIR:${PYTHONPATH:-}
export ROBOTWIN_PYTHON=$QUERY_ROBOTWIN_PY
export R4_FINALIZE_DATASET=0
export MARKER=$INNER_MARKER
unset OUTPUT_ROOT

bash "$REPO/train_scripts/kai/eval/run_pi05_r4_outcome_collection.sh"

read -r -a seed_list <<<"$SEEDS"
read -r -a task_list <<<"$TASKS"
expected=$((${#seed_list[@]} * ${#task_list[@]} * TEST_NUM))
query_count=$(find "$ROOT" -name 'query_episode*.npz' -type f | wc -l)
[[ $query_count -eq $expected ]] || {
  echo "expected $expected R4 query files, found $query_count" >&2
  exit 14
}

python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_outcome_manifest.py" \
  --result-root "$ROOT" --scene-manifest "$R4_SCENE_MANIFEST" \
  --behavior-policy "$MODEL/model.safetensors" --output "$OUTCOME_MANIFEST" \
  --tasks "${task_list[@]}" --eval-seeds "${seed_list[@]}"
python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_query_manifest.py" \
  --outcome-manifest "$OUTCOME_MANIFEST" --output "$QUERY_MANIFEST"

printf 'completed=%s\nroot=%s\nrecords=%s\nquery_manifest=%s\n' \
  "$(date -u +%FT%TZ)" "$ROOT" "$expected" "$QUERY_MANIFEST" >"$FINAL_MARKER"
