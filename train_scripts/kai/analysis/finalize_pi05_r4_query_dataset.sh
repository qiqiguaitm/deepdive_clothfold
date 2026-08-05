#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
MANIFEST_DIR=$REPO/logs/r4/outcomes/query_manifests
BASE_A=$MANIFEST_DIR/pi05_r4_query_base_train_a_v1_query.json
BASE_B=$MANIFEST_DIR/pi05_r4_query_base_train_b_v1_query.json
SUPPORT=$MANIFEST_DIR/pi05_r4_query_beat_support_v1_query.json
BALANCED_A=$MANIFEST_DIR/pi05_r4_query_balanced_support_a_v1_query.json
BALANCED_B=$MANIFEST_DIR/pi05_r4_query_balanced_support_b_v1_query.json
OUTCOME=$REPO/logs/r4/outcomes/dataset_manifest_combined_v1.json
COMBINED=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_train_v1.json
AUDIT=$REPO/logs/r4/outcomes/query_dataset_audit_v1.json
MARKER=$REPO/logs/resource_markers/pi05_r4_query_dataset.ok
MODEL=/vePFS/tim/hf_models/SidneyXie_pi05_robotwin

materialize_balanced_manifest() {
  local name=$1 scenes=$2 tasks=$3 output=$4
  if [[ -s "$output" ]]; then
    return 0
  fi
  local root=$REPO/lmvla/lawam/results/eval_runs/robotwin/$name
  local outcome=$MANIFEST_DIR/${name}_outcome.json
  python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_outcome_manifest.py" \
    --result-root "$root" --scene-manifest "$scenes" \
    --behavior-policy "$MODEL/model.safetensors" --output "$outcome" \
    --tasks $tasks --eval-seeds 0 1
  python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_query_manifest.py" \
    --outcome-manifest "$outcome" --output "$output"
}

mkdir -p "$MANIFEST_DIR"
materialize_balanced_manifest \
  pi05_r4_query_balanced_support_a_v1 \
  "$REPO/lmvla/lmwm/data/pi05_r4_balanced_train_support_a_v1.json" \
  "blocks_ranking_size blocks_ranking_rgb handover_block" "$BALANCED_A"
materialize_balanced_manifest \
  pi05_r4_query_balanced_support_b_v1 \
  "$REPO/lmvla/lmwm/data/pi05_r4_balanced_train_support_b_v1.json" \
  "stack_blocks_two stack_blocks_three" "$BALANCED_B"

test -s "$BASE_A"
test -s "$BASE_B"
test -s "$SUPPORT"
test -s "$BALANCED_A"
test -s "$BALANCED_B"
test -s "$OUTCOME"
rm -f "$MARKER"

python3 "$REPO/lmvla/lmwm/scripts/merge_pi05_r4_query_manifests.py" \
  --manifest "$BASE_A" --manifest "$BASE_B" --manifest "$SUPPORT" \
  --manifest "$BALANCED_A" --manifest "$BALANCED_B" \
  --output "$COMBINED"
python3 "$REPO/lmvla/lmwm/scripts/audit_pi05_r4_query_dataset.py" \
  --query-manifest "$COMBINED" --outcome-manifest "$OUTCOME" --output "$AUDIT"

python3 - "$AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["accepted"] is True, report
assert report["record_count"] == report["expected_train_record_count"] == 600, report
PY

printf 'completed=%s\nquery_manifest=%s\naudit=%s\n' \
  "$(date -u +%FT%TZ)" "$COMBINED" "$AUDIT" >"$MARKER"
