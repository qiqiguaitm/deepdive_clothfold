#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
BASE_A=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_base_train_a_v1/query_manifest.json
BASE_B=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_base_train_b_v1/query_manifest.json
SUPPORT=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_beat_support_v1/query_manifest.json
OUTCOME=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_outcomes_public_v1/dataset_manifest_combined_v1.json
COMBINED=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_train_v1.json
AUDIT=$REPO/logs/r4/outcomes/query_dataset_audit_v1.json
MARKER=$REPO/logs/resource_markers/pi05_r4_query_dataset.ok

test -s "$BASE_A"
test -s "$BASE_B"
test -s "$SUPPORT"
test -s "$OUTCOME"
rm -f "$MARKER"

python3 "$REPO/lmvla/lmwm/scripts/merge_pi05_r4_query_manifests.py" \
  --manifest "$BASE_A" --manifest "$BASE_B" --manifest "$SUPPORT" \
  --output "$COMBINED"
python3 "$REPO/lmvla/lmwm/scripts/audit_pi05_r4_query_dataset.py" \
  --query-manifest "$COMBINED" --outcome-manifest "$OUTCOME" --output "$AUDIT"

python3 - "$AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["accepted"] is True, report
assert report["record_count"] == report["expected_train_record_count"] == 200, report
PY

printf 'completed=%s\nquery_manifest=%s\naudit=%s\n' \
  "$(date -u +%FT%TZ)" "$COMBINED" "$AUDIT" >"$MARKER"
