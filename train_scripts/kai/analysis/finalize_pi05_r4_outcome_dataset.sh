#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
MODEL=${PUBLIC_PI05_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
BASE_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_outcomes_public_v1
SUPPORT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_beat_train_support_supplement_v1
SUPPORT_SCENES=$REPO/lmvla/lmwm/data/pi05_r4_beat_train_support_supplement_v1.json
BASE_MANIFEST=$BASE_ROOT/dataset_manifest.json
SUPPORT_MANIFEST=$SUPPORT_ROOT/dataset_manifest.json
COMBINED_MANIFEST=$REPO/logs/r4/outcomes/dataset_manifest_combined_v1.json
AUDIT=$REPO/logs/r4/outcomes/dataset_audit_combined_v1.json
MARKER=$REPO/logs/resource_markers/pi05_r4_outcome_collection.ok

test -s "$BASE_MANIFEST"
test -s "$SUPPORT_ROOT/seed0/SidneyXie_pi05_robotwin__demo_clean/r4-beat-train-support-seed0/tasks/beat_block_hammer/summary.json"
test -s "$SUPPORT_ROOT/seed1/SidneyXie_pi05_robotwin__demo_clean/r4-beat-train-support-seed1/tasks/beat_block_hammer/summary.json"

python3 "$REPO/lmvla/lmwm/scripts/build_pi05_r4_outcome_manifest.py" \
  --result-root "$SUPPORT_ROOT" --scene-manifest "$SUPPORT_SCENES" \
  --behavior-policy "$MODEL/model.safetensors" --output "$SUPPORT_MANIFEST"
python3 "$REPO/lmvla/lmwm/scripts/merge_pi05_r4_outcome_manifests.py" \
  --manifest "$BASE_MANIFEST" --manifest "$SUPPORT_MANIFEST" \
  --output "$COMBINED_MANIFEST"
python3 "$REPO/lmvla/lmwm/scripts/audit_pi05_r4_outcome_dataset.py" \
  --manifest "$COMBINED_MANIFEST" --output "$AUDIT"
test "$(jq -r '.accepted' "$AUDIT")" = true

printf 'completed=%s\nmanifest=%s\naudit=%s\nrecords=%s\n' \
  "$(date -u +%FT%TZ)" "$COMBINED_MANIFEST" "$AUDIT" \
  "$(jq -r '.record_count' "$AUDIT")" > "$MARKER"
