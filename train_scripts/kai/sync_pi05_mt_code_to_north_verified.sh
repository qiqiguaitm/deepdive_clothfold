#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
EVIDENCE_DIR=$REPO/logs/sync/pi05_mt_north_code
MARKER=$REPO/logs/resource_markers/pi05_mt_north_code_sync.ok

roots=(
  kai0/src/openpi
  kai0/scripts/serve_policy.py
  lmvla/lawam/examples/Robotwin/eval_files
  lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
  lmvla/lmwm/scripts/summarize_robotwin_eval.py
  lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
  lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz
  lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/eval_task_id.json
  train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh
  train_scripts/kai/eval/run_pi05_mt_transition_formal.sh
)

mkdir -p "$EVIDENCE_DIR" "$(dirname "$MARKER")"
local_manifest=$EVIDENCE_DIR/local.sha256
remote_manifest=$EVIDENCE_DIR/north.sha256
path_list=$EVIDENCE_DIR/paths.txt

(
  cd "$REPO"
  find "${roots[@]}" -type f -print | LC_ALL=C sort > "$path_list"
  sha256sum $(cat "$path_list") > "$local_manifest"
  tar -cf - $(cat "$path_list")
) | ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p '$NORTH_REPO' && tar -C '$NORTH_REPO' -xf -"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "cd '$NORTH_REPO' && xargs sha256sum" < "$path_list" > "$remote_manifest"
cmp "$local_manifest" "$remote_manifest"

temporary=$MARKER.tmp.$$
{
  printf 'validated=%s\nremote=%s\nfiles=%s\n' \
    "$(date -u +%FT%TZ)" "$NORTH_REPO" "$(wc -l < "$path_list")"
  cat "$local_manifest"
} > "$temporary"
mv "$temporary" "$MARKER"
printf 'verified North MT evaluator sync: files=%s marker=%s\n' \
  "$(wc -l < "$path_list")" "$MARKER"
