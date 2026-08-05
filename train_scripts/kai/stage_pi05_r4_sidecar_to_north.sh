#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
PAYLOAD=$REPO/logs/r4/north_stage/payload
REMOTE_STAGE=$NORTH_REPO/.staging/pi05_r4_sidecar_v1/repo
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_sidecar_north_stage.ok
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/pi05_r4_sidecar_north_stage.ok
SOURCE_BASE=$REPO/lmvla/lawam_local/results/eval_runs/robotwin
SELECTION=$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json
OUTCOME_FREE=$REPO/logs/r4/training/outcome_free_query_manifest.json
AMENDMENT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_sidecar_north_amendment_v1.json

test -s "$OUTCOME_FREE"
test -s "$SELECTION"
test -s "$AMENDMENT"
rm -rf "$PAYLOAD"
mkdir -p \
  "$PAYLOAD/lmvla/lawam_local/results/eval_runs/robotwin" \
  "$PAYLOAD/lmvla/lmwm/data/pi05_crave_r0_v1" \
  "$PAYLOAD/lmvla/lmwm/data/pi05_r4_training_v1" \
  "$PAYLOAD/lmvla/lmwm/data/robotwin_dinov3base" \
  "$PAYLOAD/lmvla/lmwm/scripts" \
  "$PAYLOAD/lmvla/crave" \
  "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests" \
  "$PAYLOAD/logs/r4/training"

printf 'phase=build-payload\n'
python3 - "$OUTCOME_FREE" "$SELECTION" "$SOURCE_BASE" "$REPO" "$PAYLOAD" <<'PY'
import json
import pathlib
import shutil
import sys

outcome_path, selection_path, source_base, repo, payload = map(pathlib.Path, sys.argv[1:])
outcome = json.loads(outcome_path.read_text())
selection = json.loads(selection_path.read_text())
source_manifest = pathlib.Path(outcome["source_manifest"])
for record in outcome["records"]:
    relative = pathlib.Path(record["query_observations"])
    source = (source_manifest.parent / relative).resolve()
    destination = payload / "lmvla/lawam_local/results/eval_runs/robotwin" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
shutil.copy2(
    source_manifest,
    payload / "lmvla/lawam_local/results/eval_runs/robotwin" / source_manifest.name,
)
reference_source = repo / "lmvla/lmwm/data/robotwin_dinov3base"
reference_destination = payload / "lmvla/lmwm/data/robotwin_dinov3base"
for task in sorted(selection["tasks"]):
    for episode in selection["tasks"][task]["reference_episodes"]:
        name = f"ep{int(episode)}.npz"
        destination = reference_destination / name
        if not destination.exists():
            shutil.copy2(reference_source / name, destination)
PY

cp -a "$OUTCOME_FREE" "$PAYLOAD/logs/r4/training/outcome_free_query_manifest.json"
cp -a "$SELECTION" "$PAYLOAD/lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json"
cp -a "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json" \
  "$PAYLOAD/lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json"
cp -a "$REPO/lmvla/lmwm/data/pi05_r4_training_v1/query_action_chunks.npz" \
  "$PAYLOAD/lmvla/lmwm/data/pi05_r4_training_v1/query_action_chunks.npz"
cp -a "$REPO/lmvla/lmwm/scripts/build_pi05_r4_crave_weight_sidecar.py" \
  "$REPO/lmvla/lmwm/scripts/build_pi05_crave_r0_labels.py" \
  "$REPO/lmvla/lmwm/scripts/build_pi05_r4_outcome_free_manifest.py" \
  "$PAYLOAD/lmvla/lmwm/scripts/"
cp -a "$REPO/lmvla/crave/src" "$PAYLOAD/lmvla/crave/src"
cp -a "$AMENDMENT" "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests/"

find "$PAYLOAD" -type f -printf '%P\0' | sort -z | \
  xargs -0 -I{} sha256sum "$PAYLOAD/{}" | \
  sed "s#  $PAYLOAD/#  #" >"$REPO/logs/r4/north_stage/payload.sha256"
query_count=$(find "$PAYLOAD/lmvla/lawam_local/results/eval_runs/robotwin" \
  -name 'query_episode*.npz' -type f | wc -l)
reference_count=$(find "$PAYLOAD/lmvla/lmwm/data/robotwin_dinov3base" \
  -name 'ep*.npz' -type f | wc -l)
test "$query_count" -eq 600
test "$reference_count" -eq 1200

printf 'phase=transfer query_count=%s reference_count=%s\n' "$query_count" "$reference_count"
env SRC="$PAYLOAD" DST="$REMOTE_STAGE" SYNC_EVAL_ONLY=0 \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-sidecar-v1 \
  bash "$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh"

printf 'phase=remote-preflight\n'
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "test -f $(printf %q "$REMOTE_STAGE/logs/r4/training/outcome_free_query_manifest.json") && \
   test \"\$(find $(printf %q "$REMOTE_STAGE/lmvla/lawam_local/results/eval_runs/robotwin") -name 'query_episode*.npz' -type f | wc -l)\" -eq 600 && \
   test \"\$(find $(printf %q "$REMOTE_STAGE/lmvla/lmwm/data/robotwin_dinov3base") -name 'ep*.npz' -type f | wc -l)\" -eq 1200 && \
   test -f /vePFS-North-E/vis_robot/workspace/tim/models/dinov3-vitb16-pretrain-lvd1689m/model.safetensors && \
   echo '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b  /vePFS-North-E/vis_robot/workspace/tim/models/dinov3-vitb16-pretrain-lvd1689m/model.safetensors' | sha256sum -c - && \
   mkdir -p $(printf %q "$(dirname "$REMOTE_MARKER")") && \
   printf 'validated=%s\\nstage=%s\\nquery_count=600\\nreference_count=1200\\n' \
     \"\$(date -u +%FT%TZ)\" $(printf %q "$REMOTE_STAGE") >$(printf %q "$REMOTE_MARKER")"
mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nquery_count=600\nreference_count=1200\n' \
  "$(date -u +%FT%TZ)" "$REMOTE_STAGE" >"$LOCAL_MARKER"
printf 'phase=complete marker=%s\n' "$LOCAL_MARKER"
