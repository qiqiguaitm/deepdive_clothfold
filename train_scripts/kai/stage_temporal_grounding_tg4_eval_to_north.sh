#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
STAGE=$NORTH_REPO/.staging/temporal_grounding_tg4_eval_v1/repo
TRAIN_STAGE=$NORTH_REPO/.staging/temporal_grounding_11fb843
LOCAL_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_eval_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/temporal_grounding_tg4_eval_north_stage.ok
INTEGRITY=$REPO/logs/temporal_grounding/tg4/training_integrity.json
INTEGRITY_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_training_integrity.ok
EVAL_MANIFEST=$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json
SYNC=$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh

for path in "$INTEGRITY" "$INTEGRITY_MARKER" "$EVAL_MANIFEST" "$SYNC"; do
  test -s "$path"
done

payload=$(mktemp -d)
checkpoint_manifest=$(mktemp)
trap 'rm -rf "$payload"; rm -f "$checkpoint_manifest"' EXIT
mkdir -p "$payload/lmvla/lawam"
git -C "$REPO/lmvla/lawam" archive HEAD | tar -C "$payload/lmvla/lawam" -xf -

mapfile -t FILES < <(
  python3 - "$EVAL_MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json")
for relative in sorted(manifest["sha256"]):
    print(relative)
PY
)
for relative in "${FILES[@]}"; do
  test -s "$REPO/$relative"
  mkdir -p "$payload/$(dirname "$relative")"
  cp -a "$REPO/$relative" "$payload/$relative"
done
mkdir -p "$payload/logs/temporal_grounding/tg4" "$payload/logs/resource_markers"
cp -a "$INTEGRITY" "$payload/logs/temporal_grounding/tg4/training_integrity.json"
cp -a "$INTEGRITY_MARKER" "$payload/logs/resource_markers/"

python3 - "$REPO" "$INTEGRITY" > "$checkpoint_manifest" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
payload = json.loads(pathlib.Path(sys.argv[2]).read_text())
for run_id, spec in sorted(payload["runs"].items()):
    run = repo / spec["run"]
    model = run / "final_model/pytorch_model.pt"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    print(run_id, run.name, digest.hexdigest(), model.stat().st_size, sep="\t")
PY
test "$(wc -l < "$checkpoint_manifest")" -eq 18
cp -a "$checkpoint_manifest" \
  "$payload/logs/temporal_grounding/tg4/north_eval_checkpoints.tsv"

tar -C "$payload" -cf - . | ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "set -Eeuo pipefail; mkdir -p $(printf %q "$STAGE"); incoming=\$(mktemp -d $(printf %q "$STAGE/.code.XXXXXX")); tar -C \"\$incoming\" -xf -; cp -a \"\$incoming\"/. $(printf %q "$STAGE")/; rm -rf \"\$incoming\"; mkdir -p $(printf %q "$STAGE/lmvla/lawam_local/results/Checkpoints/robotwin") $(printf %q "$STAGE/lmvla/lawam_local/results/eval_runs/robotwin") $(printf %q "$STAGE/lmvla/lawam_local"); for d in ckpts_dl dataset weights logs; do ln -sfn $(printf %q "$NORTH_REPO/lmvla/lawam_local")/\"\$d\" $(printf %q "$STAGE/lmvla/lawam_local")/\"\$d\"; done; mkdir -p $(printf %q "$STAGE/kai0"); ln -sfn $(printf %q "$TRAIN_STAGE/kai0/.venv") $(printf %q "$STAGE/kai0/.venv"); chmod 0755 $(printf %q "$STAGE/train_scripts/kai/eval/run_temporal_grounding_tg4_eval.sh") $(printf %q "$STAGE/train_scripts/kai/eval/robotwin_python_wrapper_north.sh")"

reused=0
uploaded=0
while IFS=$'\t' read -r run_id run_name expected size; do
  local_model=$(find "$REPO/lmvla/lawam/results/Checkpoints/robotwin" -maxdepth 3 \
    -path "*+${run_id}/final_model/pytorch_model.pt" -print -quit)
  test -n "$local_model"
  remote_source=$NORTH_REPO/lmvla/lawam/results/Checkpoints/robotwin/$run_name/final_model/pytorch_model.pt
  remote_target=$STAGE/lmvla/lawam_local/results/Checkpoints/robotwin/$run_name/final_model/pytorch_model.pt
  if ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
      "test -f $(printf %q "$remote_source") && test \"\$(stat -c %s $(printf %q "$remote_source"))\" = $(printf %q "$size") && echo $(printf %q "$expected  $remote_source") | sha256sum -c - >/dev/null"; then
    ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
      "mkdir -p $(printf %q "$(dirname "$remote_target")"); rm -f $(printf %q "$remote_target"); ln $(printf %q "$remote_source") $(printf %q "$remote_target")"
    reused=$((reused + 1))
  else
    env SRC="$(dirname "$local_model")" DST="$(dirname "$remote_target")" \
      SYNC_EVAL_ONLY=1 NORTH_TOS_PREFIX="temp/deepdive_kai0/tg4-eval-north-v1/$run_id" \
      bash "$SYNC"
    uploaded=$((uploaded + 1))
  fi
done < "$checkpoint_manifest"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$STAGE" "$REMOTE_MARKER" \
  "$STAGE/logs/temporal_grounding/tg4/north_eval_checkpoints.tsv" \
  "$reused" "$uploaded" <<'REMOTE'
set -Eeuo pipefail
stage=$1
marker=$2
manifest=$3
reused=$4
uploaded=$5
control=$stage/kai0/.venv/bin/python
test -x "$control"
bash "$stage/lmvla/lmwam/env/heal_lawam_symlinks.sh"
while IFS=$'\t' read -r run_id run_name expected size; do
  model=$stage/lmvla/lawam_local/results/Checkpoints/robotwin/$run_name/final_model/pytorch_model.pt
  test "$(stat -c %s "$model")" = "$size"
  echo "$expected  $model" | sha256sum -c - >/dev/null
done < "$manifest"
"$control" "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_evaluation.py" \
  --repo "$stage" \
  --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json"
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nstage=%s\ncheckpoint_count=18\nreused_north=%s\nuploaded=%s\n' \
  "$(date -u +%FT%TZ)" "$stage" "$reused" "$uploaded" > "$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\ncheckpoint_count=18\nreused_north=%s\nuploaded=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" "$reused" "$uploaded" > "$LOCAL_MARKER"
