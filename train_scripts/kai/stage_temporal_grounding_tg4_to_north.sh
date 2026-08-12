#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}
STAGE=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843
LOCAL_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/temporal_grounding_tg4_north_stage.ok

MAIN_FILES=(
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_source_decomposition_v1.json
  lmvla/lmwm/scripts/verify_temporal_grounding_tg4_bundle.py
  train_scripts/kai/run_temporal_grounding_tg4_train.sh
)
for relative in "${MAIN_FILES[@]}"; do
  test -s "$REPO/$relative"
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/lmvla/lawam"
git -C "$REPO/lmvla/lawam" archive HEAD | tar -C "$tmp/lmvla/lawam" -xf -
tar -C "$REPO" -cf - "${MAIN_FILES[@]}" | tar -C "$tmp" -xf -

tar -C "$tmp" -cf - . | ssh -p "$PORT" -o BatchMode=yes "$HOST" \
  "set -euo pipefail; incoming=\$(mktemp -d '$STAGE/.tg4-stage.XXXXXX'); tar -C \"\$incoming\" -xf -; cp -a \"\$incoming\"/. '$STAGE'/; rm -rf \"\$incoming\"; chmod 0755 '$STAGE/train_scripts/kai/run_temporal_grounding_tg4_train.sh' '$STAGE/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_bundle.py'"

ssh -p "$PORT" -o BatchMode=yes "$HOST" bash -s -- "$STAGE" "$REMOTE_MARKER" <<'REMOTE'
set -euo pipefail
stage=$1
marker=$2
"$stage/kai0/.venv/bin/python" \
  "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_bundle.py" \
  --repo "$stage" \
  --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_source_decomposition_v1.json"
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nsource_main=%s\nsource_lawam=%s\n' \
  "$(date -u +%FT%TZ)" \
  602117acff2e86b51336eddcff453ce6cfbd06dc \
  4bcf17f67b71d700885e2e279700d373dc5ecbfd >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
