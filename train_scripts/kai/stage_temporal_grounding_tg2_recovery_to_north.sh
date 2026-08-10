#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}
STAGE=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843
MANIFEST_REL=lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_v1.json
FILES=(
  "$MANIFEST_REL"
  lmvla/lmwm/scripts/verify_temporal_grounding_tg2_recovery_bundle.py
  train_scripts/kai/run_temporal_grounding_tg2_recovery_train.sh
)
LOCAL_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg2_recovery_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/temporal_grounding_tg2_recovery_north_stage.ok

for relative in "${FILES[@]}"; do
  test -s "$REPO/$relative"
done

tar -C "$REPO" -cf - "${FILES[@]}" | ssh -p "$PORT" -o BatchMode=yes "$HOST" \
  "set -euo pipefail; incoming=\$(mktemp -d '$STAGE/.tg2r-stage.XXXXXX'); tar -C \"\$incoming\" -xf -; for relative in ${FILES[*]}; do mkdir -p '$STAGE'/\"\$(dirname \"\$relative\")\"; install -m 0664 \"\$incoming/\$relative\" '$STAGE'/\"\$relative\"; done; chmod 0755 '$STAGE/train_scripts/kai/run_temporal_grounding_tg2_recovery_train.sh' '$STAGE/lmvla/lmwm/scripts/verify_temporal_grounding_tg2_recovery_bundle.py'; rm -rf \"\$incoming\""

ssh -p "$PORT" -o BatchMode=yes "$HOST" bash -s -- "$STAGE" "$REMOTE_MARKER" <<'REMOTE'
set -euo pipefail
stage=$1
marker=$2
test "$(git -C "$stage" rev-parse HEAD)" = 11fb84349809b30ddc785dc99105080540d000c2
test "$(git -C "$stage/lmvla/lawam" rev-parse HEAD)" = 71803a3f8b0e55679a4557ef6af80a76604f277a
"$stage/kai0/.venv/bin/python" \
  "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_tg2_recovery_bundle.py" \
  --repo "$stage" \
  --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2_recovery_v1.json"
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nouter_commit=11fb84349809b30ddc785dc99105080540d000c2\nlawam_commit=71803a3f8b0e55679a4557ef6af80a76604f277a\n' \
  "$(date -u +%FT%TZ)" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
