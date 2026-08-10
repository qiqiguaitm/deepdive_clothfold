#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}
STAGE=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843
MANIFEST_REL=lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg2r_future_off_seed1002_primary_duplicate_v1.json
RUNNER_REL=train_scripts/kai/run_temporal_grounding_tg2r_future_off_seed1002_primary_duplicate.sh
FILES=("$MANIFEST_REL" "$RUNNER_REL")
LOCAL_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg2r_seed1002_primary_duplicate_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/temporal_grounding_tg2r_seed1002_primary_duplicate_north_stage.ok

for relative in "${FILES[@]}"; do
  test -s "$REPO/$relative"
done
python3 - "$REPO/$MANIFEST_REL" "$REPO" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
assert manifest["operator_authorized"] is True
assert manifest["formal_attempt"]["credential_profile"] == "primary"
assert manifest["formal_attempt"]["scientific_changes"] == []
for relative, expected in manifest["file_sha256"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"amendment hash mismatch: {relative}: {actual} != {expected}")
PY

tar -C "$REPO" -cf - "${FILES[@]}" | ssh -p "$PORT" -o BatchMode=yes "$HOST" \
  "set -euo pipefail; incoming=\$(mktemp -d '$STAGE/.tg2r-primarydup.XXXXXX'); tar -C \"\$incoming\" -xf -; for relative in ${FILES[*]}; do mkdir -p '$STAGE'/\"\$(dirname \"\$relative\")\"; install -m 0664 \"\$incoming/\$relative\" '$STAGE'/\"\$relative\"; done; chmod 0755 '$STAGE/$RUNNER_REL'; rm -rf \"\$incoming\""

ssh -p "$PORT" -o BatchMode=yes "$HOST" bash -s -- "$STAGE" "$MANIFEST_REL" "$REMOTE_MARKER" <<'REMOTE'
set -euo pipefail
stage=$1
manifest_rel=$2
marker=$3
test "$(git -C "$stage" rev-parse HEAD)" = 11fb84349809b30ddc785dc99105080540d000c2
test "$(git -C "$stage/lmvla/lawam" rev-parse HEAD)" = 71803a3f8b0e55679a4557ef6af80a76604f277a
python3 - "$stage/$manifest_rel" "$stage" <<'PY'
import hashlib
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
for relative, expected in manifest["file_sha256"].items():
    if relative.endswith(".yaml"):
        continue
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert actual == expected, (relative, actual, expected)
PY
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nprotocol=temporal_grounding_tg2r_future_off_seed1002_primary_duplicate_v1\n' \
  "$(date -u +%FT%TZ)" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
