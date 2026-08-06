#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
STAGE=$NORTH_REPO/.staging/pi05_r4_eval_north_v1/repo
PAYLOAD=$REPO/logs/r4/north_python_abi_repair/payload
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_north_python_abi_repair.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/pi05_r4_north_python_abi_repair.ok
SYNC=$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh

rm -rf "$PAYLOAD"
mkdir -p "$PAYLOAD"
for file in \
  lerobot_pi05_action_bridge.py \
  robotwin_python_wrapper_north.sh \
  run_pi05_r4_formal_eval.sh \
  serve_lerobot_pi05.py; do
  cp -a "$REPO/train_scripts/kai/eval/$file" "$PAYLOAD/"
done

env SRC="$PAYLOAD" DST="$STAGE/train_scripts/kai/eval" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-north-python-abi-repair \
  bash "$SYNC"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$STAGE" "$REMOTE_MARKER" <<'REMOTE'
set -Eeuo pipefail
stage=$1
marker=$2
wrapper=$stage/train_scripts/kai/eval/robotwin_python_wrapper_north.sh
bad_site=$stage/runtime/venv/lib/python3.12/site-packages
keep_source=$stage/kai0/src

chmod 0755 "$wrapper"
test -x "$wrapper"
PYTHONPATH="$bad_site:$keep_source" "$wrapper" - "$bad_site" "$keep_source" <<'PY'
import pathlib
import sys

import numpy

bad_site = pathlib.Path(sys.argv[1]).resolve()
keep_source = pathlib.Path(sys.argv[2]).resolve()
numpy_path = pathlib.Path(numpy.__file__).resolve()
assert sys.version_info[:2] == (3, 10), sys.version
assert str(bad_site) not in sys.path, sys.path
assert str(keep_source) in sys.path
assert "/python3.12/" not in str(numpy_path), numpy_path
print(f"python={sys.version.split()[0]} numpy={numpy.__version__} numpy_path={numpy_path}")
PY

# Both prior attempts failed before producing an episode. Remove only those
# invalid scheduler trees so the fixed-protocol retry starts from 0/24 cells.
for arm in terminal_outcome outcome_free_crave; do
  rm -rf "$stage/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_${arm}_seed1000"
  rm -f "$stage/logs/resource_markers/pi05_r4_${arm}_seed1000.ok"
done
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nstage=%s\npython_abi=3.10\nfiltered_abi=3.12\n' \
  "$(date -u +%FT%TZ)" "$stage" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
