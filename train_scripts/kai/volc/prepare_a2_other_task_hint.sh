#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
NORTH_REPO=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
REMOTE=root@124.174.16.237
PORT=16370
SOURCE=$NORTH_REPO/lmvla/lmwm/data/pi05_hint/robotwin_so400m/hint.npz
REMOTE_DIR=$NORTH_REPO/lmvla/lmwm/data/pi05_hint/robotwin_so400m_other_task
LOCAL_DIR=$REPO/lmvla/lmwm/data/pi05_hint/robotwin_so400m_other_task
OUTPUT=handover_ep4497_frame0.npy
STATUS=$REPO/logs/a2_other_task_hint.status
MARKER=$REPO/logs/resource_markers/a2_other_task_hint.ok
LOG=$REPO/logs/a2_other_task_hint.log

mkdir -p "$LOCAL_DIR" "$(dirname "$MARKER")"
exec >> "$LOG" 2>&1
trap 'rc=$?; echo "FINISHED rc=$rc end=$(date -u +%FT%TZ)" > "$STATUS"' EXIT

if [[ -f "$MARKER" && -f "$LOCAL_DIR/$OUTPUT" ]]; then
  echo "ALREADY_COMPLETE output=$LOCAL_DIR/$OUTPUT" > "$STATUS"
  exit 0
fi

echo "RUNNING start=$(date -u +%FT%TZ)" > "$STATUS"
ssh -o BatchMode=yes -p "$PORT" "$REMOTE" "SOURCE='$SOURCE' REMOTE_DIR='$REMOTE_DIR' OUTPUT='$OUTPUT' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
mkdir -p "$REMOTE_DIR"
unzip -o "$SOURCE" hint.npy episode_index.npy -d "$REMOTE_DIR"
python3 - "$REMOTE_DIR" "$OUTPUT" <<'PY'
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
output = root / sys.argv[2]
episodes = np.load(root / "episode_index.npy", mmap_mode="r")
rows = np.flatnonzero(episodes == 4497)
if rows.size == 0:
    raise RuntimeError("handover episode 4497 is absent from A2 hint archive")
hints = np.load(root / "hint.npy", mmap_mode="r")
hint = np.asarray(hints[int(rows[0])], dtype=np.float32)
if hint.shape != (1152,) or not np.all(np.isfinite(hint)):
    raise RuntimeError(f"invalid extracted hint: shape={hint.shape}, finite={np.all(np.isfinite(hint))}")
np.save(output, hint)
print(f"saved {output} row={int(rows[0])} norm={np.linalg.norm(hint):.6f}")
PY
REMOTE_SCRIPT

scp -q -P "$PORT" "$REMOTE:$REMOTE_DIR/$OUTPUT" "$LOCAL_DIR/$OUTPUT"
python3 - "$LOCAL_DIR/$OUTPUT" <<'PY'
import sys
import numpy as np

hint = np.load(sys.argv[1])
assert hint.shape == (1152,) and np.all(np.isfinite(hint))
print(f"verified {sys.argv[1]} shape={hint.shape} norm={np.linalg.norm(hint):.6f}")
PY
printf 'completed=%s\nlocal=%s\nremote=%s\n' \
  "$(date -u +%FT%TZ)" "$LOCAL_DIR/$OUTPUT" "$REMOTE_DIR/$OUTPUT" > "$MARKER"
