#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
GPU_COUNT=${LOCAL_GPU_COUNT:-2}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
ARTIFACT=$REPO/lmvla/lmwm/data/pi05_crave_r0_v1
FEATURE_DIR=$REPO/lmvla/lmwm/data/robotwin_dinov3base
EXTRACTOR=$REPO/lmvla/lmwm/scripts/robotwin_dinov3base_extract.py
MARKER=$REPO/logs/resource_markers/pi05_crave_r0_features.ok
LOG_DIR=$REPO/logs/crave_r0/features

test -f "$ARTIFACT/READY_SELECTION"
test -f "$ARTIFACT/selection_manifest.json"
test -f "$ARTIFACT/required_episodes.txt"
test "$GPU_COUNT" -ge 1
mkdir -p "$LOG_DIR" "$(dirname "$MARKER")"

mapfile -t MISSING < <(python3 - "$ARTIFACT/required_episodes.txt" "$FEATURE_DIR" <<'PY'
from pathlib import Path
import sys
required = [int(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
root = Path(sys.argv[2])
for episode in required:
    if not (root / f"ep{episode}.npz").is_file():
        print(episode)
PY
)

if [[ ${#MISSING[@]} -gt 0 ]]; then
  PIDS=()
  for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    EPS=$(python3 - "$gpu" "$GPU_COUNT" "${MISSING[@]}" <<'PY'
import sys
shard, count = map(int, sys.argv[1:3])
episodes = sys.argv[3:]
print(",".join(episodes[shard::count]))
PY
)
    [[ -n "$EPS" ]] || continue
    CUDA_VISIBLE_DEVICES=$((GPU_INDEX_OFFSET + gpu)) \
      CRAVE_REPO=$REPO \
      RT_REPO=$REPO \
      "$REPO/kai0/.venv/bin/python" -u "$EXTRACTOR" \
        --eps "$EPS" --out "$FEATURE_DIR" \
        >"$LOG_DIR/gpu${gpu}.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
fi

python3 - "$ARTIFACT/required_episodes.txt" "$FEATURE_DIR" \
  "$REPO/../VLANeXt-main/datasets/robotwin2.0_official_prompts_v21/meta/episodes.jsonl" \
  "$MARKER" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

required = [int(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
root = Path(sys.argv[2])
lengths = {}
with Path(sys.argv[3]).open() as stream:
    for line in stream:
        row = json.loads(line)
        lengths[int(row["episode_index"])] = int(row["length"])
problems = []
frames = 0
for episode in required:
    path = root / f"ep{episode}.npz"
    if not path.is_file():
        problems.append(f"missing ep{episode}")
        continue
    values = np.load(path)["pooled"]
    if (
        values.ndim != 2
        or values.shape[1] != 768
        or len(values) != lengths[episode]
        or not np.isfinite(values).all()
    ):
        problems.append(f"invalid ep{episode}: shape={values.shape}")
    frames += len(values)
if problems:
    raise RuntimeError("; ".join(problems[:20]))
marker = Path(sys.argv[4])
temporary = marker.with_suffix(marker.suffix + ".tmp")
temporary.write_text(json.dumps({"episodes": len(required), "frames": frames}) + "\n")
temporary.replace(marker)
print(json.dumps({"episodes": len(required), "frames": frames}, indent=2))
PY
