#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843}
ARM=${TG2_ARM:?set TG2_ARM}
SEED=${TG2_TRAIN_SEED:?set TG2_TRAIN_SEED}
RUN_ID=temporal_grounding_tg2_${ARM}_seed${SEED}
REMOTE_BASE=$NORTH_REPO/lmvla/lawam/results/Checkpoints/robotwin
LOCAL_BASE=$REPO/lmvla/lawam/results/Checkpoints/robotwin
MARKER=$REPO/logs/resource_markers/${RUN_ID}_materialized.ok
LOCK=$REPO/logs/locks/temporal_grounding_tg2_materialize.lock

case "$ARM" in future_off|fixed_endpoint|raw_milestone) ;; *) exit 2 ;; esac
case "$SEED" in 1000|1001|1002) ;; *) exit 2 ;; esac
mkdir -p "$(dirname "$LOCK")" "$(dirname "$MARKER")" "$LOCAL_BASE"
exec 9>"$LOCK"
flock 9

mapfile -t sources < <(
  ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
    "find '$REMOTE_BASE' -mindepth 1 -maxdepth 1 -type d -name '*+$RUN_ID' -print | sort"
)
if [[ "${#sources[@]}" -ne 1 ]]; then
  echo "expected exactly one North TG2 run for $RUN_ID, found ${#sources[@]}" >&2
  exit 3
fi
SRC=${sources[0]}
DST=$LOCAL_BASE/$(basename "$SRC")

verify_run() {
  local root=$1
  test -s "$root/config.yaml"
  test -s "$root/dataset_statistics.json"
  test -s "$root/final_model/pytorch_model.pt"
  test -s "$root/checkpoints/steps_20000_state/optimizer.bin"
  test -s "$root/checkpoints/steps_20000_state/trainer_state.json"
  test "$(stat -c %s "$root/final_model/pytorch_model.pt")" -ge 1000000000
  test "$(stat -c %s "$root/checkpoints/steps_20000_state/optimizer.bin")" -ge 1000000000
  test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps"])' \
    "$root/checkpoints/steps_20000_state/trainer_state.json")" = 20000
}

if [[ -e "$DST" ]]; then
  verify_run "$DST" || {
    echo "refusing to replace incomplete existing destination: $DST" >&2
    exit 4
  }
else
  SRC="$SRC" DST="$DST" \
    bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"
  verify_run "$DST"
fi

cat >"$MARKER" <<EOF
materialized=$(date -u +%FT%TZ)
run_id=$RUN_ID
source=$SRC
destination=$DST
final_model_bytes=$(stat -c %s "$DST/final_model/pytorch_model.pt")
optimizer_bytes=$(stat -c %s "$DST/checkpoints/steps_20000_state/optimizer.bin")
EOF
