#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843}
ARM=${TG4_ARM:?set TG4_ARM}
SEED=${TG4_TRAIN_SEED:?set TG4_TRAIN_SEED}
RUN_ID=temporal_grounding_tg4_${ARM}_seed${SEED}
REMOTE_BASE=$NORTH_REPO/lmvla/lawam/results/Checkpoints/robotwin
LOCAL_BASE=$REPO/lmvla/lawam/results/Checkpoints/robotwin
REMOTE_INIT=$NORTH_REPO/logs/temporal_grounding/tg4/initialization/$RUN_ID.json
REMOTE_ORDER=$NORTH_REPO/logs/temporal_grounding/tg4/data_order/$RUN_ID
SIDECAR_BASE=$REPO/logs/resource_scheduler_local/temporal_grounding_tg4_sidecars/$RUN_ID
LOCAL_INIT=$SIDECAR_BASE/initialization.json
LOCAL_ORDER=$SIDECAR_BASE/data_order
MARKER=$REPO/logs/resource_markers/${RUN_ID}_train_materialized.ok
LOCK=$REPO/logs/locks/temporal_grounding_tg4_materialize.lock
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}

case "$ARM" in
  clean_base|future_off|auxiliary_only|conditioning_only|parameter_matched_null|full) ;;
  *) exit 2 ;;
esac
case "$SEED" in 1100|1101|1102) ;; *) exit 2 ;; esac
mkdir -p "$(dirname "$LOCK")" "$(dirname "$MARKER")" "$LOCAL_BASE" "$SIDECAR_BASE"
exec 9>"$LOCK"
flock 9

mapfile -t sources < <(
  ssh -p "$PORT" -o BatchMode=yes "$HOST" \
    "find '$REMOTE_BASE' -mindepth 1 -maxdepth 1 -type d -name '*+$RUN_ID' -print | sort"
)
if [[ "${#sources[@]}" -ne 1 ]]; then
  echo "expected exactly one North TG4 run for $RUN_ID, found ${#sources[@]}" >&2
  exit 3
fi
SRC=${sources[0]}
DST=$LOCAL_BASE/$(basename "$SRC")

verify_run() {
  local root=$1
  test -s "$root/config.json"
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
  verify_run "$DST" || { echo "incomplete existing TG4 destination: $DST" >&2; exit 4; }
else
  SRC="$SRC" DST="$DST" bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"
  verify_run "$DST"
fi

sync_remote_file() {
  local src=$1 dst=$2 expected temporary actual
  expected=$(ssh -p "$PORT" -o BatchMode=yes "$HOST" "sha256sum '$src'" | awk '{print $1}')
  temporary=$(mktemp "${dst}.incoming.XXXXXX")
  ssh -p "$PORT" -o BatchMode=yes "$HOST" "cat '$src'" >"$temporary"
  actual=$(sha256sum "$temporary" | awk '{print $1}')
  test "$actual" = "$expected"
  chmod 0664 "$temporary"
  mv "$temporary" "$dst"
}

tree_digest_remote() {
  ssh -p "$PORT" -o BatchMode=yes "$HOST" \
    "cd '$1' && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum" \
    | awk '{print $1}'
}

tree_digest_local() {
  (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum) \
    | awk '{print $1}'
}

sync_remote_file "$REMOTE_INIT" "$LOCAL_INIT"
remote_order_digest=$(tree_digest_remote "$REMOTE_ORDER")
local_order_digest=""
if [[ -d "$LOCAL_ORDER" ]]; then
  local_order_digest=$(tree_digest_local "$LOCAL_ORDER")
fi
if [[ "$local_order_digest" != "$remote_order_digest" ]]; then
  if [[ -e "$LOCAL_ORDER" ]]; then
    mv "$LOCAL_ORDER" "${LOCAL_ORDER}.quarantine.$(date -u +%Y%m%d_%H%M%S)"
  fi
  SRC="$REMOTE_ORDER" DST="$LOCAL_ORDER" SYNC_PARALLEL_LARGE_FILES=0 \
    bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"
fi

cat >"$MARKER" <<EOF
materialized=$(date -u +%FT%TZ)
run_id=$RUN_ID
parent_resource=Robot-North-H20
source=$SRC
destination=$DST
final_model_bytes=$(stat -c %s "$DST/final_model/pytorch_model.pt")
optimizer_bytes=$(stat -c %s "$DST/checkpoints/steps_20000_state/optimizer.bin")
initialization_sha256=$(sha256sum "$LOCAL_INIT" | awk '{print $1}')
data_order_tree_sha256=$(tree_digest_local "$LOCAL_ORDER")
EOF
chmod 0664 "$MARKER"
