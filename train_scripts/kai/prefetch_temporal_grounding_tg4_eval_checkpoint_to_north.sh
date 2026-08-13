#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
ARM=${TG4_ARM:?TG4_ARM is required}
SEED=${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}
ACCEPTANCE=${TG4_ACCEPTANCE_MARKER:?TG4_ACCEPTANCE_MARKER is required}
OUTPUT=${TG4_PREFETCH_OUTPUT:?TG4_PREFETCH_OUTPUT is required}
SYNC=${TG4_NORTH_SYNC:-$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh}
LOCK=${TG4_PREFETCH_LOCK:-$REPO/logs/locks/temporal_grounding_tg4_eval_north_prefetch.lock}
RUN_ID=temporal_grounding_tg4_${ARM}_seed${SEED}
TASK_ID=${RUN_ID}_train
LOCAL_BASE=$REPO/lmvla/lawam/results/Checkpoints/robotwin
STAGE=$NORTH_REPO/.staging/temporal_grounding_tg4_eval_v1/repo

case "$ARM:$SEED" in
  auxiliary_only:1100|auxiliary_only:1101|parameter_matched_null:1101|parameter_matched_null:1102) ;;
  *) echo "unsupported TG4 East prefetch cell: $ARM seed $SEED" >&2; exit 2 ;;
esac
for path in "$ACCEPTANCE" "$SYNC"; do
  test -s "$path"
done

mapfile -t accepted < <(
  python3 - "$ACCEPTANCE" "$TASK_ID" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
task_id = sys.argv[2]
if payload.get("complete") is not True:
    raise SystemExit("acceptance marker is not complete")
runs = [run for run in payload.get("runs", []) if run.get("task_id") == task_id]
if len(runs) != 1:
    raise SystemExit(f"expected one accepted run for {task_id}, found {len(runs)}")
run = runs[0]
print(
    run["run"],
    run["final_checkpoint_sha256"],
    run["final_checkpoint_bytes"],
    sep="\t",
)
PY
)
test "${#accepted[@]}" -eq 1
IFS=$'\t' read -r accepted_run expected_sha expected_size <<<"${accepted[0]}"

mapfile -t models < <(
  find "$LOCAL_BASE" -maxdepth 3 -type f \
    -path "*+$RUN_ID/final_model/pytorch_model.pt" -print | sort
)
if [[ ${#models[@]} -ne 1 ]]; then
  echo "expected exactly one local accepted model for $RUN_ID, found ${#models[@]}" >&2
  exit 3
fi
model=${models[0]}
run_dir=$(dirname "$(dirname "$model")")
run_name=$(basename "$run_dir")
test "$accepted_run" = "lmvla/lawam/results/Checkpoints/robotwin/$run_name"
test "$(stat -c %s "$model")" = "$expected_size"
echo "$expected_sha  $model" | sha256sum -c - >/dev/null

remote_target=$STAGE/lmvla/lawam_local/results/Checkpoints/robotwin/$run_name/final_model/pytorch_model.pt
mkdir -p "$(dirname "$LOCK")" "$(dirname "$OUTPUT")"
exec 9>"$LOCK"
flock 9

if ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
    "test -f $(printf %q "$remote_target") && test \"\$(stat -c %s $(printf %q "$remote_target"))\" = $(printf %q "$expected_size") && echo $(printf %q "$expected_sha  $remote_target") | sha256sum -c - >/dev/null"; then
  transfer=reused
else
  env SRC="$(dirname "$model")" DST="$(dirname "$remote_target")" \
    SYNC_EVAL_ONLY=1 \
    NORTH_TOS_PREFIX="temp/deepdive_kai0/tg4-eval-north-v1/$RUN_ID" \
    bash "$SYNC"
  transfer=uploaded
fi

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "test -f $(printf %q "$remote_target") && test \"\$(stat -c %s $(printf %q "$remote_target"))\" = $(printf %q "$expected_size") && echo $(printf %q "$expected_sha  $remote_target") | sha256sum -c - >/dev/null"

temporary=$OUTPUT.tmp.$$
printf 'prefetched=%s\nrun_id=%s\nsha256=%s\nbytes=%s\nremote_target=%s\ntransfer=%s\n' \
  "$(date -u +%FT%TZ)" "$RUN_ID" "$expected_sha" "$expected_size" \
  "$remote_target" "$transfer" >"$temporary"
chmod 0664 "$temporary"
mv "$temporary" "$OUTPUT"
