#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
REMOTE="${TASK_N_NORTH_REMOTE:-root@124.174.16.237}"
PORT="${TASK_N_NORTH_PORT:-16370}"
REMOTE_PARENT="${TASK_N_NORTH_PARENT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_N/self_built}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SYNC_STARTED_AT="$(date -u +%FT%TZ)"
INCOMING="$REMOTE_PARENT/.nail_v5_0806_0807_319.incoming.$STAMP"
LOG="$ROOT/logs/task_n_v5_0806_0807_319_sync_north.log"
STATUS="$ROOT/logs/task_n_v5_0806_0807_319_sync_north.status"
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p "$PORT")
RSYNC_SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p $PORT"
SCP=(scp -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -P "$PORT")
FREEZE_REPORT="$ROOT/docs/training/analysis/task_n_v5_0806_0807_319_freeze.json"
NORTH_VERIFIER="$ROOT/train_scripts/kai/data/verify_task_n_v5_0806_0807_319_north.py"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "RUNNING start=$SYNC_STARTED_AT incoming=$INCOMING" > "$STATUS"
trap 'rc=$?; echo "FINISHED rc=$rc end=$(date -u +%FT%TZ) incoming=$INCOMING" > "$STATUS"; exit $rc' EXIT

TRAIN=nail_v5_0806_0807_319_joint14_train
VAL=nail_v5_0806_0807_319_joint14_val
LOCAL_PARENT="$ROOT/kai0/data/Task_N/self_built"
test -d "$LOCAL_PARENT/$TRAIN"
test -d "$LOCAL_PARENT/$VAL"
test -f "$FREEZE_REPORT"
test -f "$NORTH_VERIFIER"

"${SSH[@]}" "$REMOTE" "set -e; test ! -e '$REMOTE_PARENT/$TRAIN'; test ! -e '$REMOTE_PARENT/$VAL'; mkdir -p '$INCOMING'"
for dataset in "$TRAIN" "$VAL"; do
  rsync -a --partial --delete --checksum --info=progress2 \
    -e "$RSYNC_SSH" "$LOCAL_PARENT/$dataset/" "$REMOTE:$INCOMING/$dataset/"
done

local_counts="$({
  find "$LOCAL_PARENT/$TRAIN" -type f -printf 'train %s\n'
  find "$LOCAL_PARENT/$VAL" -type f -printf 'val %s\n'
} | awk '{files += 1; bytes += $2} END {print files, bytes}')"
remote_counts="$("${SSH[@]}" "$REMOTE" "{
  find '$INCOMING/$TRAIN' -type f -printf 'train %s\\n'
  find '$INCOMING/$VAL' -type f -printf 'val %s\\n'
} | awk '{files += 1; bytes += \\$2} END {print files, bytes}'")"
test "$local_counts" = "$remote_counts"

"${SCP[@]}" "$FREEZE_REPORT" "$REMOTE:$INCOMING/freeze.json"
"${SCP[@]}" "$NORTH_VERIFIER" "$REMOTE:$INCOMING/verify_north.py"
"${SSH[@]}" "$REMOTE" "python3 '$INCOMING/verify_north.py' \
  --parent '$INCOMING' --freeze-report '$INCOMING/freeze.json' \
  --sync-started-at '$SYNC_STARTED_AT'"

"${SSH[@]}" "$REMOTE" "set -e
  test \"\$(find '$INCOMING/$TRAIN/data' -name '*.parquet' | wc -l)\" -eq 287
  test \"\$(find '$INCOMING/$VAL/data' -name '*.parquet' | wc -l)\" -eq 32
  test \"\$(find '$INCOMING/$TRAIN/videos' -name '*.mp4' | wc -l)\" -eq 861
  test \"\$(find '$INCOMING/$VAL/videos' -name '*.mp4' | wc -l)\" -eq 96
  test -f '$INCOMING/$TRAIN/norm_stats.json'
  test -f '$INCOMING/$TRAIN/NORTH_SYNC_OK.json'
  rm '$INCOMING/freeze.json' '$INCOMING/verify_north.py'
  mv '$INCOMING/$TRAIN' '$REMOTE_PARENT/$TRAIN'
  mv '$INCOMING/$VAL' '$REMOTE_PARENT/$VAL'
  rmdir '$INCOMING'"

echo "SYNC_OK end=$(date -u +%FT%TZ) files_bytes=$local_counts"
