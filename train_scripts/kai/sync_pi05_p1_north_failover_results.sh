#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
readonly REMOTE=root@124.174.16.237
readonly SSH_PORT=16370
readonly STAGE_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/pi05_p1_failover_20260804T1034Z
readonly REMOTE_REPORT=$STAGE_ROOT/pi05_p1_north_pair_training_report.json
readonly LOCAL_REPORT=$REPO/logs/pi05_p1_failover/north_pair_training_report.json
readonly MATERIALIZED_MARKER=$REPO/logs/resource_markers/pi05_p1_north_failover_materialized.ok
readonly RSYNC_PARALLELISM=${RSYNC_PARALLELISM:-6}
readonly RSYNC_RSH="ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1"
SSH=(ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1)

exec 9>"$REPO/logs/pi05_p1_failover/north_result_sync.lock"
if ! flock -n 9; then
  printf 'another P1 North result sync is already running\n' >&2
  exit 1
fi

rm -f "$MATERIALIZED_MARKER"

"${SSH[@]}" "$REMOTE" "test \"\$(jq -r '.pair_training_complete' '$REMOTE_REPORT')\" = true"
mkdir -p "$(dirname "$LOCAL_REPORT")"
rsync -a -e "$RSYNC_RSH" "$REMOTE:$REMOTE_REPORT" "$LOCAL_REPORT"

ensure_writable_checkpoint_parent() {
  local local_parent=$1
  local config_dir
  config_dir=$(dirname "$local_parent")
  if [[ -w "$local_parent" ]]; then
    return 0
  fi
  if [[ ! -w "$(dirname "$config_dir")" ]]; then
    printf 'checkpoint root is not writable: %s\n' "$(dirname "$config_dir")" >&2
    exit 1
  fi
  local archive=${config_dir}.pre-north-root-owned-20260805
  if [[ -e "$config_dir" && ! -e "$archive" ]]; then
    mv "$config_dir" "$archive"
  elif [[ -e "$config_dir" ]]; then
    printf 'canonical checkpoint directory remains unwritable with archive present: %s\n' \
      "$config_dir" >&2
    exit 1
  fi
  mkdir -p "$local_parent"
  printf 'preserved root-owned checkpoints at %s\n' "$archive"
}

sync_checkpoint() {
  local remote_relative=$1
  local local_parent=$2
  local final_name=49999
  local incoming=$local_parent/.north-incoming-$final_name
  local final=$local_parent/$final_name
  ensure_writable_checkpoint_parent "$local_parent"
  if [[ -s "$final/_CHECKPOINT_METADATA" && -s "$final/params/_METADATA" ]]; then
    return 0
  fi
  if [[ -e "$final" ]]; then
    printf 'refusing to replace incomplete final checkpoint: %s\n' "$final" >&2
    exit 1
  fi
  mkdir -p "$incoming"
  local file_list
  file_list=$(mktemp)
  "${SSH[@]}" "$REMOTE" \
    "find '$STAGE_ROOT/$remote_relative' -type f -printf '%P\\n'" \
    >"$file_list"
  if [[ ! -s "$file_list" ]]; then
    printf 'remote checkpoint contains no files: %s\n' "$remote_relative" >&2
    rm -f "$file_list"
    exit 1
  fi
  local active=0
  local relative
  while IFS= read -r relative; do
    mkdir -p "$incoming/$(dirname "$relative")"
    rsync -a --partial --partial-dir=.rsync-partial --info=stats2 \
      -e "$RSYNC_RSH" \
      "$REMOTE:$STAGE_ROOT/$remote_relative/$relative" \
      "$incoming/$relative" &
    active=$((active + 1))
    if ((active >= RSYNC_PARALLELISM)); then
      if ! wait -n; then
        jobs -pr | xargs -r kill
        wait || true
        rm -f "$file_list"
        return 1
      fi
      active=$((active - 1))
    fi
  done <"$file_list"
  rm -f "$file_list"
  while ((active > 0)); do
    if ! wait -n; then
      jobs -pr | xargs -r kill
      wait || true
      return 1
    fi
    active=$((active - 1))
  done
  test -s "$incoming/_CHECKPOINT_METADATA"
  test -s "$incoming/params/_METADATA"
  test -s "$incoming/train_state/_METADATA"
  mv "$incoming" "$final"
}

sync_checkpoint \
  kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999 \
  "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000"
sync_checkpoint \
  kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000/49999 \
  "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000"

mkdir -p "$(dirname "$MATERIALIZED_MARKER")"
marker_tmp=${MATERIALIZED_MARKER}.tmp.$$
printf 'materialized_at=%s\nreport=%s\na0=%s\ncandidate=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$LOCAL_REPORT" \
  "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999" \
  "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000/49999" \
  >"$marker_tmp"
mv "$marker_tmp" "$MATERIALIZED_MARKER"

printf 'P1 North failover checkpoints materialized atomically; scheduler may evaluate them.\n'
