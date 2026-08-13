#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
SCHEDULER_SESSION=${SCHEDULER_SESSION:-resource_scheduler}
MONITOR_SESSION=${MONITOR_SESSION:-paper_todo_hourly_monitor}
SCHEDULER_LOG=${SCHEDULER_LOG:-$REPO/logs/paper_todo_runtime_supervisor.log}
STATUS_FILE=${STATUS_FILE:-$REPO/logs/paper_todo_runtime_supervisor.status}
LATEST_AUDIT=${LATEST_AUDIT:-$REPO/logs/paper_todo_hourly_monitor_latest.json}
LOCK_FILE=${LOCK_FILE:-$REPO/logs/paper_todo_runtime_supervisor.lock}

mkdir -p "$(dirname "$SCHEDULER_LOG")"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$SCHEDULER_LOG"
}

write_status() {
  local state=$1
  local scheduler_alive=$2
  local monitor_alive=$3
  local temporary="${STATUS_FILE}.tmp.$$"
  printf 'pid=%s state=%s last_check=%s scheduler_alive=%s monitor_alive=%s\n' \
    "$$" "$state" "$(date -u +%FT%TZ)" "$scheduler_alive" "$monitor_alive" \
    >"$temporary"
  chmod 0664 "$temporary"
  mv -f "$temporary" "$STATUS_FILE"
}

todo_complete() {
  [[ -f "$LATEST_AUDIT" ]] || return 1
  "$REPO/kai0/.venv/bin/python" - "$LATEST_AUDIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    audit = json.load(stream)
raise SystemExit(0 if audit.get("complete") is True else 1)
PY
}

start_scheduler() {
  tmux new-session -d -s "$SCHEDULER_SESSION" \
    "cd '$REPO' && exec env NORTH_PERSONAL_LIMIT=25 NORTH_BACKUP_PERSONAL_LIMIT=8 kai0/.venv/bin/python train_scripts/kai/volc/resource_aware_scheduler.py --interval 15"
  log "restarted tmux session $SCHEDULER_SESSION"
}

start_monitor() {
  tmux new-session -d -s "$MONITOR_SESSION" \
    "cd '$REPO' && exec kai0/.venv/bin/python train_scripts/kai/volc/monitor_paper_todo_hourly.py --interval-seconds 3600"
  log "restarted tmux session $MONITOR_SESSION"
}

log "supervisor started interval=${INTERVAL_SECONDS}s"
while true; do
  if todo_complete; then
    scheduler_alive=false
    monitor_alive=false
    tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null && scheduler_alive=true
    tmux has-session -t "$MONITOR_SESSION" 2>/dev/null && monitor_alive=true
    write_status complete "$scheduler_alive" "$monitor_alive"
    log "hourly audit reports complete=true; supervisor exiting"
    exit 0
  fi

  scheduler_alive=true
  monitor_alive=true
  if ! tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
    scheduler_alive=false
    start_scheduler
    scheduler_alive=true
  fi
  if ! tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
    monitor_alive=false
    start_monitor
    monitor_alive=true
  fi
  write_status active "$scheduler_alive" "$monitor_alive"
  sleep "$INTERVAL_SECONDS"
done
