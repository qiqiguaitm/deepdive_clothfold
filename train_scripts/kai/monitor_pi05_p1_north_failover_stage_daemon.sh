#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
readonly MONITOR=${REPO}/train_scripts/kai/monitor_pi05_p1_north_failover_stage.sh
readonly PID_FILE=${REPO}/logs/pi05_p1_failover/progress_monitor.pid
readonly LOCK_FILE=/tmp/pi05_p1_north_failover_progress.lock
readonly INTERVAL_SECONDS=300

exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  printf '[%s] another North failover monitor is already running\n' "$(date -u +'%FT%TZ')" >&2
  exit 1
fi

printf '%s\n' "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

while true; do
  if ! "${MONITOR}"; then
    printf '[%s] North failover progress probe failed\n' "$(date -u +'%FT%TZ')" >&2
  fi
  sleep "${INTERVAL_SECONDS}"
done

