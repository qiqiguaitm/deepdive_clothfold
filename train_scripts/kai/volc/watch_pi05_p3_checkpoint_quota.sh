#!/usr/bin/env bash
set -euo pipefail

HOST=${P3_NORTH_HOST:-root@124.174.16.237}
PORT=${P3_NORTH_PORT:-16370}
ROOT=${P3_NORTH_CHECKPOINT_ROOT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/pi05_p1_failover_20260804T1034Z/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact}
INTERVAL=${P3_QUOTA_POLL_SECONDS:-60}
LOG=${P3_QUOTA_LOG:-/vePFS/tim/workspace/deepdive_kai0/logs/predictive/p3_quota_guard.log}

mkdir -p "$(dirname "$LOG")"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

prune_seed() {
  local seed=$1
  ssh -p "$PORT" -o BatchMode=yes "$HOST" bash -s -- "$ROOT" "$seed" <<'REMOTE'
set -euo pipefail
root=$1/pi05_predictive_adapter_p1_a0_seed$2
test -d "$root" || exit 0

latest=-1
for path in "$root"/[0-9]*; do
  [[ -d "$path" ]] || continue
  step=${path##*/}
  [[ "$step" =~ ^[0-9]+$ ]] || continue
  [[ -f "$path/params/_METADATA" && -f "$path/train_state/_METADATA" ]] || continue
  (( step > latest )) && latest=$step
done

(( latest >= 0 )) || exit 0
for path in "$root"/[0-9]*; do
  [[ -d "$path" ]] || continue
  step=${path##*/}
  [[ "$step" =~ ^[0-9]+$ ]] || continue
  if (( step < latest )); then
    printf 'delete seed=%s step=%s latest=%s path=%s\n' "$2" "$step" "$latest" "$path"
    rm -rf -- "$path"
  fi
done
printf 'retain seed=%s step=%s\n' "$2" "$latest"
REMOTE
}

log "start host=$HOST root=$ROOT interval=${INTERVAL}s"
while true; do
  complete=0
  for seed in 1001 1002; do
    while IFS= read -r line; do
      [[ -n "$line" ]] && log "$line"
    done < <(prune_seed "$seed")

    if ssh -p "$PORT" -o BatchMode=yes "$HOST" \
      "test -f '$ROOT/pi05_predictive_adapter_p1_a0_seed${seed}/49999/params/_METADATA' && test -f '$ROOT/pi05_predictive_adapter_p1_a0_seed${seed}/49999/train_state/_METADATA'"; then
      complete=$((complete + 1))
    fi
  done
  if (( complete == 2 )); then
    log "both final checkpoints complete; stop"
    exit 0
  fi
  sleep "$INTERVAL"
done
