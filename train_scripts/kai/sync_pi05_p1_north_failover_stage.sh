#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
readonly MANIFEST_REL=lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_failover_stage_v1.json
readonly MANIFEST=$REPO/$MANIFEST_REL
readonly AUDITOR_REL=lmvla/lmwm/scripts/audit_pi05_p1_north_failover_stage.py
readonly PREFLIGHT_REL=train_scripts/kai/preflight_pi05_p1_north_failover_stage.sh
readonly REMOTE=root@124.174.16.237
readonly SSH_PORT=16370
readonly STAGE_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/pi05_p1_failover_20260804T1034Z
readonly LOCAL_REPORT=$REPO/logs/pi05_p1_failover/north_stage_report.json
readonly REMOTE_REPORT=$STAGE_ROOT/north_stage_report.json
readonly RSYNC_MAX_ATTEMPTS=${RSYNC_MAX_ATTEMPTS:-20}
readonly RSYNC_RETRY_SECONDS=${RSYNC_RETRY_SECONDS:-30}

test -s "$MANIFEST"
test "$(jq -r '.launch_authorized' "$MANIFEST")" = false
mkdir -p "$(dirname "$LOCAL_REPORT")"

SSH=(ssh -n -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10)
RSYNC_RSH="ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=10"

"${SSH[@]}" "$REMOTE" "mkdir -p '$STAGE_ROOT'"

sync_relative() {
  local source_relative=$1
  local destination_relative=${2:-$source_relative}
  local source=$REPO/$source_relative
  local destination_parent=$STAGE_ROOT/$(dirname "$destination_relative")
  "${SSH[@]}" "$REMOTE" "mkdir -p '$destination_parent'"
  local attempt
  for ((attempt = 1; attempt <= RSYNC_MAX_ATTEMPTS; attempt++)); do
    if [[ -d "$source" ]]; then
      rsync -a --partial --partial-dir=.rsync-partial --info=stats2,progress2 \
        -e "$RSYNC_RSH" "$source/" "$REMOTE:$STAGE_ROOT/$destination_relative/" \
        </dev/null && return 0
    else
      rsync -a --partial --partial-dir=.rsync-partial --info=stats2,progress2 \
        -e "$RSYNC_RSH" "$source" "$REMOTE:$destination_parent/" \
        </dev/null && return 0
    fi
    printf 'rsync attempt %d/%d failed for %s; retrying in %ds\n' \
      "$attempt" "$RSYNC_MAX_ATTEMPTS" "$source_relative" "$RSYNC_RETRY_SECONDS" >&2
    sleep "$RSYNC_RETRY_SECONDS"
  done
  printf 'rsync exhausted %d attempts for %s\n' \
    "$RSYNC_MAX_ATTEMPTS" "$source_relative" >&2
  return 1
}

while IFS=$'\t' read -r source_relative destination_relative; do
  sync_relative "$source_relative" "$destination_relative"
done < <(
  jq -r '.artifacts[] | [(.source_path // .path), .path] | @tsv' "$MANIFEST"
)

while IFS= read -r relative; do
  sync_relative "$relative"
done < <(jq -r '.control_files | keys[]' "$MANIFEST")

sync_relative "$MANIFEST_REL"
sync_relative "$AUDITOR_REL"
sync_relative "$PREFLIGHT_REL"

"${SSH[@]}" "$REMOTE" \
  "python3 '$STAGE_ROOT/$AUDITOR_REL' --manifest '$STAGE_ROOT/$MANIFEST_REL' --root '$STAGE_ROOT' --output '$REMOTE_REPORT'"
"${SSH[@]}" "$REMOTE" \
  "STAGE_ROOT='$STAGE_ROOT' STAGE_REPORT='$REMOTE_REPORT' bash '$STAGE_ROOT/$PREFLIGHT_REL'"
rsync -a -e "$RSYNC_RSH" "$REMOTE:$REMOTE_REPORT" "$LOCAL_REPORT"
rsync -a -e "$RSYNC_RSH" \
  "$REMOTE:$STAGE_ROOT/pi05_p1_north_runtime_preflight.json" \
  "$REPO/logs/pi05_p1_failover/north_runtime_preflight.json"

if [[ "$(jq -r '.stage_verified' "$LOCAL_REPORT")" != true ]]; then
  printf 'North P1 failover stage verification failed: %s\n' "$LOCAL_REPORT" >&2
  exit 1
fi
if [[ "$(jq -r '.runtime_preflight_passed' "$REPO/logs/pi05_p1_failover/north_runtime_preflight.json")" != true ]]; then
  printf 'North P1 failover runtime preflight failed\n' >&2
  exit 1
fi
printf 'North P1 failover stage verified; launch remains unauthorized: %s\n' "$LOCAL_REPORT"
