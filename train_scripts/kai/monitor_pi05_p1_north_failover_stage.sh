#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
readonly MANIFEST=${REPO}/lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_failover_stage_v1.json
readonly SYNC_PID_FILE=${REPO}/logs/pi05_p1_failover/north_sync.pid
readonly STAGE_REPORT=${REPO}/logs/pi05_p1_failover/north_stage_report.json
readonly OUTPUT=${REPO}/logs/pi05_p1_failover/progress.json
readonly REMOTE=root@124.174.16.237
readonly SSH_PORT=16370

test -s "${MANIFEST}"
mkdir -p "$(dirname "${OUTPUT}")"

expected_bytes=$(jq '[.artifacts[].bytes] | add' "${MANIFEST}")
stage_root=$(jq -r '.stage_root' "${MANIFEST}")
launch_authorized=$(jq -r '.launch_authorized' "${MANIFEST}")
now_timestamp=$(date -u +'%FT%TZ')
now_epoch=$(date -u +%s)

previous_bytes=0
previous_epoch=0
previous_rate=0
if [[ -s "${OUTPUT}" ]]; then
  previous_bytes=$(jq -r '.staged_bytes // 0' "${OUTPUT}")
  previous_timestamp=$(jq -r '.timestamp // empty' "${OUTPUT}")
  previous_rate=$(jq -r '.rate_bytes_per_second // 0' "${OUTPUT}")
  if [[ -n "${previous_timestamp}" ]]; then
    previous_epoch=$(date -u -d "${previous_timestamp}" +%s 2>/dev/null || printf '0')
  fi
fi

sync_pid=""
sync_alive=false
if [[ -s "${SYNC_PID_FILE}" ]]; then
  sync_pid=$(tr -cd '0-9' < "${SYNC_PID_FILE}")
  if [[ -n "${sync_pid}" ]] && kill -0 "${sync_pid}" 2>/dev/null; then
    sync_alive=true
  fi
fi

stage_verified=false
if [[ -s "${STAGE_REPORT}" ]]; then
  stage_verified=$(jq -r '.stage_verified // false' "${STAGE_REPORT}")
fi

probe_error=""
staged_bytes=""
if ! staged_bytes=$(ssh -p "${SSH_PORT}" -o BatchMode=yes \
  -o ConnectTimeout=10 -o ConnectionAttempts=1 "${REMOTE}" \
  "du -sb '$stage_root' 2>/dev/null | cut -f1" 2>&1); then
  probe_error=${staged_bytes}
  staged_bytes=""
fi
if [[ ! "${staged_bytes}" =~ ^[0-9]+$ ]]; then
  if [[ -z "${probe_error}" ]]; then
    probe_error="remote stage size is unavailable"
  fi
  staged_bytes=0
fi

status=STOPPED
if [[ "${stage_verified}" == true ]]; then
  status=VERIFIED
elif [[ "${sync_alive}" == true ]]; then
  status=SYNCING
elif [[ -n "${probe_error}" ]]; then
  status=PROBE_ERROR
fi

rate_bytes_per_second=${previous_rate}
elapsed_seconds=$((now_epoch - previous_epoch))
if (( elapsed_seconds >= 60 && staged_bytes >= previous_bytes )); then
  rate_bytes_per_second=$(( (staged_bytes - previous_bytes) / elapsed_seconds ))
fi
eta_seconds=null
if [[ "${stage_verified}" == true ]]; then
  eta_seconds=0
elif (( rate_bytes_per_second > 0 && expected_bytes > staged_bytes )); then
  eta_seconds=$(( (expected_bytes - staged_bytes) / rate_bytes_per_second ))
fi

tmp=${OUTPUT}.tmp.$$
jq -n \
  --arg timestamp "${now_timestamp}" \
  --arg status "${status}" \
  --arg stage_root "${stage_root}" \
  --arg sync_pid "${sync_pid}" \
  --argjson sync_alive "${sync_alive}" \
  --argjson stage_verified "${stage_verified}" \
  --argjson launch_authorized "${launch_authorized}" \
  --argjson staged_bytes "${staged_bytes}" \
  --argjson expected_bytes "${expected_bytes}" \
  --argjson rate_bytes_per_second "${rate_bytes_per_second}" \
  --argjson eta_seconds "${eta_seconds}" \
  --arg probe_error "${probe_error}" \
  '{
    timestamp: $timestamp,
    status: $status,
    stage_root: $stage_root,
    sync_pid: (if $sync_pid == "" then null else ($sync_pid | tonumber) end),
    sync_alive: $sync_alive,
    stage_verified: $stage_verified,
    launch_authorized: $launch_authorized,
    staged_bytes: $staged_bytes,
    expected_bytes: $expected_bytes,
    progress_fraction: (
      if $expected_bytes <= 0 then 0
      elif $staged_bytes >= $expected_bytes then 1
      else ($staged_bytes / $expected_bytes)
      end
    ),
    rate_bytes_per_second: $rate_bytes_per_second,
    eta_seconds: $eta_seconds,
    probe_error: (if $probe_error == "" then null else $probe_error end)
  }' > "${tmp}"
mv "${tmp}" "${OUTPUT}"
