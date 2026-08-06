#!/usr/bin/env bash
set -Eeuo pipefail

readonly STAGE_ROOT=${STAGE_ROOT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/pi05_p1_failover_20260804T1034Z}
readonly RUNTIME_REPO=${RUNTIME_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
readonly MANIFEST=$STAGE_ROOT/lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_failover_stage_v1.json
readonly STAGE_REPORT=$STAGE_ROOT/north_stage_report.json
readonly RUNTIME_PREFLIGHT=$STAGE_ROOT/pi05_p1_north_runtime_preflight.json
readonly AUTHORIZATION=${AUTHORIZATION:-$STAGE_ROOT/pi05_p1_north_failover_authorization.json}
readonly AUTH_AUDIT=$STAGE_ROOT/logs/pi05_p1_failover/authorization_audit.json
readonly AUTH_AUDITOR=$STAGE_ROOT/lmvla/lmwm/scripts/audit_pi05_p1_failover_authorization.py
readonly PYTHON_BIN=$RUNTIME_REPO/kai0/.venv/bin/python
readonly LOG_DIR=$STAGE_ROOT/logs/pi05_p1_failover/pair_train
readonly REPORT=$STAGE_ROOT/pi05_p1_north_pair_training_report.json
readonly LOCK_FILE=/tmp/pi05_p1_north_failover_pair.lock

test "$(jq -r '.stage_verified' "$STAGE_REPORT")" = true
test "$(jq -r '.runtime_preflight_passed' "$RUNTIME_PREFLIGHT")" = true
test "$(jq -r '.launch_authorized' "$MANIFEST")" = false
test -s "$AUTHORIZATION"
test -x "$PYTHON_BIN"
mkdir -p "$LOG_DIR" "$(dirname "$AUTH_AUDIT")"

"$PYTHON_BIN" "$AUTH_AUDITOR" \
  --manifest "$MANIFEST" \
  --authorization "$AUTHORIZATION" \
  --output "$AUTH_AUDIT" >/dev/null
test "$(jq -r '.launch_authorized' "$AUTH_AUDIT")" = true

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'another P1 North failover pair is already running\n' >&2
  exit 1
fi

final_checkpoint() {
  case "$1" in
    a0)
      printf '%s\n' "$STAGE_ROOT/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999"
      ;;
    candidate)
      printf '%s\n' "$STAGE_ROOT/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000/49999"
      ;;
  esac
}

run_arm() {
  local arm=$1
  local devices=$2
  local final
  final=$(final_checkpoint "$arm")
  if [[ -s "$final/_CHECKPOINT_METADATA" && -s "$final/params/_METADATA" ]]; then
    printf '[%s] %s already complete\n' "$(date -u +'%FT%TZ')" "$arm"
    return 0
  fi
  printf '[%s] starting %s on CUDA_VISIBLE_DEVICES=%s\n' \
    "$(date -u +'%FT%TZ')" "$arm" "$devices"
  env CUDA_VISIBLE_DEVICES="$devices" \
    REPO="$STAGE_ROOT" \
    DATA_REPO="$STAGE_ROOT/datasets/robotwin2.0_official_prompts_v21" \
    PYTHON_BIN="$PYTHON_BIN" \
    ARM="$arm" SEED=1000 WORKERS=8 \
    bash "$STAGE_ROOT/train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh" \
    >"$LOG_DIR/${arm}.log" 2>&1
}

run_arm a0 0,1,2,3 &
a0_pid=$!
run_arm candidate 4,5,6,7 &
candidate_pid=$!
trap 'kill "$a0_pid" "$candidate_pid" 2>/dev/null || true' INT TERM

set +e
wait "$a0_pid"
a0_status=$?
wait "$candidate_pid"
candidate_status=$?
set -e
trap - INT TERM

a0_final=$(final_checkpoint a0)
candidate_final=$(final_checkpoint candidate)
a0_complete=false
candidate_complete=false
[[ -s "$a0_final/_CHECKPOINT_METADATA" && -s "$a0_final/params/_METADATA" ]] && a0_complete=true
[[ -s "$candidate_final/_CHECKPOINT_METADATA" && -s "$candidate_final/params/_METADATA" ]] && candidate_complete=true

tmp=${REPORT}.tmp.$$
jq -n \
  --arg timestamp "$(date -u +'%FT%TZ')" \
  --argjson a0_exit "$a0_status" \
  --argjson candidate_exit "$candidate_status" \
  --argjson a0_complete "$a0_complete" \
  --argjson candidate_complete "$candidate_complete" \
  '{
    schema_version: 1,
    timestamp: $timestamp,
    protocol: "pi05_p1_north_failover_pair_v1",
    authorization_audit_passed: true,
    a0: {exit_code: $a0_exit, checkpoint_49999_complete: $a0_complete},
    candidate: {exit_code: $candidate_exit, checkpoint_49999_complete: $candidate_complete},
    pair_training_complete: ($a0_exit == 0 and $candidate_exit == 0 and $a0_complete and $candidate_complete)
  }' > "$tmp"
mv "$tmp" "$REPORT"
test "$(jq -r '.pair_training_complete' "$REPORT")" = true
