#!/usr/bin/env bash
set -euo pipefail

: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
: "${TG4_READY_OUTPUT:?TG4_READY_OUTPUT is required}"

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
SCHEDULER_STATE=${TG4_SCHEDULER_STATE:-$REPO/logs/resource_scheduler_state.json}
POLL_SECONDS=${TG4_RECOVERY_POLL_SECONDS:-300}
TIMEOUT_SECONDS=${TG4_RECOVERY_TIMEOUT_SECONDS:-86400}
RUN_ID=temporal_grounding_tg4_${TG4_ARM}_seed${TG4_TRAIN_SEED}
TASK_ID=${RUN_ID}_train
CHECKPOINT_ROOT=$REPO/lmvla/lawam/results/Checkpoints/robotwin
LOG_ROOT=$REPO/logs/temporal_grounding/tg4/entrypoint
EXPECTED_ERROR="line 118: el.future_action_window_size=49: command not found"

platform_completion_job() {
  python3 - "$SCHEDULER_STATE" "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

state_path, task_id = Path(sys.argv[1]), sys.argv[2]
try:
    task = json.loads(state_path.read_text())["tasks"][task_id]
    attempt = task["attempts"][-1]
except (OSError, KeyError, IndexError, TypeError, ValueError):
    raise SystemExit(1)
if (
    task.get("status") == "completed"
    and attempt.get("kind") == "platform"
    and attempt.get("resource") == "Robot-East-H20"
    and attempt.get("last_state") == "Completed"
    and attempt.get("job_id")
):
    print(attempt["job_id"])
else:
    raise SystemExit(1)
PY
}

checkpoint_complete() {
  local run=$1
  python3 - "$run/checkpoints/steps_20000_state/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    steps = json.loads(path.read_text())["steps"]
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if steps == 20000 else 1)
PY
}

deadline=$((SECONDS + TIMEOUT_SECONDS))
count=0
logs=()
terminal_ready=0
terminal_mode=
platform_job_id=
while :; do
  mapfile -t runs < <(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d \
    -name "*+$RUN_ID" | sort)
  count=${#runs[@]}
  mapfile -t logs < <(find "$LOG_ROOT" -maxdepth 1 -type f \
    -name "${TG4_ARM}_s${TG4_TRAIN_SEED}_east_*.log" | sort)
  if [[ $count == 1 && ${#logs[@]} == 1 ]] \
      && [[ -s "${runs[0]}/final_model/pytorch_model.pt" ]] \
      && [[ -s "${runs[0]}/checkpoints/steps_20000_state/optimizer.bin" ]] \
      && checkpoint_complete "${runs[0]}" \
      && grep -Fq "$RUN_ID: 100%" "${logs[0]}" \
      && grep -Fq "and that's all" "${logs[0]}"; then
    error_count=$(grep -Fc "$EXPECTED_ERROR" "${logs[0]}" || true)
    if [[ $error_count == 1 ]]; then
      terminal_mode=validated_post_training_shell_error
      terminal_ready=1
      break
    fi
    if [[ $error_count == 0 ]] && platform_job_id=$(platform_completion_job); then
      terminal_mode=clean_platform_completion
      terminal_ready=1
      break
    fi
  fi
  if (( SECONDS >= deadline )); then
    break
  fi
  sleep "$POLL_SECONDS"
done
if (( terminal_ready != 1 )); then
  echo "timed out waiting for exact complete terminal evidence for $RUN_ID" >&2
  exit 1
fi

mkdir -p "$(dirname "$TG4_READY_OUTPUT")"
temporary=$TG4_READY_OUTPUT.tmp.$$
printf 'ready=%s\nrun_id=%s\nentrypoint_log=%s\n' \
  "$(date -u +%FT%TZ)" "$RUN_ID" "${logs[0]}" >"$temporary"
printf 'terminal_mode=%s\nplatform_job_id=%s\n' \
  "$terminal_mode" "$platform_job_id" >>"$temporary"
chmod 0664 "$temporary"
mv "$temporary" "$TG4_READY_OUTPUT"
