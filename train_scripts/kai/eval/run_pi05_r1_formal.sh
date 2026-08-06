#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
CONDITION=${R1_CONDITION:?set R1_CONDITION}
SEED=${SEED:-1000}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json
VERIFY_REPO=${R1_VERIFY_REPO:-$REPO}
PROTOCOL_OUTPUT_DIR=${R1_PROTOCOL_OUTPUT_DIR:-$REPO/logs/r1_runtime}

case "$CONDITION" in
  a0)
    CONFIG=pi05_robotwin_a0_public_exact_bj
    PREDICTIVE_INTERVENTION=normal
    RECURRENCE_INTERVENTION=normal
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed${SEED}/49999
    ;;
  predictive)
    CONFIG=pi05_predictive_adapter_p1_eval
    PREDICTIVE_INTERVENTION=normal
    RECURRENCE_INTERVENTION=normal
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed${SEED}/49999
    ;;
  crave)
    CONFIG=pi05_r1_crave_eval
    PREDICTIVE_INTERVENTION=normal
    RECURRENCE_INTERVENTION=normal
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_r1_crave/pi05_r1_crave_seed${SEED}/49999
    ;;
  combined|combined_zero_gate|combined_shuffled)
    CONFIG=pi05_r1_combined_eval
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_r1_combined/pi05_r1_combined_seed${SEED}/49999
    case "$CONDITION" in
      combined) PREDICTIVE_INTERVENTION=normal; RECURRENCE_INTERVENTION=normal ;;
      combined_zero_gate) PREDICTIVE_INTERVENTION=zero_gate; RECURRENCE_INTERVENTION=zero_gate ;;
      combined_shuffled) PREDICTIVE_INTERVENTION=shuffled; RECURRENCE_INTERVENTION=shuffled ;;
    esac
    ;;
  *) echo "unsupported R1 condition: $CONDITION" >&2; exit 2 ;;
esac

CKPT=${CKPT:-$DEFAULT_CKPT}
RESULT_NAME=${RESULT_NAME:-pi05_r1_seed${SEED}_${CONDITION}}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json

for required in \
  "$REPO/logs/predictive/p0_eval/p0_gate.accepted" \
  "$REPO/logs/crave_r0/probe_gate/r0_gate.accepted" \
  "$CKPT/params/_METADATA" \
  "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json" \
  "$MANIFEST" "$PROTOCOL"; do
  test -s "$required"
done
if [[ "$SEED" != 1000 ]]; then
  test -s "$REPO/logs/r1/seed1000/r1_gate.accepted"
fi
if [[ "$VERIFY_REPO" != "$REPO" ]]; then
  test -s "$VERIFY_REPO/READY"
fi
mkdir -p "$PROTOCOL_OUTPUT_DIR"
python3 "$REPO/lmvla/lmwm/scripts/verify_pi05_r1_protocol.py" \
  --repo "$VERIFY_REPO" --protocol "$PROTOCOL" \
  --output "$PROTOCOL_OUTPUT_DIR/protocol_eval_${CONDITION}_s${SEED}.json"

env \
  PI05_EVAL_CONFIG_NAME="$CONFIG" \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  PREDICTIVE_ACTION_INTERVENTION="$PREDICTIVE_INTERVENTION" \
  RECURRENCE_ACTION_INTERVENTION="$RECURRENCE_INTERVENTION" \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS="0 1 2 3" \
  LOCAL_GPU_COUNT=${LOCAL_GPU_COUNT:-4} \
  GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0} \
  MAX_PARALLEL_SEEDS=${MAX_PARALLEL_SEEDS:-4} \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-22800} \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$RESULT_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncondition=%s\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CONDITION" "$CKPT" "$REPORT" >"$MARKER"
