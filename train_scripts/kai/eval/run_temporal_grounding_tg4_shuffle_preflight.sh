#!/usr/bin/env bash
set -euo pipefail
umask 0002

readonly REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
readonly LAWAM="$REPO/lmvla/lawam"
readonly RUN_ID=temporal_grounding_tg4_full_seed1100
readonly SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
readonly SHUFFLE="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_shuffle_v1.json"
readonly EVAL_MANIFEST="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json"
readonly NORMAL_PREFLIGHT_MARKER="$REPO/logs/resource_markers/temporal_grounding_tg4_eval_local_preflight.ok"
readonly MARKER="$REPO/logs/resource_markers/temporal_grounding_tg4_shuffle_preflight.ok"
readonly ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
readonly ROOT="$REPO/logs/resource_scheduler_local/tg4_shuffle_preflight/attempt_${ATTEMPT_ID}"
readonly CAPTURE_ROOT="$ROOT/features"
readonly CAPTURE_RESULT="$ROOT/capture_result"
readonly SHUFFLE_RESULT="$ROOT/shuffle_result"
readonly CAPTURE_SCENES="$ROOT/capture_scene_seeds.json"
readonly SHUFFLE_SCENES="$ROOT/shuffle_scene_seeds.json"
readonly SOURCE_SEED_FILE="$ROOT/shuffle_source_seed.txt"
readonly CONTROL_PYTHON="${TG4_CONTROL_PYTHON:-$REPO/kai0/.venv/bin/python}"

test ! -e "$MARKER" || exit 0
test -f "$NORMAL_PREFLIGHT_MARKER"
bash "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh"
source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
"$CONTROL_PYTHON" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_evaluation.py" \
  --repo "$REPO" --manifest "$EVAL_MANIFEST"

mapfile -t run_dirs < <(find "$LAWAM/results/Checkpoints/robotwin" -maxdepth 1 \
  -type d -name "*+${RUN_ID}" -print | sort)
[[ ${#run_dirs[@]} -eq 1 ]] || {
  echo "expected exactly one accepted shuffle-preflight checkpoint, found ${#run_dirs[@]}" >&2
  exit 13
}
readonly CKPT="${run_dirs[0]}/final_model/pytorch_model.pt"
test -f "$CKPT"
test -f "$SCENES"
test -f "$SHUFFLE"
mkdir -p "$CAPTURE_ROOT" "$CAPTURE_RESULT/seed0" "$SHUFFLE_RESULT/seed0"

"$CONTROL_PYTHON" - "$SCENES" "$SHUFFLE" "$CAPTURE_SCENES" "$SHUFFLE_SCENES" "$SOURCE_SEED_FILE" <<'PY'
import json
import sys
from pathlib import Path

scenes_path, shuffle_path, capture_path, eval_path, source_path = map(Path, sys.argv[1:])
scenes = json.loads(scenes_path.read_text())
shuffle = json.loads(shuffle_path.read_text())
task = "beat_block_hammer"
target = scenes["eval_seeds"]["0"][task][0]
source = int(shuffle["mapping"][task]["0"][str(target)])
if source == target:
    raise SystemExit("shuffle preflight requires a no-self source")
available = scenes["eval_seeds"]["0"][task]
if source not in available:
    raise SystemExit("frozen shuffle source is absent from the frozen scene manifest")
capture = {
    "episodes_per_cell": 2,
    "eval_seeds": {"0": {task: [target, source]}},
}
evaluation = {
    "episodes_per_cell": 1,
    "eval_seeds": {"0": {task: [target]}},
}
capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
eval_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
source_path.write_text(f"{source}\n")
PY

export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/vePFS/tim/workspace/miniconda3_gf0/envs/lawam/bin/python}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}"
export ROBOTWIN_TASKS=beat_block_hammer
export TASK_CONFIG=demo_clean
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_REPLAN_STEPS=36
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export ROBOTWIN_TASK_SCOPED_SERVER=1
export LAWAM_FUTURE_CAPTURE_ROOT="$CAPTURE_ROOT"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export USE_BF16=1
export GPU_IDS=0
export SEED=0
unset LAWAM_FUTURE_OFF LAWAM_AUXILIARY_OFF LAWAM_CONDITIONING_OFF
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE

cd "$LAWAM"
export LAWAM_FUTURE_INTERVENTION=normal
unset LAWAM_FUTURE_SHUFFLE_MANIFEST
export ROBOTWIN_TEST_NUM=2
export ROBOTWIN_EPISODE_SEED_MANIFEST="$CAPTURE_SCENES"
export PORT_BASE=28700
export ROBOTWIN_CKPT_ALIAS=tg4_full_s1100_shuffle_preflight_capture
export ROBOTWIN_EVAL_ROOT="$CAPTURE_RESULT/seed0"
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  "$CKPT" "$TASK_CONFIG" tg4-full-s1100-shuffle-preflight-capture \
  >"$ROOT/capture.log" 2>&1

readonly SOURCE_SEED="$(cat "$SOURCE_SEED_FILE")"
[[ "$SOURCE_SEED" =~ ^[0-9]+$ ]]
readonly SOURCE_DIR="$CAPTURE_ROOT/beat_block_hammer/eval_seed_0/scene_seed_${SOURCE_SEED}"
find "$SOURCE_DIR" -type f -name 'query_*.npy' -print -quit | grep -q . || {
  echo "shuffle preflight capture produced no source-scene features" >&2
  exit 14
}

export LAWAM_FUTURE_INTERVENTION=shuffled
export LAWAM_FUTURE_SHUFFLE_MANIFEST="$SHUFFLE"
export ROBOTWIN_TEST_NUM=1
export ROBOTWIN_EPISODE_SEED_MANIFEST="$SHUFFLE_SCENES"
export PORT_BASE=28800
export ROBOTWIN_CKPT_ALIAS=tg4_full_s1100_shuffle_preflight
export ROBOTWIN_EVAL_ROOT="$SHUFFLE_RESULT/seed0"
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  "$CKPT" "$TASK_CONFIG" tg4-full-s1100-shuffle-preflight \
  >"$ROOT/shuffle.log" 2>&1

mapfile -t summaries < <(find "$SHUFFLE_RESULT" -type f -name summary.json -print | sort)
[[ ${#summaries[@]} -eq 1 ]] || {
  echo "expected one shuffled preflight summary, found ${#summaries[@]}" >&2
  exit 15
}

"$CONTROL_PYTHON" - "$CKPT" "${summaries[0]}" "$MARKER" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

checkpoint, summary_path, marker_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text())
if summary.get("task_name") != "beat_block_hammer":
    raise SystemExit("unexpected shuffled preflight task")
if summary.get("n_episodes") != 1 or len(summary.get("episodes", [])) != 1:
    raise SystemExit("shuffled preflight did not execute exactly one episode")
if int(summary.get("model_queries", 0)) <= 0:
    raise SystemExit("shuffled preflight policy server handled no model queries")
if int(summary.get("obs_fetch_count", 0)) <= 0:
    raise SystemExit("shuffled preflight simulator produced no observations")
payload = {
    "complete": True,
    "protocol": "temporal_grounding_tg4_shuffle_eval_preflight_v1",
    "claim_bearing": False,
    "checkpoint": str(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "summary": str(summary_path),
    "task": summary["task_name"],
    "episodes": summary["n_episodes"],
    "model_queries": summary["model_queries"],
    "obs_fetch_count": summary["obs_fetch_count"],
    "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_suffix(marker_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, marker_path)
PY

echo "TG4 shuffled evaluator preflight complete marker=$MARKER"
