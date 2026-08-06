#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
MARKER=$REPO/logs/resource_markers/pi05_r2_north_sync.ok
EVIDENCE=$REPO/logs/sync/pi05_r2_north

roots=(
  lmvla/crave/src/crave/config
  lmvla/crave/src/crave/encoders
  lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi.py
  lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi_r2.py
  lmvla/lawam/examples/Robotwin/eval_files/robotwin_batch_bridge.py
  lmvla/lawam/examples/Robotwin/eval_files/batched_eval_runner.py
  lmvla/lmwm/data/pi05_crave_r0_v1
  lmvla/lmwm/data/pi05_r2_causal_readout_v1
  lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json
  lmvla/lmwm/scripts/pi05_r2_adaptive_execution.py
  lmvla/lmwm/scripts/analyze_pi05_r2_adaptive_screen.py
  lmvla/lmwm/scripts/summarize_robotwin_eval.py
  lmvla/lmwm/scripts/summarize_pi05_r2_eval.py
  lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
  lmvla/lmwm/scripts/verify_pi05_r2_protocol.py
  lmvla/paper_iclr_lmvla/manifests/pi05_r2_adaptive_execution_protocol_v1.json
  train_scripts/kai/eval/pi05_r2_python_wrapper.sh
  train_scripts/kai/eval/run_pi05_r2_batched_eval_runner.py
  train_scripts/kai/eval/run_pi05_r2_adaptive_screen.sh
  train_scripts/kai/eval/serve_lerobot_pi05.py
  train_scripts/kai/volc/pi05_r2_fixed4_east_4h20.yaml
  train_scripts/kai/volc/pi05_r2_fixed4_north_4h20.yaml
  train_scripts/kai/volc/pi05_r2_fixed4_cnsh_4a100.yaml
  train_scripts/kai/volc/pi05_r2_adaptive_east_4h20.yaml
  train_scripts/kai/volc/pi05_r2_adaptive_north_4h20.yaml
  train_scripts/kai/volc/pi05_r2_adaptive_cnsh_4a100.yaml
)

for path in "${roots[@]}"; do test -e "$REPO/$path"; done
mkdir -p "$EVIDENCE" "$(dirname "$MARKER")"
paths=$EVIDENCE/paths.txt
local_manifest=$EVIDENCE/local.sha256
remote_manifest=$EVIDENCE/north.sha256
(
  cd "$REPO"
  find "${roots[@]}" -type f -print | LC_ALL=C sort >"$paths"
  xargs sha256sum <"$paths" >"$local_manifest"
  tar -cf - -T "$paths"
) | ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p '$NORTH_REPO' && tar -C '$NORTH_REPO' -xf -"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "cd '$NORTH_REPO' && xargs sha256sum" <"$paths" >"$remote_manifest"
cmp "$local_manifest" "$remote_manifest"
temporary=$MARKER.tmp.$$
printf 'validated=%s\nremote=%s\nfiles=%s\n' \
  "$(date -u +%FT%TZ)" "$NORTH_REPO" "$(wc -l <"$paths")" >"$temporary"
cat "$local_manifest" >>"$temporary"
mv "$temporary" "$MARKER"
