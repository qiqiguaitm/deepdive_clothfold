#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
STAGE=$NORTH_REPO/.staging/pi05_r4_eval_north_v1/repo
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_north_triton_exec_repair.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/pi05_r4_north_triton_exec_repair.ok
REMOTE_BIN=$STAGE/runtime/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$STAGE" "$REMOTE_BIN" "$REMOTE_MARKER" <<'REMOTE'
set -Eeuo pipefail
stage=$1
bin_dir=$2
marker=$3

cat <<'HASHES' | (cd "$bin_dir" && sha256sum -c -)
ad1d0f0699f46603416eb58e7b9fdf9a293e95f9312c9df7fd6fda56a6c30d41  cuobjdump
1080bc909a3020761ff4c78a9b7bd9b4937bdb9c0978709cbe52381de612a1f2  nvdisasm
c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca  ptxas
983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461  ptxas-blackwell
HASHES
chmod 0755 "$bin_dir/cuobjdump" "$bin_dir/nvdisasm" \
  "$bin_dir/ptxas" "$bin_dir/ptxas-blackwell"
for tool in cuobjdump nvdisasm ptxas ptxas-blackwell; do
  test -x "$bin_dir/$tool"
  "$bin_dir/$tool" --version >/dev/null
done

# The stopped attempts failed during their first compiled policy query and
# produced no valid episode. Reset only their non-canonical result trees.
for arm in terminal_outcome outcome_free_crave; do
  rm -rf "$stage/lmvla/lawam/results/eval_runs/robotwin/pi05_r4_${arm}_seed1000"
  rm -f "$stage/logs/resource_markers/pi05_r4_${arm}_seed1000.ok"
done
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nstage=%s\ntriton_tools=cuobjdump,nvdisasm,ptxas,ptxas-blackwell\nmode=0755\n' \
  "$(date -u +%FT%TZ)" "$stage" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
