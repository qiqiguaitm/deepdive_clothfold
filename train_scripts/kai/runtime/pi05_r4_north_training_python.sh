#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -n ${PI05_R4_MOUNT_ROOT:-} ]]; then
  mount_root=$PI05_R4_MOUNT_ROOT
elif [[ -d /vePFS/tim/workspace/deepdive_kai0 ]]; then
  mount_root=/vePFS
else
  mount_root=/vePFS-North-E/vis_robot
fi

repo=$mount_root/tim/workspace/deepdive_kai0
eval_runtime=$mount_root/workspace/deepdive_kai0/.staging/pi05_r4_eval_north_v1/repo/runtime
python=$eval_runtime/python/bin/python3.12
site_packages=$eval_runtime/venv/lib/python3.12/site-packages
overlay=$repo/runtime/pi05_r4_north_training/site-packages
lerobot_src=$mount_root/tim/workspace/lerobot-main/src

test -x "$python"
test -d "$site_packages"
test -d "$overlay/accelerate"
test -f "$lerobot_src/lerobot/__init__.py"

export PYTHONPATH="$overlay:$lerobot_src:$site_packages${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$@"
