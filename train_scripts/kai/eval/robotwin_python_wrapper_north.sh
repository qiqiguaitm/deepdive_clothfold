#!/usr/bin/env bash
set -euo pipefail

CLIENT=${ROBOTWIN_NORTH_CLIENT:-/vePFS-North-E/vis_robot/huanqian/conda_envs/RoboTwin}
TIM_RT=${ROBOTWIN_NORTH_RUNTIME:-/vePFS-North-E/vis_robot/tim/RoboTwin}
CLIENT_DEPS=${ROBOTWIN_NORTH_CLIENT_DEPS:-/vePFS-North-E/vis_robot/tim/robotwin_client_deps}

test -x "$CLIENT/bin/python"
test -d "$TIM_RT/envs/curobo/src"
test -d "$TIM_RT/_shim"
test -d "$CLIENT_DEPS"

export VK_ICD_FILENAMES="$CLIENT/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export PYTHONPATH="$TIM_RT/envs/curobo/src:$TIM_RT/_shim:$CLIENT_DEPS:${ROBOTWIN_EXTRA_SITE:+$ROBOTWIN_EXTRA_SITE:}${PYTHONPATH:-}"

exec "$CLIENT/bin/python" "$@"
