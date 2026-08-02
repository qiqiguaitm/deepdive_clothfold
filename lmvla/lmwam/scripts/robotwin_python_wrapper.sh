#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../env/prepare_robotwin_renderer.sh"
if [ -n "${ROBOTWIN_EXTRA_SITE:-}" ]; then
  export PYTHONPATH="/vePFS/tim/robotwin_client_deps:$ROBOTWIN_EXTRA_SITE:${PYTHONPATH:-}"
else
  export PYTHONPATH="/vePFS/tim/robotwin_client_deps:${PYTHONPATH:-}"
fi
exec /vePFS/HuanQian/conda_envs/RoboTwin/bin/python "$@"
