#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
export PYTHONPATH="/vePFS/tim/robotwin_client_deps:$SCRIPT_DIR:${PYTHONPATH:-}"

script=${1:?RoboTwin Python script is required}
shift
exec /vePFS/HuanQian/conda_envs/RoboTwin/bin/python -c '
import runpy
import sys
from hook import install

install()
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
' "$script" "$@"
