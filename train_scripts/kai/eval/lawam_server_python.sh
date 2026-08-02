#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
hf_overlay="$repo/.runtime/lawam_hf_5_2"
python="$repo/kai0/.venv/bin/python"

test -x "$python"
test -d "$hf_overlay/transformers"
export PYTHONPATH="$hf_overlay${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$@"
