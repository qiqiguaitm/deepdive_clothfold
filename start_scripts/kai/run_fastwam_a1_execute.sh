#!/usr/bin/env bash
# Compatibility entry point for fastwam/deploy/run_a1_execute.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/fastwam/deploy/run_a1_execute.sh" "$@"
