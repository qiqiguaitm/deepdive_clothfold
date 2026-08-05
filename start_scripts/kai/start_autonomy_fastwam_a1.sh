#!/usr/bin/env bash
# Compatibility entry point for the canonical FastWAM A1 preset.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/fastwam/deploy/start_a1.sh" "$@"
