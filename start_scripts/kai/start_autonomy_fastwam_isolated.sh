#!/usr/bin/env bash
# Compatibility entry point. Canonical implementation: fastwam/deploy/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/fastwam/deploy/start_autonomy_isolated.sh" "$@"
