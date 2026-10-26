#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
cd "$DEID_ROOT"
exec cargo "$@"
