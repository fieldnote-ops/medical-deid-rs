#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
exec python3 "$DEID_ROOT/scripts/model_bundle.py" "$@"
