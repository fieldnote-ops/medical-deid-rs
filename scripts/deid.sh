#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
cd "$DEID_ROOT"
DEID_BIN="$CARGO_TARGET_DIR/release/medical-deid"
if [ -x "$DEID_ROOT/bin/medical-deid" ]; then
  DEID_BIN="$DEID_ROOT/bin/medical-deid"
fi
if [ ! -x "$DEID_BIN" ]; then
  echo "Build first: bash scripts/cargo.sh build --release" >&2
  exit 1
fi
exec "$DEID_BIN" "$@"
