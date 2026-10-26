#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
cd "$DEID_ROOT"
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.13
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r requirements-model.txt
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/prepare_model.py --output "$DEID_WORK/models/rbt3"
