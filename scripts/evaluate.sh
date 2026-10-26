#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$#" -ne 4 ]; then
  echo 'Usage: bash scripts/evaluate.sh MODEL_DIR CHECKPOINT DATASET_JSONL REPORT_JSON' >&2
  exit 2
fi
exec bash "$DEID_ROOT/scripts/deid.sh" evaluate --model-dir "$1" --checkpoint "$2" --dataset "$3" --report "$4"
