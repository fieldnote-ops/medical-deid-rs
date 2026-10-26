#!/usr/bin/env bash
set -euo pipefail
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
cd "$DEID_ROOT"
DEID_RUN="${1:-$DEID_WORK/runs/demo-v0.1}"
python3 scripts/make_synthetic.py --output "$DEID_WORK/runs/demo-data"
bash scripts/cargo.sh build --release --locked
bash scripts/deid.sh audit --dataset "$DEID_WORK/runs/demo-data"
bash scripts/deid.sh train --model-dir "$DEID_WORK/models/rbt3" \
  --dataset "$DEID_WORK/runs/demo-data" --run "$DEID_RUN" --epochs 3
bash scripts/deid.sh evaluate --model-dir "$DEID_WORK/models/rbt3" \
  --checkpoint "$DEID_RUN/best.bpk" --dataset "$DEID_WORK/runs/demo-data/test.jsonl" \
  --report "$DEID_RUN/test-report.json"
