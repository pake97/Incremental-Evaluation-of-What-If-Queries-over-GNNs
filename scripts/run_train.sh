#!/usr/bin/env bash
# Train (only) the missing frozen checkpoints for every dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
echo "PASS train: all checkpoints present"
