#!/usr/bin/env bash
# Experiment 1 — exactness gate. Must pass before any timing experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_exactness --config configs/exactness.yaml
