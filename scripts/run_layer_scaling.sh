#!/usr/bin/env bash
# Experiment 3 — layer scaling (L = 1..4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_layer_scaling --config configs/layer_scaling.yaml
