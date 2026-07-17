#!/usr/bin/env bash
# Scaling experiment — intervention size as a FRACTION of the dataset (5..30%),
# sweeping arch x kind x frac x layer, incremental (best-impl) vs from-scratch.
# Feeds the scaling_L2_*, scaling_layers_*, scaling_signflips_* figures.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_scaling --config configs/scaling.yaml
