#!/usr/bin/env bash
# Full single-query grid: kind x size x depth L x arch, per dataset.
# Feeds the comprehensive latency / operations / sign-flip figures.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_grid --config configs/grid.yaml
