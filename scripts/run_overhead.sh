#!/usr/bin/env bash
# Overhead analysis — proxy (edges/nodes) vs wall-clock, and the framework tax
# between them (see results/README.md "Two currencies"). Explains why wall-clock
# understates the algorithmic speedup; the proxy is the primary efficiency metric.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_overhead --config configs/overhead.yaml
