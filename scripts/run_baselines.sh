#!/usr/bin/env bash
# Baseline comparison: ours vs Ripple vs NeutronRT vs from-scratch
# (forward counterfactual query; op-counts + latency + exactness).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_baselines --config configs/baselines.yaml
