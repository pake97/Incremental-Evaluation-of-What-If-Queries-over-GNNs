#!/usr/bin/env bash
# Experiment 2 — single-query efficiency (incremental vs from-scratch latency).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
done
$PY -m src.eval.exp_single_query --config configs/single_query.yaml
