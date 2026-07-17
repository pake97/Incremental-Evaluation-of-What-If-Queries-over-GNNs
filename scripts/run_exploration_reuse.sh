#!/usr/bin/env bash
# Experiment 4 — exploration reuse (the DAG experiment, three rungs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

$PY -m src.eval.ensure_ckpts --dataset finbench --train
$PY -m src.eval.exp_exploration --config configs/exploration_reuse.yaml
