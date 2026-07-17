#!/usr/bin/env bash
# Interactive what-if SESSION case study on eICU (RGCN + hGAT), Neo4j-backed.
# A composing DAG (branch + backtrack): shows ours stays exact on ground truth
# while from-scratch/Ripple/NeutronRT re-fetch the graph every state and Ripple
# drifts. Requires the DB stack up + eICU loaded + configs/db.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
$PY -m src.eval.ensure_ckpts --dataset eicu --train
$PY -m src.eval.exp_interactive_eicu
