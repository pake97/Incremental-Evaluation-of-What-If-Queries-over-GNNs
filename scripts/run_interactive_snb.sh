#!/usr/bin/env bash
# Interactive what-if SESSION case study on SNB (RGCN + hGAT), Neo4j-backed.
# Social-network query (predicted home-country fraction over a cohort); same
# composing DAG + backtrack as the eICU study. Requires the DB stack up +
# SNB loaded + configs/db.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
$PY -m src.eval.ensure_ckpts --dataset snb --train
$PY -m src.eval.exp_interactive_snb
