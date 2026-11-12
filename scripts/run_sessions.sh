#!/usr/bin/env bash
# R2.O2 session experiment (REVISION_PLAN.md §7): 500-state exploration DAGs, ours
# (impl v2, reuse, retention all/16/1) vs from-scratch, in-memory and Neo4j-backed.
# Resumable: settings already complete in results/sessions/states.csv are skipped.
# Quiet machine only: check `top`, kill BTLEServer / neo4r first.
# The store part needs Neo4j up (docker compose up -d neo4j); otherwise it is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in eicu finbench snb faers; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --tasks nodelabel --layers 2 --train
done
$PY -m src.eval.exp_sessions --config configs/sessions.yaml \
    --parts generate ours scratch store figures "$@"
