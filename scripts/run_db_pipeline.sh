#!/usr/bin/env bash
# Store-backed per-step PIPELINE timing (download+build / update / GNN) on
# Neo4j + Memgraph. Requires the DB stack up + loaded and configs/db.yaml.
# Feeds db_pipeline_steps_*, db_finbench_2x2, db_full_appendix figures.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f configs/db.yaml ]]; then
  echo "SKIP db_pipeline: configs/db.yaml not present (see db/README.md)"; exit 0
fi
venv/bin/python -m src.eval.exp_db_pipeline --config configs/db.yaml
