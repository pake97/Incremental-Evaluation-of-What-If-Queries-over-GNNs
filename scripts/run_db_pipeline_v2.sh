#!/usr/bin/env bash
# R3.O4: store-backed pipeline with MEASURED inference for every method
# (src/eval/exp_db_pipeline_v2.py -> results/db_pipeline_v2/). Exactness gate
# first; every step logs the top 5 CPU processes (quiet-machine proof).
# Needs the Neo4j and Memgraph containers up (docker compose up -d neo4j memgraph).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
LOG=results/logs/db_pipeline_v2_$(date +%Y%m%d_%H%M%S).log
mkdir -p results/logs
top5() { echo "# $(date '+%F %T') $1 — top 5 CPU:"; ps -Ao %cpu,comm -r | head -6 || true
         echo "# loadavg: $(sysctl -n vm.loadavg)"; }
{ top5 pytest; $PY -m pytest tests/ -q -W ignore
  top5 gate; bash scripts/run_exactness.sh
  top5 experiment; caffeinate -i $PY -W ignore -m src.eval.exp_db_pipeline_v2 --config configs/db.yaml
} 2>&1 | tee "$LOG"
