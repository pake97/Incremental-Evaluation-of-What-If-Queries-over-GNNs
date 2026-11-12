#!/usr/bin/env bash
# SIGMOD revision data (R1.O3, R1.Q2, R2.O1, R3.O1, R3.O4, Fig. 3/4): cache size,
# per-query memory, one-time build cost, MEASURED latency + closure, break-even.
# Resumable: each part skips settings already in results/revision/*.csv.
# Store-backed build (Part C.2) needs the Bolt DBs up (docker compose up -d
# neo4j memgraph); without them it is marked skipped and everything else runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --tasks nodelabel --train
done
# D (timing) runs before B (parallel subprocesses) so nothing competes for cores
$PY -m src.eval.exp_revision --config configs/revision.yaml --parts A C D B E report
$PY -m src.eval.figures --revision
