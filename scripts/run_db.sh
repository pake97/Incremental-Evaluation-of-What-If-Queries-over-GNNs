#!/usr/bin/env bash
# Store-backed what-if round-trip on Neo4j / Memgraph / TigerGraph.
# Requires the DBs up (docker compose up -d) and loaded (./db/load_all.sh ...)
# and configs/db.yaml present. Skipped gracefully if no DB is enabled.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f configs/db.yaml ]]; then
  echo "SKIP db: configs/db.yaml not present (see db/README.md)"; exit 0
fi
venv/bin/python -m src.eval.exp_db_roundtrip --config configs/db.yaml
