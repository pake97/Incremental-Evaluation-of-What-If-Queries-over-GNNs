#!/usr/bin/env bash
# Load one or more exported datasets into whichever DBs are running.
# Usage:  ./db/load_all.sh finbench eicu
# Reads connection settings from environment overrides or the compose defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASS=${NEO4J_PASS:-rcfpassword}
MEMGRAPH_URI=${MEMGRAPH_URI:-bolt://localhost:7688}
DBS=${DBS:-"neo4j memgraph"}   # space-separated: neo4j memgraph tigergraph

for ds in "$@"; do
  [[ -d "db_export/$ds" ]] || $PY -m src.db.export_csv --datasets "$ds"
  for db in $DBS; do
    echo "== loading $ds into $db =="
    case "$db" in
      neo4j)    $PY -m src.db.load_bolt --backend neo4j --uri "$NEO4J_URI" \
                   --user "$NEO4J_USER" --password "$NEO4J_PASS" --dataset "$ds" --wipe ;;
      memgraph) $PY -m src.db.load_bolt --backend memgraph --uri "$MEMGRAPH_URI" \
                   --dataset "$ds" --wipe ;;
      tigergraph) $PY -m src.db.load_tigergraph --graph "rcf_${ds}" --dataset "$ds" ;;
    esac
  done
done
echo "PASS load_all: $* loaded into [$DBS]"
