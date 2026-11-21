#!/usr/bin/env bash
# eICU session at scale, 500 states, one machine/one sitting: INQ (7 policies) + naive-GNN on Neo4j and Memgraph.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/sessions_eicu500_$(date +%Y%m%d_%H%M%S).log
{ echo "== top"; top -l 1 -n 5 -o cpu -stats command,cpu | tail -6; } > "$LOG"
A=(--config configs/sessions_eicu500.yaml --configs 0 --exp sessions_eicu500)
caffeinate -i venv/bin/python -W ignore -m src.eval.exp_sessions "${A[@]}" --parts ours store >> "$LOG" 2>&1
caffeinate -i venv/bin/python -W ignore -m src.eval.exp_sessions "${A[@]}" --parts store --store-backend memgraph >> "$LOG" 2>&1
echo "$LOG"
