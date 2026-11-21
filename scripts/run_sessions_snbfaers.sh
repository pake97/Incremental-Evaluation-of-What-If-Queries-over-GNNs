#!/usr/bin/env bash
# SNB + FAERS sessions, one sitting: INQ (7 policies) + naive-GNN on Neo4j and Memgraph (100 states).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/sessions_snbfaers_$(date +%Y%m%d_%H%M%S).log
{ echo "== top"; top -l 1 -n 5 -o cpu -stats command,cpu | tail -6; } > "$LOG"
A=(--config configs/sessions_snbfaers.yaml --configs 0 --exp sessions_snbfaers)
caffeinate -i venv/bin/python -W ignore -m src.eval.exp_sessions "${A[@]}" --parts ours store >> "$LOG" 2>&1
caffeinate -i venv/bin/python -W ignore -m src.eval.exp_sessions "${A[@]}" --parts store --store-backend memgraph >> "$LOG" 2>&1
echo "$LOG"
