#!/usr/bin/env bash
# Paper run on a quiet machine: overhead fix (v2), then the R2.O2 sessions sweep,
# then the v2 + sessions figures. Each step logs to results/logs/<step>_<date>.log,
# headed by the top 5 CPU processes (proof the machine was quiet). Stops at the
# first failure. caffeinate -i keeps the Mac from idle-sleeping.
# Needs the Neo4j container up for the store-backed parts (docker compose up -d neo4j).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=results/logs
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"

step() {   # step <name> <command...>
  local log="$LOGS/$1_$DATE.log"; shift
  { echo "# $(date '+%F %T') $*"; echo "# top 5 CPU:"; { ps -Ao %cpu,comm -r | head -6 || true; }
    echo "# loadavg: $(sysctl -n vm.loadavg)"; } > "$log"
  echo "$(date '+%T') start $*  -> $log"
  if ! caffeinate -i "$@" >> "$log" 2>&1; then
    echo "FAIL $* (see $log)"; exit 1
  fi
  echo "$(date '+%T') done  $*"
}

step run_overhead_fix bash scripts/run_overhead_fix.sh
step run_sessions bash scripts/run_sessions.sh
step figures bash -c "venv/bin/python -m src.eval.figures --v2 && venv/bin/python -m src.eval.figures --sessions"
echo "PASS run_quiet ($DATE)"
