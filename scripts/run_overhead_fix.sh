#!/usr/bin/env bash
# Overhead fix (impl="v2"): exactness gate first (v1 and v2, v2 also vs v1),
# then the revision experiment with v2 into results/revision_v2, the v2
# figures, the v1-vs-v2 profile and tables (results/overhead_fix).
# The gate fails loudly and stops everything (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

# Fresh v2 timings: archive the Oct 28 run (parts skip settings already in their
# CSVs). Once only, so a re-run after a failure resumes instead of re-archiving.
# store_size* stay: they are inputs (src/db/store_size), not exp_revision outputs.
OLD=results/revision_v2_oct28
if [ ! -d "$OLD" ]; then
  mkdir -p "$OLD"
  for f in results/revision_v2/*.csv results/revision_v2/timings.json; do
    if [ -e "$f" ] && [ "$(basename "$f")" != store_size.csv ]; then mv "$f" "$OLD/"; fi
  done
fi

$PY -m pytest tests/ -q
bash scripts/run_exactness.sh
for impl in v1 v2; do
  $PY -m src.eval.profile_overhead --impl $impl --out results/overhead_fix/profile_$impl.json
done
# C.2 (store-backed) runs only when the Bolt DBs answer; otherwise it is marked skipped
$PY -m src.eval.exp_revision --config configs/revision.yaml --impl v2 --exp revision_v2 \
    --parts A C D F B E report
$PY -m src.eval.figures --v2
$PY -m src.eval.report_overhead_fix
