#!/usr/bin/env bash
# Build any missing processed dataset (raw data must already be in dataset/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

[[ -f dataset/finbench01/processed/finbench_hetero.pt ]] \
  || $PY -m src.data.finbench.prepare_finbench
[[ -f dataset/faers/processed/aggregates.pkl ]] \
  || $PY -m src.data.faers.scan_faers
[[ -f dataset/faers/processed/faers_hetero.pt ]] \
  || $PY -m src.data.faers.build_faers
[[ -f dataset/eicu/processed/eicu_hetero.pt ]] \
  || $PY -m src.data.eicu.build_eicu
[[ -f dataset/snb/processed/snb_hetero.pt ]] \
  || $PY -m src.data.snb.build_snb

echo "PASS prepare_data: all processed graphs present"
