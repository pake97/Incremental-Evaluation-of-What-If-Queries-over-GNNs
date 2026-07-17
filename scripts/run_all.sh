#!/usr/bin/env bash
# Full pipeline in dependency order. Fails loudly and early; a failed
# exactness gate stops everything downstream (never silently skipped).
set -euo pipefail
cd "$(dirname "$0")"

./run_prepare_data.sh
./run_train.sh
./run_exactness.sh                  # THE GATE — timing below is meaningless without it
./run_single_query_efficiency.sh
./run_layer_scaling.sh
./run_overhead.sh
./run_grid.sh                       # full kind x size x L x arch single-query grid
./run_baselines.sh                  # ours vs Ripple vs NeutronRT vs from-scratch
./run_db.sh                         # store-backed round-trip (skipped if no configs/db.yaml)
./run_compile.sh                    # diagnostic: torch.compile can't fuse the dynamic maintainer
./run_exploration_reuse.sh
./run_figures.sh

echo "PASS run_all: full pipeline complete"
