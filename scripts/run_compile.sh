#!/usr/bin/env bash
# torch.compile diagnostic — can compilation close the Python-orchestration gap?
# (Reports honestly: baseline forward compiles ~1.6x; the dynamic maintainer
# graph-breaks and does not. See results/README.md "Two currencies".)
# Diagnostic only — never fails the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
mkdir -p results/compile

for ds in finbench faers eicu snb; do
  $PY -m src.eval.ensure_ckpts --dataset "$ds" --train
  $PY -m src.eval.bench_compile --dataset "$ds" --arch rgcn --layers 2 \
      2>/dev/null | tee "results/compile/${ds}.log"
done
echo "PASS compile: diagnostics written to results/compile/"
