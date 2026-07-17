#!/usr/bin/env bash
# Generate paper figures from whatever results exist.
set -euo pipefail
cd "$(dirname "$0")/.."
venv/bin/python -m src.eval.figures
