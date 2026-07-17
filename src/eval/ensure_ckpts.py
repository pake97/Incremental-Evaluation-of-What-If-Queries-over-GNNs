"""Check the frozen checkpoints an experiment needs; optionally train missing.

  python -m src.eval.ensure_ckpts --dataset finbench            # exit 1 if missing
  python -m src.eval.ensure_ckpts --dataset finbench --train    # train the gaps
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.eval.runner import missing_checkpoints
from src.models.tasks import spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--archs", nargs="+", default=["rgcn", "hgat"])
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=60)
    args = ap.parse_args()

    tasks = args.tasks or spec(args.dataset).tasks
    missing = missing_checkpoints(args.dataset, args.archs, args.layers, tasks)
    if not missing:
        print(f"checkpoints OK: {args.dataset}")
        return 0
    print(f"missing {len(missing)} checkpoints for {args.dataset}:")
    for _, t, a, L in missing:
        print(f"  {t}_{a}_L{L}")
    if not args.train:
        return 1

    # train only the missing combinations (grouped: one call per arch/L pair set)
    archs = sorted({a for _, _, a, _ in missing})
    layers = sorted({L for _, _, _, L in missing})
    tasks_m = sorted({t for _, t, _, _ in missing})
    cmd = [sys.executable, "-m", "src.models.train", "--dataset", args.dataset,
           "--archs", *archs, "--layers", *map(str, layers),
           "--tasks", *tasks_m, "--max-epochs", str(args.max_epochs)]
    print("+", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
