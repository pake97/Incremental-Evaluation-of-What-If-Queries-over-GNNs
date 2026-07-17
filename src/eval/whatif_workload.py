"""State-management overhead of serving a WHAT-IF workload (commit + drop).

A streaming/committed-timeline system (Ripple, NeutronRT) has no disposable
overlay: to answer a hypothetical Δ and then return to the base graph for the
next sibling hypothetical, it must COMMIT the batch (write the affected
per-layer embeddings into its persistent store) and then UNDO it (restore those
embeddings). Our method treats Δ as a copy-on-write overlay and DROPS it — the
base cache is never mutated, so commit+undo cost is ~0.

This is the cost of retrofitting a committed-timeline system to the branching,
disposable what-if workload (paper reason iii — state integrity). It is a
memory/state cost of order O(affected nodes) — same order as the maintenance
itself, i.e. a constant-factor argument, reported alongside (not folded into)
the maintenance ops.

  python -m src.eval.whatif_workload
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

from src.eval.runner import write_csv
from src.paths import RESULTS

HIDDEN = 64                       # src.models.train.HIDDEN (fixed)
COMMITTING = {"ripple", "neutronrt"}   # maintain + mutate a persistent cache


def _write_restore_rate() -> float:
    """Scalars per second for a commit(write)+undo(restore) of embedding rows,
    measured on this machine."""
    store = torch.zeros(200_000, HIDDEN)
    rows = torch.arange(50_000)
    new = torch.randn(rows.numel(), HIDDEN)
    saved = store[rows].clone()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        store[rows] = new          # commit
        store[rows] = saved        # undo
    dt = (time.perf_counter() - t0) / reps
    scalars = 2 * rows.numel() * HIDDEN
    return scalars / dt


def main():
    path = os.path.join(RESULTS, "baselines", "results.csv")
    if not os.path.exists(path):
        print("FAIL whatif_workload: run exp_baselines first")
        return 1
    df = pd.read_csv(path)
    rate = _write_restore_rate()
    print(f"commit+undo memory rate: {rate/1e9:.2f} Gscalar/s")

    def state_scalars(r):
        return 2 * int(r.verts) * HIDDEN if r.method in COMMITTING else 0

    df = df.copy()
    df["state_write_scalars"] = df.apply(state_scalars, axis=1)
    df["commit_undo_ms"] = df.state_write_scalars / rate * 1e3
    # what-if per-hypothetical cost = maintenance wall + state management
    df["maintenance_ms"] = df.wall_s * 1e3
    df["whatif_ms"] = df.maintenance_ms + df.commit_undo_ms
    write_csv("baselines", df.to_dict("records"), "whatif_workload.csv")

    # headline: how much does commit+undo add for the committing baselines?
    comm = df[df.method.isin(COMMITTING)]
    frac = (comm.commit_undo_ms / comm.whatif_ms.clip(lower=1e-9)).mean()
    print(f"\nFor committed-timeline baselines, commit+undo is on average "
          f"{100*frac:.0f}% of the per-hypothetical what-if cost.")
    print("Ours / from-scratch pay 0 (overlay dropped / no persistent cache).")
    print(f"\nWrote results/baselines/whatif_workload.csv")
    print("PASS whatif_workload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
