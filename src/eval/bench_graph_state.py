"""Latency micro-benchmark for GraphState.apply on the FinBench graph.

Run:  python -m src.eval.bench_graph_state
Correctness lives in tests/test_graph_state.py; this file only times.
"""
import os
import time

import torch

from src.data.graph_state import GraphState, Intervention
from src.paths import FINBENCH_PROC

TR = ("account", "transfer", "account")


def timeit(gs, make_iv, reps=200):
    ivs = [make_iv() for _ in range(reps)]
    t = time.perf_counter()
    for iv in ivs:
        gs.apply(iv)
    return (time.perf_counter() - t) / reps * 1e6  # microseconds


def main():
    data = torch.load(os.path.join(FINBENCH_PROC, "finbench_hetero.pt"),
                      weights_only=False)
    gs = GraphState(data)
    E0 = gs.edge_index[TR].size(1)
    print(f"transfer edges: {E0}")

    g = torch.Generator().manual_seed(0)

    def feat_delta():
        d = torch.zeros_like(gs.x["account"])
        d[3] = 1.0
        return Intervention(feat_deltas={"account": d})

    print("\napply() latency (mean over 200 reps):")
    for n in (1, 100, 1000):
        us = timeit(gs, lambda: Intervention(
            del_cols={TR: torch.randint(0, E0, (n,), generator=g)}))
        print(f"  delete {n:5d} edges: {us:8.1f} us")
    us = timeit(gs, lambda: Intervention(
        add_edges={TR: torch.randint(0, 20000, (2, 1000), generator=g)}))
    print(f"  add     1000 edges: {us:8.1f} us")
    print(f"  feature delta     : {timeit(gs, feat_delta):8.1f} us")

    # one-time pair index build cost (the only O(E) python step, cached)
    t = time.perf_counter()
    gs.cols_for_pairs(TR, [[1, 2]])
    print(f"\npair-index build (once, transfer): {(time.perf_counter() - t) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
