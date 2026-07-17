"""Overhead decomposition — why wall-clock UNDERSTATES the method.

Both from-scratch and incremental call the same PyG/PyTorch C++/CUDA kernels;
the difference is op granularity. From-scratch issues a few large kernels
(fixed per-op dispatch/launch overhead amortized over millions of elements);
incremental issues many tiny kernels (mask, unique, gather, small matmul per
relation per layer) whose fixed overhead does NOT amortize. So a large,
constant fraction of the incremental wall-clock is framework glue, not compute
— which biases the wall-clock comparison AGAINST the method.

This experiment separates the two currencies with an exact identity:

    wall_speedup  =  proxy_speedup  /  overhead_ratio

where
  * proxy_speedup  = scratch_edges / inc_edges — the implementation-INDEPENDENT
    algorithmic advantage (immune to the Python tax);
  * overhead_ratio = (wall/edge for incremental) / (wall/edge for from-scratch)
    — how many times more wall-clock the many-small-kernels path spends PER unit
    of work than the few-large-kernels path. This is the framework/granularity
    tax; it is > 1 purely because tiny kernels don't amortize dispatch/launch
    overhead, and it is what a fused/compiled kernel would drive toward 1.

So wall-clock understates the method by exactly overhead_ratio.

  python -m src.eval.exp_overhead --config configs/overhead.yaml
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from src.data.graph_state import GraphState, Intervention
from src.data.loader import load_graph
from src.eval.exactness import check
from src.eval.runner import (ckpt_path, dataset_card, load_config,
                             missing_checkpoints, summary_line, write_csv,
                             write_manifest)
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.models.model import load_model
from src.models.tasks import spec

EXP = "overhead"


def bench(fn, reps):
    walls, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/overhead.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])

    rows, all_ok = [], True
    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        Etot = sum(t.size(1) for t in gs.edge_index.values()) * 2  # +reverse
        Ntot = sum(t.size(0) for t in gs.x.values())

        for arch in cfg["archs"]:
            L = cfg["layers"]
            if missing_checkpoints(dataset, [arch], [L], ["nodelabel"]):
                print(f"SKIP (no checkpoint): {dataset} {arch} L{L}")
                continue
            model = load_model(ckpt_path(dataset, "nodelabel", arch, L))
            cache = build_cache(model, gs.base_view())
            scratch_edges = Etot * L        # full forward aggregates every edge, every layer
            scratch_nodes = Ntot * L

            for kind in ("feature", "topo"):
                for size in cfg["sizes"]:
                    g = torch.Generator().manual_seed(cfg["seed"])
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                    else:
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=size)
                    truth, wall_scr = bench(lambda: whatif_forward(model, gs, iv),
                                            cfg["reps"])
                    res, wall_inc = bench(
                        lambda: incremental_forward(model, gs, cache, iv), cfg["reps"])
                    ok = check(res.h, truth).passed
                    all_ok &= ok
                    inc_edges = max(res.stats["edges_touched"], 1)
                    proxy = scratch_edges / inc_edges
                    wall_sp = wall_scr / wall_inc if wall_inc else float("inf")
                    inc_per_edge = wall_inc / inc_edges
                    scr_per_edge = wall_scr / scratch_edges
                    overhead_ratio = inc_per_edge / scr_per_edge if scr_per_edge else float("inf")
                    rows.append({
                        "dataset": dataset, "arch": arch, "kind": kind, "size": size,
                        "method": res.stats["method"],
                        "inc_edges": res.stats["edges_touched"],
                        "inc_nodes": res.stats["nodes_recomputed"],
                        # sparse ReLU correction size (delta path only; the
                        # closure path recomputes rows fully and reports none)
                        "sign_flips": res.stats.get("sign_flips", ""),
                        "scratch_edges": scratch_edges, "scratch_nodes": scratch_nodes,
                        # implementation-INDEPENDENT algorithmic advantage:
                        "proxy_speedup_edges": proxy,
                        # what the Python prototype actually delivers (lower bound):
                        "wall_scratch_s": wall_scr, "wall_incremental_s": wall_inc,
                        "wall_speedup": wall_sp,
                        # framework/granularity tax: wall/edge_inc vs wall/edge_scratch.
                        # identity: wall_speedup == proxy_speedup / overhead_ratio
                        "inc_wall_per_edge_us": inc_per_edge * 1e6,
                        "scratch_wall_per_edge_us": scr_per_edge * 1e6,
                        "overhead_ratio": overhead_ratio,
                        "exact": ok,
                    })
                    print(f"{dataset} {arch} {kind:7} size={size:<5} "
                          f"proxy x{proxy:8.1f}  wall x{wall_sp:5.1f}  "
                          f"overhead_ratio {overhead_ratio:6.0f}x  "
                          f"(inc {inc_per_edge*1e6:6.2f} vs scratch "
                          f"{scr_per_edge*1e6:.3f} us/edge)", flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok,
                        f"{len(rows)} points; proxy = algorithmic advantage, "
                        f"wall = Python-prototype lower bound")


if __name__ == "__main__":
    raise SystemExit(main())
