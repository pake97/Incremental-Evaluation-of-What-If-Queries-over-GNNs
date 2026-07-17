"""Experiment 2 — single-query efficiency.

Incremental vs from-scratch latency for feature and topological Δ, across
datasets/models, varying intervention size/locality. Baseline = from-scratch
full recompute. Reports wall-clock medians AND hardware-independent proxies
(#nodes recomputed, #edges aggregated). Runs only if the exactness gate passed
(run_all.sh enforces the order; the per-query results here are spot-checked).

  python -m src.eval.exp_single_query --config configs/single_query.yaml
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from src.data.graph_state import GraphState
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

EXP = "single_query_efficiency"


def bench(fn, reps):
    walls = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, statistics.median(walls)


def time_graph_load(dataset: str, reps: int = 3) -> float:
    """Median wall-clock of loading the processed graph from disk — the
    setup cost a cold from-scratch pipeline pays per job and a resident
    system pays once per session."""
    import os

    import torch

    from src.data.loader import GRAPHS
    path = os.path.join(*GRAPHS[dataset])
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.load(path, weights_only=False)
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_query.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])

    rows, all_ok = [], True
    for dataset in cfg["datasets"]:
        load_wall = time_graph_load(dataset)
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)

        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                task = cfg["task"]
                if missing_checkpoints(dataset, [arch], [L], [task]):
                    print(f"SKIP (no checkpoint): {dataset} {task} {arch} L{L}")
                    continue
                model = load_model(ckpt_path(dataset, task, arch, L))
                t0 = time.perf_counter()
                cache = build_cache(model, gs.base_view())
                cache_wall = time.perf_counter() - t0

                # topo split into add-only / delete-only / swap: deletions
                # change the normalized-aggregator denominator, additions bring
                # in new sources — distinct maintenance work worth separating.
                sweeps = ([("feature", c) for c in cfg["cohort_sizes"]]
                          + [("edge_add", e) for e in cfg["edit_sizes"]]
                          + [("edge_del", e) for e in cfg["edit_sizes"]]
                          + [("edge_swap", e) for e in cfg["edit_sizes"]])
                for kind, size in sweeps:
                    g = torch.Generator().manual_seed(cfg["seed"])
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                    elif kind == "edge_add":
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=0)
                    elif kind == "edge_del":
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=0, n_del=size)
                    else:  # edge_swap
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=size)
                    truth, wall_scr = bench(lambda: whatif_forward(model, gs, iv),
                                            cfg["reps"])
                    res, wall_inc = bench(
                        lambda: incremental_forward(model, gs, cache, iv),
                        cfg["reps"])
                    ok = check(res.h, truth).passed  # untimed spot check
                    all_ok &= ok
                    # Timing model (see results/README.md):
                    #   warm         — both sides resident; per-query marginal cost
                    #   cold one-off — each side pays its own setup for ONE query
                    #   cold vs resident — naive per-job pipeline vs kept-alive system
                    #   break-even   — queries until the session amortizes the cache
                    gap = wall_scr - wall_inc
                    rows.append({
                        "dataset": dataset, "arch": arch, "layers": L,
                        "kind": kind, "size": size,
                        "method": res.stats["method"],
                        "wall_scratch_s": wall_scr, "wall_incremental_s": wall_inc,
                        "graph_load_s": load_wall, "cache_build_s": cache_wall,
                        "speedup": wall_scr / wall_inc if wall_inc > 0 else float("inf"),
                        "speedup_cold_oneoff": (load_wall + wall_scr)
                                               / (load_wall + cache_wall + wall_inc),
                        "speedup_cold_vs_resident": (load_wall + wall_scr) / wall_inc
                                                    if wall_inc > 0 else float("inf"),
                        "break_even_queries": (cache_wall / gap if gap > 0
                                               else float("inf")),
                        "nodes_recomputed": res.stats["nodes_recomputed"],
                        "edges_touched": res.stats["edges_touched"],
                        "exact": ok,
                    })
                    print(f"{dataset} {arch} L{L} {kind:8} size={size:<6} "
                          f"scratch={wall_scr*1e3:8.1f}ms inc={wall_inc*1e3:8.1f}ms "
                          f"x{rows[-1]['speedup']:.1f}", flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok, f"{len(rows)} sweep points, all exact={all_ok}")


if __name__ == "__main__":
    raise SystemExit(main())
