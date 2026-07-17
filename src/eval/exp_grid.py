"""Full grid — the comprehensive single-query comparison.

Sweeps, per dataset: intervention kind {feature, edge_add, edge_del, edge_swap}
x size {1,100,1000} x depth L {1,2,3,4} x architecture {rgcn, hgat}, and records
for each cell BOTH the incremental and the from-scratch cost, in two currencies:

  * operations — edge-messages aggregated (implementation-independent);
  * latency    — best-implementation latency = the method's operations costed at
                 the measured per-edge rate of the dense from-scratch kernel
                 (i.e. the maintainer running at the same kernel efficiency as
                 the baseline). from-scratch latency is measured directly.

Plus, for feature Δ on RGCN, the number of ReLU sign-flips (the nonlinear
correction the low-rank path must apply), which grows with L and size.

  python -m src.eval.exp_grid --config configs/grid.yaml
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

EXP = "grid"


def bench(fn, reps):
    walls, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, statistics.median(walls)


def make_iv(kind, schema, gs, g, sp, size):
    if kind == "feature":
        return sample_feature_iv(schema, gs, g, sp.label_nt, size)
    if kind == "edge_add":
        return sample_topo_iv(schema, gs, g, sp.core_rel, n_add=size, n_del=0)
    if kind == "edge_del":
        return sample_topo_iv(schema, gs, g, sp.core_rel, n_add=0, n_del=size)
    return sample_topo_iv(schema, gs, g, sp.core_rel, n_add=size, n_del=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    set_determinism(cfg["seed"])

    rows, all_ok = [], True
    for dataset in datasets:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        Etot = sum(t.size(1) for t in gs.edge_index.values()) * 2  # +reverse
        Ntot = sum(t.size(0) for t in gs.x.values())

        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                if missing_checkpoints(dataset, [arch], [L], [cfg["task"]]):
                    print(f"SKIP (no checkpoint): {dataset} {arch} L{L}")
                    continue
                model = load_model(ckpt_path(dataset, cfg["task"], arch, L))
                cache = build_cache(model, gs.base_view())
                scratch_edges = Etot * L
                scratch_nodes = Ntot * L

                for kind in cfg["kinds"]:
                    for size in cfg["sizes"]:
                        g = torch.Generator().manual_seed(cfg["seed"])
                        iv = make_iv(kind, schema, gs, g, sp, size)
                        truth, wall_scr = bench(
                            lambda: whatif_forward(model, gs, iv), cfg["reps"])
                        res, wall_inc = bench(
                            lambda: incremental_forward(model, gs, cache, iv),
                            cfg["reps"])
                        ok = check(res.h, truth).passed
                        all_ok &= ok
                        inc_edges = max(res.stats["edges_touched"], 1)
                        scr_per_edge = wall_scr / scratch_edges
                        # best-implementation latency: method's work at the dense
                        # kernel's per-edge efficiency
                        inc_best_s = inc_edges * scr_per_edge
                        rows.append({
                            "dataset": dataset, "arch": arch, "layers": L,
                            "kind": kind, "size": size, "method": res.stats["method"],
                            "inc_edges": res.stats["edges_touched"],
                            "scratch_edges": scratch_edges,
                            "inc_nodes": res.stats["nodes_recomputed"],
                            "scratch_nodes": scratch_nodes,
                            "sign_flips": res.stats.get("sign_flips", ""),
                            "wall_scratch_s": wall_scr,
                            "wall_incremental_best_s": inc_best_s,
                            "latency_speedup": wall_scr / inc_best_s if inc_best_s else float("inf"),
                            "ops_speedup": scratch_edges / inc_edges,
                            "exact": ok,
                        })
                        sf = res.stats.get("sign_flips", "")
                        print(f"{dataset:8} {arch} L{L} {kind:9} size={size:<5} "
                              f"ops x{rows[-1]['ops_speedup']:8.1f}  "
                              f"lat x{rows[-1]['latency_speedup']:8.1f}  "
                              f"flips={sf}", flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in datasets})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok, f"{len(rows)} grid cells, all exact={all_ok}")


if __name__ == "__main__":
    raise SystemExit(main())
