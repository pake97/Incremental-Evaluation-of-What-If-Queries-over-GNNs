"""Scaling experiments with intervention size as a FRACTION of the dataset.

Sweeps dataset x arch x kind x frac x layer, for the two approaches (ours =
incremental best-impl, and from-scratch), recording:
  * operations : edge-messages aggregated + nodes recomputed (impl-independent)
  * latency    : from-scratch = measured full forward; ours = best-implementation
                 roofline (its op-count costed at the from-scratch per-edge rate)
  * sign_flips : ours' nonlinear ReLU corrections (from-scratch has none)
Feeds figures 1 (L=2: ops & latency vs size), 2 (latency vs L, per size) and
3 (sign-flips vs L, per size).

  python -m src.eval.exp_scaling --config configs/scaling.yaml
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

EXP = "scaling"


def bench(fn, reps):
    walls, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scaling.yaml")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    set_determinism(cfg["seed"])
    reps = cfg["reps"]

    rows, all_ok = [], True
    for dataset in datasets:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        n_label = schema.num_nodes[sp.label_nt]
        e_core = schema.num_edges[tuple(sp.core_rel)]
        Etot = sum(t.size(1) for t in gs.edge_index.values()) * 2   # +reverse
        Ntot = sum(t.size(0) for t in gs.x.values())

        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                if missing_checkpoints(dataset, [arch], [L], [cfg["task"]]):
                    print(f"SKIP (no ckpt): {dataset} {arch} L{L}")
                    continue
                model = load_model(ckpt_path(dataset, cfg["task"], arch, L))
                cache = build_cache(model, gs.base_view())
                scratch_edges, scratch_verts = Etot * L, Ntot * L
                for kind in cfg["kinds"]:
                    dim = n_label if kind == "feature" else e_core
                    for frac in cfg["fracs"]:
                        size = max(1, round(frac * dim))
                        g = torch.Generator().manual_seed(cfg["seed"])
                        if kind == "feature":
                            iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                        else:
                            iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                                n_add=size, n_del=size)
                        truth, wall_scr = bench(lambda: whatif_forward(model, gs, iv),
                                                reps)
                        res, wall_inc = bench(
                            lambda: incremental_forward(model, gs, cache, iv), reps)
                        ok = check(res.h, truth).passed
                        all_ok &= ok
                        per_edge = wall_scr / max(scratch_edges, 1)
                        inc_edges = res.stats["edges_touched"]
                        common = {"dataset": dataset, "arch": arch, "kind": kind,
                                  "frac": frac, "size": size, "layers": L}
                        rows.append({**common, "method": "from_scratch",
                                     "edges": scratch_edges, "verts": scratch_verts,
                                     "sign_flips": 0,
                                     "wall_measured_ms": round(wall_scr * 1e3, 3),
                                     "latency_best_ms": round(wall_scr * 1e3, 3),
                                     "exact": True})
                        rows.append({**common, "method": "ours",
                                     "edges": inc_edges,
                                     "verts": res.stats["nodes_recomputed"],
                                     "sign_flips": res.stats.get("sign_flips", 0),
                                     "wall_measured_ms": round(wall_inc * 1e3, 3),
                                     "latency_best_ms": round(inc_edges * per_edge * 1e3, 4),
                                     "exact": ok})
                        print(f"{dataset:8} {arch} L{L} {kind:7} {int(frac*100):>2}% "
                              f"(size={size}) inc_edges={inc_edges:>9} "
                              f"flips={res.stats.get('sign_flips',0):>7} exact={ok}",
                              flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in datasets})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok, f"{len(rows)} rows; all exact={all_ok}")


if __name__ == "__main__":
    raise SystemExit(main())
