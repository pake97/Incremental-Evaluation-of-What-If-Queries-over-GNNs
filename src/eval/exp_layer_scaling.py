"""Experiment 3 — layer scaling.

Same protocol as single-query efficiency but sweeping depth L = 1..4 at fixed
intervention sizes: how incremental cost and speedup change with depth (the
locality closure grows with L; the delta path's flip-correction support grows
with L too).

  python -m src.eval.exp_layer_scaling --config configs/layer_scaling.yaml
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

EXP = "layer_scaling"


def bench(fn, reps):
    walls = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/layer_scaling.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])

    rows, all_ok = [], True
    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        for arch in cfg["archs"]:
            for L in (1, 2, 3, 4):
                task = cfg["task"]
                if missing_checkpoints(dataset, [arch], [L], [task]):
                    print(f"SKIP (no checkpoint): {dataset} {task} {arch} L{L}")
                    continue
                model = load_model(ckpt_path(dataset, task, arch, L))
                cache = build_cache(model, gs.base_view())
                for kind in ("feature", "topo"):
                    g = torch.Generator().manual_seed(cfg["seed"])
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt,
                                               cfg["cohort_size"])
                    else:
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=cfg["topo_edits"],
                                            n_del=cfg["topo_edits"])
                    truth, wall_scr = bench(lambda: whatif_forward(model, gs, iv),
                                            cfg["reps"])
                    res, wall_inc = bench(
                        lambda: incremental_forward(model, gs, cache, iv),
                        cfg["reps"])
                    ok = check(res.h, truth).passed
                    all_ok &= ok
                    frontier = res.stats.get("dirty_frontier", [])
                    rows.append({
                        "dataset": dataset, "arch": arch, "layers": L, "kind": kind,
                        "method": res.stats["method"],
                        "wall_scratch_s": wall_scr, "wall_incremental_s": wall_inc,
                        "speedup": wall_scr / wall_inc if wall_inc > 0 else float("inf"),
                        "nodes_recomputed": res.stats["nodes_recomputed"],
                        "edges_touched": res.stats["edges_touched"],
                        "closure_final_frontier": (sum(frontier[-1].values())
                                                   if frontier else None),
                        "exact": ok,
                    })
                    print(f"{dataset} {arch} L{L} {kind:8} "
                          f"scratch={wall_scr*1e3:8.1f}ms inc={wall_inc*1e3:8.1f}ms "
                          f"x{rows[-1]['speedup']:.1f}", flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok, f"{len(rows)} points, all exact={all_ok}")


if __name__ == "__main__":
    raise SystemExit(main())
