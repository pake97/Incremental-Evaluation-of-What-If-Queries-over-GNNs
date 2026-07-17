"""Baseline comparison experiment (see baselines.md).

Compares OUR incremental method against the published streaming-incremental
primitives — Ripple (linear look-forward delta) and NeutronRT (decoupled
operators) — plus from-scratch, on the forward counterfactual query. Two
experiments (feature Δ over a cohort; topological Δ = edge edits), sizes
{1,100,1000}, per dataset, RGCN + hGAT, L=2. Reports op-counts (the clean
signal), wall-clock (median), and exactness (MSE + max-abs vs full recompute).

  python -m src.eval.exp_baselines --config configs/baselines.yaml
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from src.baselines.harness import BaselineCache, Counters, evaluate
from src.baselines.methods import FromScratch, NeutronRT, Ours, Ripple
from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.eval.runner import (ckpt_path, dataset_card, load_config,
                             missing_checkpoints, summary_line, write_csv,
                             write_manifest)
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.models.model import load_model
from src.models.tasks import spec

EXP = "baselines"
METHODS = ["from_scratch", "ripple", "neutronrt", "ours"]


def affected_fraction(truth, base_h, tol=1e-6):
    tot = chg = 0
    for nt in truth:
        d = (truth[nt] - base_h[nt]).abs().sum(-1)
        chg += int((d > tol).sum())
        tot += d.numel()
    return chg / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    set_determinism(cfg["seed"])

    rows = []
    for dataset in datasets:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        for arch in cfg["archs"]:
            if missing_checkpoints(dataset, [arch], [cfg["layers"]], ["nodelabel"]):
                print(f"SKIP (no checkpoint): {dataset} {arch}")
                continue
            model = load_model(ckpt_path(dataset, "nodelabel", arch, cfg["layers"]))
            bcache = BaselineCache(model, gs)
            icache = build_cache(model, gs.base_view())
            base_h = bcache.h_final
            methods = {
                "from_scratch": FromScratch(model, gs),
                "ripple": Ripple(model, gs, icache),
                "neutronrt": NeutronRT(model, gs, icache),
                "ours": Ours(model, gs, icache),
            }
            for kind in ("feature", "topo"):
                for size in cfg["sizes"]:
                    g = torch.Generator().manual_seed(cfg["seed"])
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                    else:
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=size)
                    truth = whatif_forward(model, gs, iv)
                    aff = affected_fraction(truth, base_h)
                    for nm in METHODS:
                        m = methods[nm]
                        res = evaluate(m, bcache, iv, truth, aff)
                        walls = []
                        for _ in range(cfg["reps"]):
                            t0 = time.perf_counter()
                            m.apply(bcache, iv)
                            walls.append(time.perf_counter() - t0)
                        rows.append({
                            "dataset": dataset, "arch": arch, "kind": kind,
                            "size": size, "method": nm,
                            "edges": res.edges, "verts": res.verts,
                            "wall_s": statistics.median(walls),
                            "mse": res.mse, "max_abs": res.max_abs,
                            "affected_frac": round(aff, 4),
                            "degenerate": res.degenerate,
                        })
                        print(f"{dataset:8} {arch} {kind:7} sz={size:<5} {nm:12} "
                              f"edges={res.edges:>9} maxabs={res.max_abs:.1e}"
                              f"{'  [DEGEN]' if res.degenerate else ''}", flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in datasets})
    write_csv(EXP, rows)
    # exactness sanity: NeutronRT + ours must be ~exact everywhere; that gates the comparison
    bad = [r for r in rows if r["method"] in ("ours", "neutronrt")
           and r["max_abs"] > 1e-3]
    ok = not bad
    return summary_line(EXP, ok, f"{len(rows)} rows; ours+NeutronRT exact everywhere={ok}"
                        + (f" ({len(bad)} violations)" if bad else ""))


if __name__ == "__main__":
    raise SystemExit(main())
