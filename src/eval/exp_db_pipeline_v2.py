"""Store-backed pipeline, MEASURED inference (R3.O4). Successor of exp_db_pipeline,
whose t_gnn_ms for INQ / Ripple / NeutronRT was a projection (edges processed x
per-edge time of a full pass). Here every method is timed on the same Δ:

  naive-GNN   full forward (encode) on G^Δ tensors; the tensors are built untimed
              (in the pipeline they come from the download, timed separately)
  INQ         incremental_forward(impl="v2", dense_frac=inq_dense_frac) on the base cache + index; RGCN uses the
              delta path for every Δ (edge edits degree-renormalised), hGAT the closure;
              inq_closure (RGCN structure only) keeps the closure path for comparison
  Ripple      Ripple.apply (one shot)
  NeutronRT   batched: every sub-batch of batch_iv applied in sequence, summed
              (each against the base cache: the reference implementation has no
              chained cache update); one-shot: NeutronRT.apply on the whole Δ
  store       _apply_delta + rollback (fixed: one deletion per pair); INQ pays none

Protocol: warm-up, then reps; median, p10, p90 in ms; everything under no_grad.
The inference methods of one setting are timed round-robin (series_rr).
Deviation = max |h_method - h_naive| over all node types (one untimed run).
t_gnn_proj_ms keeps the old projection for comparison; it is never plotted as time.

  python -m src.eval.exp_db_pipeline_v2 --config configs/db.yaml --datasets finbench --dbs neo4j
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from src.baselines.harness import BaselineCache, Counters
from src.baselines.methods import NeutronRT, Ripple
from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.db.roundtrip import (BoltStore, _apply_delta, _build_tensors,
                              _download_raw, _edit_pairs)
from src.eval.exactness import check
from src.eval.exp_db_pipeline import batch_iv, clear_query_cache
from src.eval.runner import (ckpt_path, dataset_card, summary_line, write_csv,
                             write_manifest)
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import encode, set_determinism
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.incremental.index import get_index
from src.models.model import load_model
from src.models.tasks import spec
from src.paths import RESULTS

EXP = "db_pipeline_v2"


def series(fn, reps, warm):
    """Wall ms of fn over reps after warm-up. fn may return a float of seconds to
    report instead of its own wall (used to exclude untimed setup/teardown)."""
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        xs.append((r if isinstance(r, float) else time.perf_counter() - t0) * 1e3)
    return xs


def series_rr(fns, reps, warm):
    """series() for several methods, repetitions interleaved round-robin: rep i of
    every method runs before rep i+1 of any, so a load spike hits all of them."""
    for _ in range(warm):
        for fn in fns.values():
            fn()
    xs = {k: [] for k in fns}
    for _ in range(reps):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            xs[k].append((time.perf_counter() - t0) * 1e3)
    return xs


def stats(prefix, xs):
    return {f"{prefix}_median_ms": float(np.median(xs)),
            f"{prefix}_p10_ms": float(np.percentile(xs, 10)),
            f"{prefix}_p90_ms": float(np.percentile(xs, 90))}


def top5():
    return subprocess.run("ps -Ao %cpu,comm -r | head -6; sysctl -n vm.loadavg",
                          shell=True, capture_output=True, text=True).stdout


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/db.yaml")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--dbs", nargs="+", default=None, help="subset of neo4j memgraph")
    ap.add_argument("--kinds", nargs="+", default=["feature", "topo"])
    ap.add_argument("--feature-nt", default=None, help="node type the attribute Δ edits (default: label type)")
    ap.add_argument("--exp", default=EXP, help="results/<exp>/ output folder")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    set_determinism(cfg.get("seed", 42))
    reps, warm = cfg.get("reps_measured", 10), cfg.get("warmup_measured", 1)
    L = cfg.get("layers", 2)
    dfrac = cfg.get("inq_dense_frac")     # closure_v2 dense blocks (None = off)
    archs = cfg.get("archs", ["rgcn", "hgat"])
    start_top = top5()
    print("# top 5 CPU at start:\n" + start_top, flush=True)

    dbs = {}
    for name in args.dbs or ("neo4j", "memgraph"):
        c = cfg.get(name, {})
        if c.get("enabled"):
            dbs[name] = BoltStore(c["uri"], c.get("user", ""), c.get("password", ""),
                                  c.get("database"))
    if not dbs:
        print("FAIL db_pipeline_v2: no Bolt DB enabled"); return 1
    for st in dbs.values():
        clear_query_cache(st)

    exp = args.exp
    path = os.path.join(RESULTS, exp, "results.csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    done = {(r["db"], r["dataset"], r["arch"], r["kind"], r["size"]) for r in rows}
    ok = all(r["passed"] in ("True", "") for r in rows)

    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset, refresh_card=False)
        gs = GraphState(data)
        sp = spec(dataset)
        fnt = args.feature_nt or sp.label_nt
        n_label = schema.num_nodes[fnt]
        e_core = schema.num_edges[tuple(sp.core_rel)]
        E_stored = sum(t.size(1) for t in gs.edge_index.values())
        scratch_edges = E_stored * 2 * L
        b = max(1, round(cfg.get("neutron_batch_frac", 1e-4) * E_stored))
        models, icaches, bcaches = {}, {}, {}
        pfx = f"{dataset}_"

        for db_name, store in dbs.items():
            def download():
                with store._session() as s:
                    tx = s.begin_transaction()
                    raw = _download_raw(tx, schema, pfx)
                    tx.rollback()
                    return raw
            dl = tb = None              # per (db, dataset), only if something is left

            for kind in args.kinds:
                dim = n_label if kind == "feature" else e_core
                for frac in cfg["fracs"]:
                    size = max(1, round(frac * dim))
                    todo = [a for a in archs
                            if (db_name, dataset, a, kind, str(size)) not in done]
                    if not todo:
                        continue
                    if dl is None:
                        dl = series(download, reps, warm)
                        raw = download()
                        tb = series(lambda: _build_tensors(raw, schema), reps, warm)
                    g = torch.Generator().manual_seed(cfg.get("seed", 42))
                    iv = (sample_feature_iv(schema, gs, g, fnt, size) if kind == "feature"
                          else sample_topo_iv(schema, gs, g, sp.core_rel, n_add=size, n_del=size))
                    adds, dels, cohort = _edit_pairs(gs, iv)
                    cap = cfg.get("db_max_edits", 50000)
                    n_edits = sum(map(len, adds.values())) + sum(map(len, dels.values())) \
                        + sum(map(len, cohort.values()))
                    adds = {et: p[:cap] for et, p in adds.items()}
                    dels = {et: p[:cap] for et, p in dels.items()}
                    cohort = {nt: ids[:cap] for nt, ids in cohort.items()}
                    n_written = sum(map(len, adds.values())) + sum(map(len, dels.values())) \
                        + sum(map(len, cohort.values()))

                    def apply_t():
                        with store._session() as s:
                            tx = s.begin_transaction()
                            t0 = time.perf_counter()
                            _apply_delta(tx, adds, dels, cohort, pfx)
                            dt = time.perf_counter() - t0
                            tx.rollback()
                            return dt

                    def rollback_t():
                        with store._session() as s:
                            tx = s.begin_transaction()
                            _apply_delta(tx, adds, dels, cohort, pfx)
                            t0 = time.perf_counter()
                            tx.rollback()
                            return time.perf_counter() - t0
                    t_ap = series(apply_t, reps, warm)
                    t_rb = series(rollback_t, reps, warm)
                    view = gs.apply(iv)     # G^Δ tensors, as the download would return
                    subs = batch_iv(iv, b)

                    for arch in todo:
                        if arch not in models:
                            models[arch] = load_model(ckpt_path(dataset, "nodelabel", arch, L))
                            icaches[arch] = build_cache(models[arch], gs.base_view())
                            get_index(icaches[arch])      # one-time, like the cache
                            bcaches[arch] = BaselineCache(models[arch], gs)
                        model, icache, bcache = models[arch], icaches[arch], bcaches[arch]
                        truth = encode(model, view)
                        meth = "delta" if arch == "rgcn" else "auto"
                        res = incremental_forward(model, gs, icache, iv, method=meth, impl="v2", dense_frac=dfrac)
                        rep = check(res.h, truth)
                        # (method, impl) -> callable; every one is run once untimed for
                        # its output and operation count, then timed
                        fns = {("naive", "-"): lambda: encode(model, view),
                               ("inq", "-"): lambda: incremental_forward(
                                   model, gs, icache, iv, method=meth, impl="v2", dense_frac=dfrac).h}
                        if arch == "rgcn" and kind == "topo":
                            clo = incremental_forward(model, gs, icache, iv, impl="v2", dense_frac=dfrac)
                            fns[("inq_closure", "-")] = lambda: incremental_forward(
                                model, gs, icache, iv, impl="v2", dense_frac=dfrac).h
                        for impl in ("reference", "optimised"):
                            rip = Ripple(model, gs, icache, impl=impl)
                            neu = NeutronRT(model, gs, icache, impl=impl)
                            fns[("ripple", impl)] = lambda rip=rip: rip.apply(bcache, iv)
                            fns[("neutronrt", impl)] = \
                                lambda neu=neu: [neu.apply(bcache, s) for s in subs]
                            fns[("neutronrt_oneshot", impl)] = lambda neu=neu: neu.apply(bcache, iv)
                        out, edges = {}, {}
                        for k, fn in fns.items():
                            Counters.reset()
                            out[k] = fn()
                            edges[k] = Counters.edges
                        edges[("naive", "-")] = scratch_edges
                        edges[("inq", "-")] = res.stats["edges_touched"]
                        if ("inq_closure", "-") in fns:
                            edges[("inq_closure", "-")] = clo.stats["edges_touched"]
                        t = series_rr(fns, reps, warm)
                        per_edge_naive = np.median(t[("naive", "-")]) / scratch_edges
                        Counters.reset(); neu.apply(bcache, subs[0]); first_batch = Counters.edges
                        meta = {"db": db_name, "dataset": dataset, "arch": arch, "kind": kind, "feature_nt": fnt if kind == "feature" else "",
                                "size": size, "pct": round(frac * 100, 2), "layers": L,
                                "n_batches": len(subs), "batch_edges": b,
                                "batch_frac": cfg.get("neutron_batch_frac", 1e-4), "inq_dense_frac": dfrac,
                                "n_edits": n_edits, "n_edits_written": n_written}
                        store_cols = {**stats("t_apply", t_ap), **stats("t_rollback", t_rb),
                                      **stats("t_download", dl), **stats("t_tensor", tb)}
                        for (m, impl), xs in t.items():
                            # batched NeutronRT: no single answer; its exactness is one-shot's
                            h = out[(m if m != "neutronrt" else "neutronrt_oneshot", impl)]
                            dev = rep if m == "inq" else check(h, truth)
                            ok = ok and (dev.passed or m not in ("inq", "inq_closure"))
                            ref = out.get((m if m != "neutronrt" else "neutronrt_oneshot",
                                           "reference"))
                            med = float(np.median(xs))
                            nonfin = sum(int((~v.isfinite()).sum()) for v in h.values())
                            # the July projection: edges x per-edge time of a full pass;
                            # batched = n_batches x first-batch edges
                            e_proj = len(subs) * first_batch if m == "neutronrt" else edges[(m, impl)]
                            rows.append({**meta, "method": m, "impl": impl, **stats("t_gnn", xs),
                                         "t_gnn_proj_ms": med if m == "naive"
                                         else e_proj * per_edge_naive,
                                         "edges_processed": edges[(m, impl)],
                                         "us_per_edge": med * 1e3 / max(edges[(m, impl)], 1),
                                         # check() drops NaN from its max: non-finite = inf
                                         "max_abs_dev": dev.max_abs_err if not nonfin
                                         else float("inf"),
                                         "exceeds_tol": not dev.passed,
                                         "n_nonfinite": nonfin,
                                         "opt_equals_ref": all(torch.allclose(
                                             h[nt], ref[nt], rtol=0, atol=0, equal_nan=True)
                                             for nt in h) if impl == "optimised" else "",
                                         "inq_path": res.stats["method"] if m == "inq" else "",
                                         "passed": dev.passed if m.startswith("inq") else "",
                                         **({k: 0.0 for k in store_cols} if m.startswith("inq")
                                            else store_cols)})
                        ok = ok and rep.passed
                        print(f"  {db_name} {dataset} {arch} {kind:7} {frac:.2f} sz={size:<6} "
                              + " ".join(f"{m}{'' if i == '-' else '/' + i[:3]}={np.median(xs):7.1f}"
                                         for (m, i), xs in t.items())
                              + f" apply={np.median(t_ap):7.1f} rb={np.median(t_rb):6.1f}"
                              f" inq_exact={rep.passed}", flush=True)
                        write_csv(exp, rows)
    for st in dbs.values():
        st.close()
    write_manifest(exp, {**cfg, "feature_nt": fnt, "kinds": args.kinds, "top5_at_start": start_top},
                   {d: dataset_card(d) for d in cfg["datasets"]})
    return summary_line(exp, ok, f"{len(rows)} rows across {list(dbs)}")


if __name__ == "__main__":
    raise SystemExit(main())
