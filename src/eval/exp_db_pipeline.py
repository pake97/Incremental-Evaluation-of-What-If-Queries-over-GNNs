"""Store-backed what-if PIPELINE with per-step timing (exploded).

Times each step of serving a what-if when the graph lives in a property store,
so the figure can break the cost down by step.

OURS  (track steps 2-3; step 1 DB-load is one-time setup, untimed):
  1. load graph in DB               [setup, not timed]
  2. download graph + build X,A     [ONE-TIME at session start]  -> t_build
  3. apply our incremental (best)   [PER QUERY]                  -> t_incr

FROM-SCRATCH (track 2-5, all PER QUERY):
  2. apply Δ to DB (update query)   -> t_apply
  3. download graph + build X,A     -> t_build
  4. GNN from scratch (full fwd)    -> t_fwd
  5. rollback the update on the DB  -> t_rollback

STREAMING baselines (Ripple / NeutronRT): same shape as from-scratch but the
GNN step is their incremental method; still pays the DB download + update every
query (committed-timeline store, no disposable overlay):
  t_apply + t_build + t_method + t_rollback   [PER QUERY]

Emits per (db, dataset, kind, size) the per-step milliseconds for each method.

  python -m src.eval.exp_db_pipeline --config configs/db.yaml
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import yaml

from src.baselines.harness import BaselineCache, Counters
from src.baselines.methods import NeutronRT, Ripple
from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.db.roundtrip import (BoltStore, _apply_delta, _build_tensors,
                              _download_raw, _edit_pairs)
from src.eval.runner import dataset_card, summary_line, write_csv, write_manifest
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import set_determinism
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.models.model import load_model
from src.models.tasks import spec
from src.eval.runner import ckpt_path

EXP = "db_pipeline"


def med(fn, reps, warmup=0):
    for _ in range(warmup):          # discarded: excludes plan-compile / cold pages
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def clear_query_cache(store):
    """Best-effort clear of the DB query-plan cache (so a run starts from a known
    state). Neo4j: CALL db.clearQueryCaches(). Memgraph: no-op (unsupported)."""
    try:
        with store._session() as s:
            s.run("CALL db.clearQueryCaches()").consume()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/db.yaml")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="override cfg['datasets'] (e.g. just finbench)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    set_determinism(cfg.get("seed", 42))
    reps = cfg.get("reps", 3)
    L = cfg.get("layers", 2)
    archs = cfg.get("archs", ["rgcn", "hgat"])

    dbs = {}
    for name in ("neo4j", "memgraph"):
        c = cfg.get(name, {})
        if c.get("enabled"):
            dbs[name] = BoltStore(c["uri"], c.get("user", ""), c.get("password", ""),
                                  c.get("database"))
    if not dbs:
        print("FAIL db_pipeline: no Bolt DB enabled"); return 1
    print("connected:", ", ".join(dbs))
    for st in dbs.values():          # start each run from a cold plan cache
        clear_query_cache(st)

    fracs = cfg.get("fracs")   # if set, intervention size = frac x dataset dim

    rows = []
    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset, refresh_card=False)
        gs = GraphState(data)
        sp = spec(dataset)
        n_label = schema.num_nodes[sp.label_nt]
        e_core = schema.num_edges[tuple(sp.core_rel)]
        models = {a: load_model(ckpt_path(dataset, "nodelabel", a, L)) for a in archs}
        icaches = {a: build_cache(models[a], gs.base_view()) for a in archs}
        bcaches = {a: BaselineCache(models[a], gs) for a in archs}
        pfx = f"{dataset}_"

        for db_name, store in dbs.items():
            # DOWNLOAD and TENSOR-BUILD are method- and Δ-independent (same graph,
            # tiny Δ), so measure each once per (db,dataset) and reuse. Split so
            # tensor construction is its own visible, identical-for-all cost.
            def download():
                with store._session() as s:
                    tx = s.begin_transaction()
                    raw = _download_raw(tx, schema, pfx)
                    tx.rollback()
                    return raw
            t_download = med(download, max(2, reps // 2)) * 1e3
            raw = download()
            t_tensor = med(lambda: _build_tensors(raw, schema), max(2, reps)) * 1e3
            xa = _build_tensors(raw, schema)   # tensors for the from-scratch GNN

            for kind in ("feature", "topo"):
                dim = n_label if kind == "feature" else e_core
                sizes = ([max(1, round(f * dim)) for f in fracs] if fracs
                         else cfg["sizes"])
                for size in sizes:
                    g = torch.Generator().manual_seed(cfg.get("seed", 42))
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                    else:
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=size)
                    adds, dels, cohort = _edit_pairs(gs, iv)
                    # Cap the edits actually WRITTEN to the DB for the apply/rollback
                    # timing (a huge transaction would OOM the store, and apply cost
                    # is negligible vs the download anyway). GNN steps use full size.
                    cap = cfg.get("db_max_edits", 50000)
                    adds = {et: p[:cap] for et, p in adds.items()}
                    dels = {et: p[:cap] for et, p in dels.items()}
                    cohort = {nt: ids[:cap] for nt, ids in cohort.items()}

                    # apply Δ and rollback are DB ops the baselines/from-scratch pay
                    # every query (in the chained flow apply -> download -> rollback);
                    # measured separately here, negligible vs the download.
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
                    # WARM UP once per size (compile plan + warm pages) so the
                    # timed samples measure pure update work, not one-time compile.
                    apply_t(); rollback_t()
                    t_apply = statistics.median([apply_t() for _ in range(reps)]) * 1e3
                    t_rollback = statistics.median([rollback_t() for _ in range(reps)]) * 1e3

                    for arch in archs:
                        model = models[arch]
                        icache, bcache = icaches[arch], bcaches[arch]
                        # --- GNN steps (in-memory), per architecture ---
                        # from-scratch: measured full-forward wall (a few large kernels,
                        # already near-optimal). Incremental methods: report the impl-
                        # INDEPENDENT roofline (op-count x from-scratch per-edge rate),
                        # matching latency_best_ms in the scaling figures. Raw prototype
                        # wall-clock would be dominated by per-kernel framework overhead
                        # and understate the method (see results/README "Two currencies").
                        Etot = sum(t.size(1) for t in gs.edge_index.values()) * 2
                        scratch_edges = Etot * L
                        t_fwd = med(lambda: model.encode(xa[0], xa[1]), reps) * 1e3
                        per_edge = t_fwd / max(scratch_edges, 1)
                        inc_edges = incremental_forward(model, gs, icache, iv).stats["edges_touched"]
                        Counters.reset(); Ripple(model, gs, icache).apply(bcache, iv)
                        rip_edges = Counters.edges
                        Counters.reset(); NeutronRT(model, gs, icache).apply(bcache, iv)
                        neu_edges = Counters.edges
                        t_incr = inc_edges * per_edge
                        t_ripple = rip_edges * per_edge
                        t_neu = neu_edges * per_edge

                        meta = {"db": db_name, "dataset": dataset, "arch": arch,
                                "kind": kind, "size": size,
                                "pct": round(size / dim * 100)}
                        # OURS: never touches the DB per query (Δ is hypothetical,
                        # in memory) -> apply/download/tensor/rollback = 0 per query;
                        # download+tensor is a one-time session cost.
                        rows.append({**meta, "method": "ours",
                                     "t_apply_ms": 0.0, "t_download_ms": 0.0,
                                     "t_tensor_ms": 0.0, "t_rollback_ms": 0.0,
                                     "t_gnn_ms": round(t_incr, 3),
                                     "per_query_ms": round(t_incr, 3),
                                     "one_time_ms": round(t_download + t_tensor, 2)})
                        # from-scratch & baselines: per query = apply Δ + ONE
                        # download + build tensors + GNN + rollback.
                        for m, tg in (("from_scratch", t_fwd), ("ripple", t_ripple),
                                      ("neutronrt", t_neu)):
                            rows.append({**meta, "method": m,
                                         "t_apply_ms": round(t_apply, 3),
                                         "t_download_ms": round(t_download, 2),
                                         "t_tensor_ms": round(t_tensor, 2),
                                         "t_rollback_ms": round(t_rollback, 3),
                                         "t_gnn_ms": round(tg, 3),
                                         "per_query_ms": round(t_apply + t_download
                                              + t_tensor + tg + t_rollback, 2),
                                         "one_time_ms": 0.0})
                        print(f"  {db_name} {dataset} {arch} {kind:7} sz={size:<5} "
                              f"download={t_download:7.0f} tensor={t_tensor:6.0f} "
                              f"apply={t_apply:5.1f} rb={t_rollback:5.1f} "
                              f"fwd={t_fwd:7.1f} incr={t_incr:7.1f}", flush=True)
    for st in dbs.values():
        st.close()
    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, True, f"{len(rows)} rows (per-step) across {list(dbs)}")


if __name__ == "__main__":
    raise SystemExit(main())
