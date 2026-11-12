"""SIGMOD revision measurements — one experiment, five parts.

  A  cache size: ForwardCache bytes (nbytes + independent storage count), per
     layer / tensor kind / node type, lazily built extras (_deg, _ei_all), and
     context (graph tensors, model params, analytic estimate).
  B  memory per query (L=2): peak-RSS growth of one query in a FRESH subprocess,
     plus the copy-on-write state the incremental result retains.
  C  one-time build cost (L=2): in-memory build_cache vs plain encode; store-
     backed download / tensor build / build_cache per Bolt backend.
  D  MEASURED latency (median, p10, p90 over reps after warm-up) of whatif_forward
     vs incremental_forward on the same base cache, closure statistics, and the
     old results/scaling numbers side by side.
  E  break-even number of queries k*, in-memory and store-backed.

Instrumentation only: calls src/incremental, src/forward and src/db as-is.
Every part is resumable (settings already in its CSV are skipped). A row that
fails the exactness gate is kept in latency.csv with passed=False and is never
used by Part E or the figures.

  python -m src.eval.exp_revision --config configs/revision.yaml --parts A B C D E report
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import yaml

from src.data.graph_state import GraphState, GraphView
from src.data.loader import load_graph
from src.eval.exactness import check
from src.eval.runner import (ckpt_path, dataset_card, load_config,
                             missing_checkpoints, summary_line, write_csv,
                             write_manifest)
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import encode, set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.incremental.index import build_index, get_index
from src.incremental.overlay import OverlayCache
from src.models.model import load_model
from src.models.tasks import spec
from src.paths import RESULTS

EXP = "revision"                 # set from --exp in main()
OUT = os.path.join(RESULTS, EXP)
IMPL = "v1"                      # set from --impl in main()
MB = 2 ** 20


# ---------------------------------------------------------------- shared setup
class Setting:
    """One dataset loaded once: graph, schema, sizes — as in exp_scaling."""

    def __init__(self, dataset):
        self.dataset = dataset
        data, self.schema = load_graph(dataset, refresh_card=False)
        self.gs = GraphState(data)
        self.sp = spec(dataset)
        self.n_label = self.schema.num_nodes[self.sp.label_nt]
        self.e_core = self.schema.num_edges[tuple(self.sp.core_rel)]
        self.Etot = sum(t.size(1) for t in self.gs.edge_index.values()) * 2  # +reverse
        self.Ntot = sum(t.size(0) for t in self.gs.x.values())

    def iv(self, kind, frac, seed):
        """Byte-identical to exp_scaling's sampling (same generator, same size)."""
        dim = self.n_label if kind == "feature" else self.e_core
        size = max(1, round(frac * dim))
        g = torch.Generator().manual_seed(seed)
        if kind == "feature":
            return size, sample_feature_iv(self.schema, self.gs, g, self.sp.label_nt, size)
        return size, sample_topo_iv(self.schema, self.gs, g, self.sp.core_rel,
                                    n_add=size, n_del=size)


def tbytes(t):
    return t.numel() * t.element_size()


def read_rows(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fkey(x):
    return f"{float(x):.4f}"


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def has_ckpt(dataset, arch, L, task):
    return not missing_checkpoints(dataset, [arch], [L], [task])


# ---------------------------------------------------------------- Part A
def part_a(cfg, settings):
    long_rows = read_rows("cache_size.csv")
    summ = read_rows("cache_size_summary.csv")
    done = {(r["dataset"], r["arch"], r["L"]) for r in summ}
    for ds in cfg["datasets"]:
        st = settings(ds)
        gs = st.gs
        graph_b = (sum(tbytes(t) for t in gs.x.values())
                   + sum(tbytes(t) for t in gs.edge_index.values()))
        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                if (ds, arch, str(L)) in done or not has_ckpt(ds, arch, L, cfg["task"]):
                    continue
                model = load_model(ckpt_path(ds, cfg["task"], arch, L))
                cache = build_cache(model, gs.base_view())
                index_b = get_index(cache).nbytes() if IMPL == "v2" else 0
                H = model.encoder.hidden
                base = dict(dataset=ds, arch=arch, L=L)

                # h/z tensors: nbytes() dedups by id; check against a storage-
                # level count (dedup by data_ptr) that does not share its logic.
                seen_id, seen_ptr, storage_b, dtypes = set(), set(), 0, set()
                for kind, layers in (("h", cache.h), ("z", cache.z)):
                    for li, layer in enumerate(layers):
                        for nt, t in layer.items():
                            alias = id(t) in seen_id
                            seen_id.add(id(t))
                            ptr = t.untyped_storage().data_ptr()
                            if ptr not in seen_ptr:
                                seen_ptr.add(ptr)
                                storage_b += t.untyped_storage().nbytes()
                            dtypes.add(str(t.dtype))
                            long_rows.append({**base, "component": "cache", "tensor": kind,
                                              "layer": li, "node_type": nt,
                                              "dtype": str(t.dtype), "rows": t.size(0),
                                              "cols": t.size(1), "bytes": tbytes(t),
                                              "aliased": alias})
                nb = cache.nbytes()
                assert nb == storage_b, f"nbytes {nb} != storage count {storage_b}"

                # A.2 lazily built extras after one feature + one topo query
                for kind in ("feature", "topo"):
                    _, iv = st.iv(kind, cfg["extras_frac"], cfg["seed"])
                    incremental_forward(model, gs, cache, iv, impl=IMPL)
                graph_ids = ({id(t) for t in cache.edge_index_dict.values()}
                             | {id(t) for t in gs.edge_index.values()})
                extras_b = 0
                for comp, d in (("extras_deg", cache._deg), ("extras_ei_all", cache._ei_all)):
                    for et, t in d.items():
                        if id(t) in graph_ids:
                            continue
                        extras_b += tbytes(t)
                        long_rows.append({**base, "component": comp, "tensor": comp[7:],
                                          "layer": "", "node_type": "/".join(et),
                                          "dtype": str(t.dtype), "rows": t.size(0),
                                          "cols": t.size(1) if t.dim() > 1 else 1,
                                          "bytes": tbytes(t), "aliased": False})

                params_b = sum(tbytes(p) for p in model.parameters())
                analytic_b = (2 * L + 1) * sum(gs.num_nodes[nt] for nt in cache.h[0]) * H * 4
                summ.append({**base, "dtype": "/".join(sorted(dtypes)),
                             "cache_bytes": nb, "storage_bytes": storage_b,
                             "cache_MB": nb / MB, "extras_MB": extras_b / MB,
                             "index_MB": index_b / MB,
                             "graph_MB": graph_b / MB, "params_MB": params_b / MB,
                             "analytic_MB": analytic_b / MB,
                             "analytic_over_cache": analytic_b / nb})
                print(f"A {ds:8} {arch} L{L} cache={nb/MB:8.2f}MB extras={extras_b/MB:6.2f}MB "
                      f"analytic={analytic_b/MB:8.2f}MB", flush=True)
                write_csv(EXP, long_rows, "cache_size.csv")
                write_csv(EXP, summ, "cache_size_summary.csv")


# ---------------------------------------------------------------- Part B
def maxrss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024   # macOS bytes, Linux KB


def mem_worker(cfg, ds, arch, kind, frac, method):
    """Runs in a fresh process: identical setup for both methods, then ONE
    query; prints a JSON line."""
    import psutil
    L = cfg["fixed_layers"]
    st = Setting(ds)
    model = load_model(ckpt_path(ds, cfg["task"], arch, L))
    cache = build_cache(model, st.gs.base_view())
    if IMPL == "v2":
        get_index(cache)                 # session index: part of setup, as the cache
    _, iv = st.iv(kind, float(frac), cfg["seed"])
    gc.collect()
    peak0, cur0 = maxrss_bytes(), psutil.Process().memory_info().rss
    if method == "ours":
        res = incremental_forward(model, st.gs, cache, iv, impl=IMPL)
    else:
        res = whatif_forward(model, st.gs, iv)
    peak1 = maxrss_bytes()
    out = {"peak_extra_MB": (peak1 - peak0) / MB,
           "peak_over_current_MB": (peak1 - cur0) / MB,
           "setup_peak_MB": peak0 / MB, "setup_rss_MB": cur0 / MB}
    if method == "ours" and isinstance(res.cache, OverlayCache):
        # v2: row overlays (+ the dense output layer) on top of the base; the
        # graph is the base + the intervention it references (edit list, dense
        # feature delta as sampled)
        nc = res.cache
        n_ov = sum(1 for layer in nc.h + nc.z for e in getattr(layer, "raw", layer).values()
                   if type(e).__name__ == "RowOverlay")
        out.update(retained_hz_MB=nc.nbytes() / MB,
                   retained_graph_MB=sum(tbytes(t) for d in (iv.add_edges, iv.del_cols,
                                                             iv.feat_deltas)
                                         for t in d.values()) / MB,
                   tensors_cloned=n_ov, tensors_shared=-1,
                   inc_method=res.stats["method"])
    elif method == "ours":
        nc = res.cache
        base_ids = {id(t) for layer in cache.h + cache.z for t in layer.values()}
        seen, retained_hz, cloned, shared = set(), 0, 0, 0
        for layer in nc.h + nc.z:
            for t in layer.values():
                if id(t) in seen:
                    continue
                seen.add(id(t))
                if id(t) in base_ids:
                    shared += 1
                else:
                    cloned += 1
                    retained_hz += tbytes(t)
        gids = ({id(t) for t in st.gs.x.values()}
                | {id(t) for t in st.gs.edge_index.values()})
        retained_graph = sum(tbytes(t) for d in (nc.x_dict, nc.edge_index_dict)
                             for t in d.values() if id(t) not in gids)
        out.update(retained_hz_MB=retained_hz / MB, retained_graph_MB=retained_graph / MB,
                   tensors_cloned=cloned, tensors_shared=shared,
                   inc_method=res.stats["method"])
    print("RESULT " + json.dumps(out))


def part_b(cfg, config_path):
    rows = read_rows("query_memory.csv")
    done = {(r["dataset"], r["arch"], r["kind"], fkey(r["frac"]), r["method"]) for r in rows}
    L = cfg["fixed_layers"]
    jobs = [(ds, arch, kind, frac, m)
            for ds in cfg["datasets"] for arch in cfg["archs"]
            if has_ckpt(ds, arch, L, cfg["task"])
            for kind in cfg["kinds"] for frac in cfg["fracs"]
            for m in ("ours", "from_scratch")
            if (ds, arch, kind, fkey(frac), m) not in done]

    def run(job):
        ds, arch, kind, frac, m = job
        reps = []
        for _ in range(cfg["reps_memory"]):
            p = subprocess.run([sys.executable, "-m", "src.eval.exp_revision",
                                "--config", config_path, "--impl", IMPL, "--exp", EXP,
                                "--mem-worker",
                                ds, arch, kind, str(frac), m],
                               capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if p.returncode or not line:
                raise RuntimeError(f"mem worker failed {job}:\n{p.stderr[-2000:]}")
            reps.append(json.loads(line[0][7:]))
        agg = {k: statistics.median(r[k] for r in reps) for k in reps[0]
               if isinstance(reps[0][k], (int, float))}
        agg.update({k: v for k, v in reps[0].items() if isinstance(v, str)})
        return {"dataset": ds, "arch": arch, "kind": kind, "frac": frac, "layers": L,
                "method": m, "reps": len(reps), **agg,
                "peak_extra_MB_min": min(r["peak_extra_MB"] for r in reps),
                "peak_extra_MB_max": max(r["peak_extra_MB"] for r in reps)}

    # datasets in the outer order so slow faers jobs do not all land together
    with ThreadPoolExecutor(cfg["mem_workers"]) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            print(f"B {row['dataset']:8} {row['arch']} {row['kind']:7} {row['frac']:.2f} "
                  f"{row['method']:12} peak_extra={row['peak_extra_MB']:8.2f}MB", flush=True)
            write_csv(EXP, rows, "query_memory.csv")


# ---------------------------------------------------------------- Part C
def part_c1(cfg, settings):
    rows = read_rows("build_cost.csv")
    done = {(r["dataset"], r["arch"]) for r in rows if r["backend"] == "memory"}
    L, reps = cfg["fixed_layers"], cfg["reps_build"]
    for ds in cfg["datasets"]:
        for arch in cfg["archs"]:
            if (ds, arch) in done or not has_ckpt(ds, arch, L, cfg["task"]):
                continue
            st = settings(ds)
            view = st.gs.base_view()
            model = load_model(ckpt_path(ds, cfg["task"], arch, L))
            enc = model.encoder

            def lazy_extras():
                # everything the delta path may memoize: an upper bound (it
                # only touches relations leaving the cohort's node type)
                c = build_cache(model, view)
                t0 = time.perf_counter()
                c.ei_all(enc.fwd_types)
                for et in enc.all_types:
                    c.deg(et, st.gs.num_nodes[et[2]])
                return time.perf_counter() - t0

            # build_cache measured FIRST in this process: its first run is the
            # honest one-time cost; encode's first run then sees warm kernels.
            phases = [("build_cache", lambda: timed(lambda: build_cache(model, view))[1]),
                      ("encode", lambda: timed(lambda: encode(model, view))[1])]
            if IMPL == "v2":
                n = st.gs.num_nodes
                phases.append(("index_build", lambda: build_index(view.edge_index_dict, n).build_s))
            elif arch == "rgcn":
                phases.append(("lazy_extras", lazy_extras))
            for phase, fn in phases:
                first = fn()
                warm = statistics.median(fn() for _ in range(reps))
                rows.append({"dataset": ds, "arch": arch, "backend": "memory",
                             "phase": phase, "first_s": first, "warm_median_s": warm,
                             "status": "ok"})
                print(f"C1 {ds:8} {arch} {phase:12} first={first:.3f}s warm={warm:.3f}s",
                      flush=True)
            write_csv(EXP, rows, "build_cost.csv")


def part_c2(cfg, settings):
    from src.db.roundtrip import BoltStore, _build_tensors, _download_raw
    from src.eval.exp_db_pipeline import clear_query_cache
    rows = [r for r in read_rows("build_cost.csv") if r["status"] != "skipped"]
    done = {(r["dataset"], r["backend"]) for r in rows if r["backend"] != "memory"}
    dbc = yaml.safe_load(open(cfg["db_config"]))
    L, reps = cfg["fixed_layers"], cfg["reps_build"]
    for backend in ("neo4j", "memgraph"):
        c = dbc.get(backend, {})
        if not c.get("enabled"):
            continue
        try:
            store = BoltStore(c["uri"], c.get("user", ""), c.get("password", ""),
                              c.get("database"))
            with store._session() as s:
                s.run("RETURN 1").consume()
        except Exception as e:  # DB not running: mark and move on
            reason = f"skipped: {type(e).__name__}: {str(e)[:120]}".replace("\n", " ")
            print(f"C2 {backend}: {reason}", flush=True)
            rows.append({"dataset": "*", "arch": "*", "backend": backend, "phase": "*",
                         "first_s": "", "warm_median_s": "", "status": "skipped",
                         "note": reason})
            continue
        for ds in dbc["datasets"]:
            if ds not in cfg["datasets"] or (ds, backend) in done:
                continue
            st = settings(ds)
            pfx = f"{ds}_"

            def download():
                with store._session() as s:
                    tx = s.begin_transaction()
                    raw = _download_raw(tx, st.schema, pfx)
                    tx.rollback()
                return raw

            # cold: empty plan cache (Neo4j; no-op on Memgraph). The DB page
            # cache is NOT flushed — only a container restart does that.
            clear_query_cache(store)
            raw, t_dl0 = timed(download)
            (x, ei), t_tb0 = timed(lambda: _build_tensors(raw, st.schema))
            view = GraphView(x, ei, {})
            warm_dl = statistics.median(timed(download)[1] for _ in range(reps))
            warm_tb = statistics.median(timed(lambda: _build_tensors(raw, st.schema))[1]
                                        for _ in range(reps))
            n_edges = sum(len(v) for v in raw[1].values())
            phases = [("download", t_dl0, warm_dl), ("tensors", t_tb0, warm_tb)]
            if IMPL == "v2":
                n = st.gs.num_nodes
                ib = [build_index(ei, n).build_s for _ in range(reps + 1)]
                phases.append(("index_build", ib[0], statistics.median(ib[1:])))
            for phase, f, w in phases:
                rows.append({"dataset": ds, "arch": "*", "backend": backend, "phase": phase,
                             "first_s": f, "warm_median_s": w, "status": "ok",
                             "note": f"edges={n_edges}"})
            for arch in cfg["archs"]:
                if not has_ckpt(ds, arch, L, cfg["task"]):
                    continue
                model = load_model(ckpt_path(ds, cfg["task"], arch, L))
                for phase, fn in (("build_cache", lambda: build_cache(model, view)),
                                  ("encode", lambda: encode(model, view))):
                    first = timed(fn)[1]
                    warm = statistics.median(timed(fn)[1] for _ in range(reps))
                    rows.append({"dataset": ds, "arch": arch, "backend": backend,
                                 "phase": phase, "first_s": first, "warm_median_s": warm,
                                 "status": "ok", "note": "on DB-built tensors"})
            print(f"C2 {backend} {ds:8} download first={t_dl0:.2f}s warm={warm_dl:.2f}s "
                  f"tensors warm={warm_tb:.2f}s", flush=True)
            write_csv(EXP, rows, "build_cost.csv")
        store.close()
    write_csv(EXP, rows, "build_cost.csv")


# ---------------------------------------------------------------- Part D
def pct(xs, q):
    return float(np.percentile(xs, q))


def part_d(cfg, settings):
    rows = read_rows("latency.csv")
    done = {(r["dataset"], r["arch"], r["kind"], fkey(r["frac"]), r["layers"]) for r in rows}
    old = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "scaling", "results.csv"))):
        old[(r["dataset"], r["arch"], r["kind"], fkey(r["frac"]), r["layers"],
             r["method"])] = r
    v1rows = {}                      # measured v1 (results/revision) for side-by-side
    v1path = os.path.join(RESULTS, "revision", "latency.csv")
    if IMPL != "v1" and os.path.exists(v1path):
        for r in csv.DictReader(open(v1path)):
            v1rows[(r["dataset"], r["arch"], r["kind"], fkey(r["frac"]), r["layers"])] = r
    reps, warm = cfg["reps_latency"], cfg["warmup_latency"]
    for ds in cfg["datasets"]:
        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                todo = [(k, f) for k in cfg["kinds"] for f in cfg["fracs"]
                        if (ds, arch, k, fkey(f), str(L)) not in done]
                if not todo or not has_ckpt(ds, arch, L, cfg["task"]):
                    continue
                st = settings(ds)
                gs = st.gs
                model = load_model(ckpt_path(ds, cfg["task"], arch, L))
                cache = build_cache(model, gs.base_view())
                if IMPL == "v2":
                    get_index(cache)     # one-time, like the cache (Part C counts it)
                for kind, frac in todo:
                    size, iv = st.iv(kind, frac, cfg["seed"])

                    def series(fn):
                        for _ in range(warm):
                            fn()
                        walls, out = [], None
                        for _ in range(reps):
                            t0 = time.perf_counter()
                            out = fn()
                            walls.append((time.perf_counter() - t0) * 1e3)
                        return out, walls

                    truth, ws = series(lambda: whatif_forward(model, gs, iv))
                    res, wo = series(lambda: incremental_forward(model, gs, cache, iv,
                                                                 impl=IMPL))
                    rep = check(res.h, truth)
                    s = res.stats
                    if IMPL != "v1":     # untimed cross-check against v1
                        r1 = incremental_forward(model, gs, cache, iv, impl="v1")
                        rep_v1 = check(res.h, r1.h)
                        same_stats = all(r1.stats[k] == s[k] for k in
                                         ("nodes_recomputed", "edges_touched"))
                    per_layer = [sum(d.values()) for d in s["dirty_frontier"]]
                    k6 = (ds, arch, kind, fkey(frac), str(L))
                    o_s, o_o = old.get(k6 + ("from_scratch",)), old.get(k6 + ("ours",))
                    row = {"dataset": ds, "arch": arch, "kind": kind, "frac": frac,
                           "size": size, "layers": L, "inc_method": s["method"],
                           "scratch_median_ms": statistics.median(ws),
                           "scratch_p10_ms": pct(ws, 10), "scratch_p90_ms": pct(ws, 90),
                           "ours_median_ms": statistics.median(wo),
                           "ours_p10_ms": pct(wo, 10), "ours_p90_ms": pct(wo, 90),
                           "speedup_measured": statistics.median(ws) / statistics.median(wo),
                           "impl": IMPL, "max_abs_err": rep.max_abs_err,
                           "passed": rep.passed and (IMPL == "v1" or (rep_v1.passed
                                                                      and same_stats)),
                           "nodes_recomputed": s["nodes_recomputed"],
                           "edges_touched": s["edges_touched"],
                           "edges_scanned": s.get("edges_scanned", ""),
                           "sign_flips": s["sign_flips"],
                           "rows_dirty_sum": sum(per_layer),
                           "frontier_by_layer": json.dumps(s["dirty_frontier"]),
                           "closure_frac": per_layer[-1] / st.Ntot,
                           "edges_frac": s["edges_touched"] / (L * st.Etot),
                           "n_total": st.Ntot, "e_total_incl_rev": st.Etot,
                           "old_scratch_wall_ms": o_s and float(o_s["wall_measured_ms"]),
                           "old_ours_wall_ms": o_o and float(o_o["wall_measured_ms"]),
                           "old_ours_best_ms": o_o and float(o_o["latency_best_ms"])}
                    for li in range(4):
                        row[f"rows_l{li + 1}"] = per_layer[li] if li < L else ""
                    if IMPL != "v1":
                        v1 = v1rows.get(k6, {})
                        row.update(vs_v1_max_abs_err=rep_v1.max_abs_err,
                                   v1_stats_equal=same_stats,
                                   v1_edges_scanned=r1.stats["edges_scanned"],
                                   v1_ours_median_ms=v1.get("ours_median_ms", ""),
                                   v1_scratch_median_ms=v1.get("scratch_median_ms", ""))
                    if o_o and o_o["edges"] != str(s["edges_touched"]):
                        print(f"  WARN edges_touched {s['edges_touched']} != scaling "
                              f"{o_o['edges']} (sampling drift?)", flush=True)
                    rows.append(row)
                    print(f"D {ds:8} {arch} L{L} {kind:7} {frac:.2f} scratch="
                          f"{row['scratch_median_ms']:9.2f} ours={row['ours_median_ms']:9.2f} "
                          f"x{row['speedup_measured']:6.2f} closure={row['closure_frac']:.3f} "
                          f"exact={rep.passed}", flush=True)
                    write_csv(EXP, rows, "latency.csv")


# ---------------------------------------------------------------- Part F
def part_f(cfg, settings):
    """REFERENCE ONLY: from-scratch with the encoder under torch.compile, same
    settings and protocol as Part D (compile happens in the warm-up runs)."""
    import torch._dynamo as dynamo
    for k in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, k):
            setattr(dynamo.config, k, 256)    # never fall back to eager silently
    rows = read_rows("compiled.csv")
    done = {(r["dataset"], r["arch"], r["kind"], fkey(r["frac"]), r["layers"]) for r in rows}
    reps, warm = cfg["reps_latency"], max(cfg["warmup_latency"], 2)
    for ds in cfg["datasets"]:
        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                todo = [(k, f) for k in cfg["kinds"] for f in cfg["fracs"]
                        if (ds, arch, k, fkey(f), str(L)) not in done]
                if not todo or not has_ckpt(ds, arch, L, cfg["task"]):
                    continue
                st = settings(ds)
                model = load_model(ckpt_path(ds, cfg["task"], arch, L))
                dynamo.reset()
                cenc = torch.compile(model.encoder)
                for kind, frac in todo:
                    _, iv = st.iv(kind, frac, cfg["seed"])

                    def run():
                        with torch.no_grad():
                            v = st.gs.apply(iv)
                            return cenc(v.x_dict, v.edge_index_dict)
                    for _ in range(warm):
                        out = run()
                    walls = [timed(run)[1] * 1e3 for _ in range(reps)]
                    rep = check(out, whatif_forward(model, st.gs, iv))
                    rows.append({"dataset": ds, "arch": arch, "kind": kind, "frac": frac,
                                 "layers": L, "scratch_compiled_median_ms": statistics.median(walls),
                                 "scratch_compiled_p10_ms": pct(walls, 10),
                                 "scratch_compiled_p90_ms": pct(walls, 90),
                                 "max_abs_err": rep.max_abs_err, "passed": rep.passed})
                    print(f"F {ds:8} {arch} L{L} {kind:7} {frac:.2f} compiled scratch="
                          f"{rows[-1]['scratch_compiled_median_ms']:9.2f} exact={rep.passed}",
                          flush=True)
                    write_csv(EXP, rows, "compiled.csv")


# ---------------------------------------------------------------- Part E
def kstar(build_s, t_ours, t_scratch):
    """Smallest integer k with build + k*t_ours < k*t_scratch (inf if never)."""
    gain = t_scratch - t_ours
    if gain <= 0:
        return float("inf")
    return int(build_s // gain) + 1


def part_e(cfg):
    L = cfg["fixed_layers"]
    lat = [r for r in read_rows("latency.csv")
           if r["layers"] == str(L) and r["passed"] == "True"]
    bc = read_rows("build_cost.csv")
    mem = {(r["dataset"], r["arch"], r["phase"]): float(r["warm_median_s"])
           for r in bc if r["backend"] == "memory"}
    store = {}   # (backend, dataset, arch|*, phase) -> s
    for r in bc:
        if r["backend"] != "memory" and r["status"] == "ok":
            store[(r["backend"], r["dataset"], r["arch"], r["phase"])] = float(r["warm_median_s"])
    src_store = "build_cost.csv (this run)"
    if not store:
        # C.2 skipped: fall back to the per-step medians of the last
        # exp_db_pipeline run (download/tensor are method- and arch-independent)
        import pandas as pd
        p = os.path.join(RESULTS, "db_pipeline", "results.csv")
        if os.path.exists(p):
            dp = pd.read_csv(p)
            dp = dp[(dp.method == "from_scratch")]
            for (db, ds), g in dp.groupby(["db", "dataset"]):
                store[(db, ds, "*", "download")] = g.t_download_ms.median() / 1e3
                store[(db, ds, "*", "tensors")] = g.t_tensor_ms.median() / 1e3
            src_store = "results/db_pipeline/results.csv (previous run; C.2 skipped)"

    rows = []
    for r in lat:
        ds, arch = r["dataset"], r["arch"]
        t_o, t_s = float(r["ours_median_ms"]) / 1e3, float(r["scratch_median_ms"]) / 1e3
        b_cache = mem.get((ds, arch, "build_cache"))
        if b_cache is None:
            continue
        b_extra = mem.get((ds, arch, "index_build" if IMPL == "v2" else "lazy_extras"), 0.0)
        base = {"dataset": ds, "arch": arch, "kind": r["kind"], "frac": r["frac"],
                "layers": L, "t_ours_s": t_o}
        k = kstar(b_cache + b_extra, t_o, t_s)
        rows.append({**base, "backend": "memory", "k_star": k,
                     "k_star_build_cache_only": kstar(b_cache, t_o, t_s),
                     "build_s": b_cache, "lazy_extras_s": b_extra, "t_scratch_s": t_s,
                     "source": "build_cost.csv memory + latency.csv"})
        for backend in ("neo4j", "memgraph"):
            dl = store.get((backend, ds, "*", "download"))
            tb = store.get((backend, ds, "*", "tensors"))
            if dl is None or tb is None:
                continue
            # build/encode on DB tensors if C.2 ran, else the in-memory medians
            bcache = store.get((backend, ds, arch, "build_cache"), b_cache)
            enc = store.get((backend, ds, arch, "encode"), mem.get((ds, arch, "encode")))
            if IMPL == "v2":            # index built on the DB tensors when C.2 ran
                b_extra = store.get((backend, ds, "*", "index_build"), b_extra)
            one_time = dl + tb + bcache + b_extra
            t_scr = dl + tb + enc     # LOWER bound: ignores apply/rollback of Δ in the DB
            rows.append({**base, "backend": backend, "k_star": kstar(one_time, t_o, t_scr),
                         "k_star_build_cache_only": kstar(dl + tb + bcache, t_o, t_scr),
                         "build_s": one_time, "lazy_extras_s": b_extra,
                         "t_scratch_s": t_scr, "download_s": dl, "tensors_s": tb,
                         "encode_s": enc, "source": src_store})
    write_csv(EXP, rows, "breakeven.csv")
    print(f"E {len(rows)} break-even rows; store inputs from {src_store}", flush=True)


# ---------------------------------------------------------------- report tables
def report():
    import pandas as pd
    R = lambda n: pd.read_csv(os.path.join(OUT, n))
    out = []
    def md(df):
        f = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
        lines = ["| " + " | ".join(map(str, df.columns)) + " |",
                 "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(f(v) for v in r) + " |"
                  for r in df.itertuples(index=False)]
        return "\n".join(lines)

    s = R("cache_size_summary.csv")
    s = s[s.L == 2]
    bc = R("build_cost.csv")
    piv = bc[(bc.phase == "build_cache") & (bc.status == "ok")].pivot_table(
        index=["dataset", "arch"], columns="backend", values="warm_median_s").reset_index()
    t1 = s.merge(piv, on=["dataset", "arch"], how="left")
    cols = ["dataset", "arch", "cache_MB", "extras_MB", "graph_MB"] + \
           [c for c in ("memory", "neo4j", "memgraph") if c in t1]
    t1 = t1[cols].rename(columns={c: f"build_s_{c}" for c in ("memory", "neo4j", "memgraph")})
    # store-backed one-time = download + tensors + build_cache (break-even inputs)
    okb = bc[bc.status == "ok"]
    for b in ("neo4j", "memgraph"):
        w = {(r.dataset, r.arch, r.phase): r.warm_median_s
             for r in okb[okb.backend == b].itertuples()}
        if w:
            t1[f"one_time_s_{b}"] = [w.get((d, "*", "download"), float("nan"))
                                     + w.get((d, "*", "tensors"), float("nan"))
                                     + w.get((d, a, "build_cache"), float("nan"))
                                     for d, a in zip(t1.dataset, t1.arch)]
    out.append("### T1 — cache footprint and one-time build (L=2)\n\n" + md(t1))
    st = bc[bc.backend != "memory"]
    if (st.status == "ok").any():
        out.append("Store-backed phases (first_s = cold: first run after a container restart + plan-cache clear; warm = median):\n\n"
                   + md(st[st.status == "ok"][["dataset", "arch", "backend", "phase",
                                                "first_s", "warm_median_s"]]))
    else:
        out.append("Store-backed build (C.2): **skipped** — "
                   + "; ".join(st.note.dropna().str.replace("\n", " ").unique()))

    lat = R("latency.csv")
    ok = lat[lat.passed]
    t2 = ok[(ok.layers == 2) & (ok.frac.isin([0.05, 0.30]))].copy()
    t2["speedup_old_best"] = t2.old_scratch_wall_ms / t2.old_ours_best_ms
    t2["speedup_old_wall"] = t2.old_scratch_wall_ms / t2.old_ours_wall_ms
    t2 = t2[["dataset", "arch", "kind", "frac", "scratch_median_ms", "ours_median_ms",
             "speedup_measured", "speedup_old_wall", "speedup_old_best", "closure_frac"]]
    out.append("### T2 — measured speedup at L=2 (5% and 30%) vs the old projection\n\n"
               + md(t2.sort_values(["dataset", "arch", "kind", "frac"])))

    be = R("breakeven.csv")
    t3 = be[be.frac == 0.10].pivot_table(index=["dataset", "arch", "kind"],
                                         columns="backend", values="k_star",
                                         aggfunc="first").reset_index()
    t3 = t3.replace(float("inf"), "no break-even")
    out.append("### T3 — break-even k* at 10% (L=2)\n\n" + md(t3))

    t4 = ok[ok.ours_median_ms > ok.scratch_median_ms][
        ["dataset", "arch", "layers", "kind", "frac", "inc_method", "scratch_median_ms",
         "ours_median_ms", "speedup_measured", "closure_frac", "edges_frac"]]
    out.append(f"### T4 — settings where measured ours is SLOWER ({len(t4)} of {len(ok)})\n\n"
               + md(t4.sort_values(["dataset", "arch", "layers", "kind", "frac"])))
    failed = lat[~lat.passed]
    out.append(f"Exactness gate: {len(lat) - len(failed)}/{len(lat)} rows passed; "
               f"max_abs_err over all rows = {lat.max_abs_err.max():.3g}.")

    qm = R("query_memory.csv")
    pm = qm.pivot_table(index=["dataset", "arch", "kind", "frac"], columns="method",
                        values=["peak_extra_MB", "peak_over_current_MB"])
    pm.columns = [f"{v}:{m}" for v, m in pm.columns]
    out.append("### Memory per query (L=2, median over fresh processes)\n\n"
               + md(pm.reset_index()))
    if "retained_hz_MB" in qm:
        out.append("Retained copy-on-write state of ours' new cache:\n\n" + md(
            qm[qm.method == "ours"][["dataset", "arch", "kind", "frac", "inc_method",
                                     "retained_hz_MB", "retained_graph_MB",
                                     "tensors_cloned", "tensors_shared"]]))
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n\n".join(out) + "\n")
    print(f"wrote {os.path.join(OUT, 'tables.md')}")
    return lat


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/revision.yaml")
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C", "D", "E", "report"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--impl", default="v1", choices=["v1", "v2"])
    ap.add_argument("--exp", default="revision", help="results/<exp>/")
    ap.add_argument("--mem-worker", nargs=5, default=None,
                    metavar=("DS", "ARCH", "KIND", "FRAC", "METHOD"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.datasets:
        cfg["datasets"] = args.datasets
    global EXP, OUT, IMPL
    EXP, OUT, IMPL = args.exp, os.path.join(RESULTS, args.exp), args.impl
    cfg["impl"] = IMPL
    set_determinism(cfg["seed"])
    if args.mem_worker:
        return mem_worker(cfg, *args.mem_worker) or 0

    os.makedirs(OUT, exist_ok=True)
    cache = {}

    def settings(ds):   # one loaded dataset at a time keeps RAM bounded
        if ds not in cache:
            cache.clear()
            gc.collect()
            cache[ds] = Setting(ds)
        return cache[ds]

    tpath = os.path.join(OUT, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}
    steps = {"A": lambda: part_a(cfg, settings), "B": lambda: part_b(cfg, args.config),
             "C": lambda: (part_c1(cfg, settings), part_c2(cfg, settings)),
             "D": lambda: part_d(cfg, settings), "E": lambda: part_e(cfg),
             "F": lambda: part_f(cfg, settings), "report": report}
    for p in args.parts:
        _, dt = timed(steps[p])
        timings[p] = timings.get(p, 0.0) + dt
        print(f"part {p} done in {dt:.1f}s", flush=True)
        json.dump(timings, open(tpath, "w"), indent=2)

    write_manifest(EXP, {**cfg, "torch_threads": torch.get_num_threads()},
                   {d: dataset_card(d) for d in cfg["datasets"]})
    lat = read_rows("latency.csv")
    n_fail = sum(r["passed"] != "True" for r in lat)
    return summary_line(EXP, n_fail == 0,
                        f"{len(lat)} latency rows; exactness failures={n_fail}")


if __name__ == "__main__":
    raise SystemExit(main())
