"""R2.O2 session experiment (REVISION_PLAN.md §7): whole exploration sessions.

For every (dataset, arch, configuration) ONE 500-state DAG trace
(src/explore/trace.generate) is evaluated state by state in visit order:

  scratch        whatif_forward on the composed intervention, one full pass per state
  ours           v2 reuse: resume from the nearest materialised ancestor with the
                 residual delta (run_trace(rung="reuse", impl="v2")), under the
                 retention policies max_caches in {all, 16, 1} (1 = base only)
  store_scratch  Neo4j: apply the state's delta on an open transaction, download the
                 full graph, build tensors, full pass, roll back (eICU, FinBench,
                 configuration 0, first 100 states)

The 10 / 50 / 100 / 200 points are prefixes of the same trace. Ours runs the whole
trace reps_ours[policy] times (median per state, median cumulative); scratch once; one
untimed warm-up state first. Exactness vs whatif_forward every check_every states
(untimed); a failure aborts the run.

Store-backed feature deltas: _apply_delta writes the cohort in the store (SET n.f),
but the downloaded `feat` property is the base feature, so the delta is added to the
DB-built X in memory (inside tensors_s). Topology comes from the store as edited.
Exactness gate for store_scratch = the DB-built graph equals gs.apply(iv) (X bitwise,
edge multiset per relation); its output is not compared at ATOL because the store
returns edges in a different order and hGAT's from-scratch sums are not reassociation-
invariant at 1e-4 (eICU base graph: 1.5e-4 on |h| <= 31; RGCN 4e-6). The output
difference to in-memory from-scratch is recorded (out_vs_memory_max_abs_err).

  python -m src.eval.exp_sessions --parts generate ours scratch store figures
  python -m src.eval.exp_sessions --datasets eicu --configs 0 --n-states 30 --exp sessions_pilot
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
import yaml

from src.data.graph_state import GraphView
from src.eval.exactness import check
from src.eval.exp_revision import Setting, has_ckpt
from src.eval.runner import (ckpt_path, dataset_card, load_config, summary_line,
                             write_csv, write_manifest)
from src.explore.reuse import run_trace
from src.explore.state import compose
from src.explore.trace import TraceConfig, generate
from src.forward.forward import encode, set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.index import get_index
from src.models.model import load_model
from src.paths import RESULTS

EXP = "sessions"
OUT = os.path.join(RESULTS, EXP)
BIG = 10 ** 9                      # max_caches for policy "all"


def cname(c):
    return f"b{c[0]}_d{c[1]}_bt{c[2]}"


def read_rows(name):
    path = os.path.join(OUT, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


class Traces:
    """One trace per (dataset, configuration), generated once per process and
    checked against the structure recorded by the `generate` part."""

    def __init__(self, cfg):
        self.cfg, self.mem = cfg, {}
        p = os.path.join(OUT, "traces.json")
        self.saved = json.load(open(p)) if os.path.exists(p) else {}

    def get(self, st, ci):
        key = f"{st.dataset}/{cname(self.cfg['configs'][ci])}"
        if key not in self.mem:
            b, d, bt = self.cfg["configs"][ci]
            tc = TraceConfig(n_states=self.cfg["n_states"], branching=b, max_depth=d,
                             backtrack_rate=bt, node_type=st.sp.label_nt,
                             relation=tuple(st.sp.core_rel), seed=self.cfg["seed"],
                             **self.cfg["trace"])
            tr = generate(st.schema, st.gs, tc)
            s = tr.structure()
            if key in self.saved and self.saved[key]["structure"] != s:
                raise SystemExit(f"trace {key} does not match traces.json: {s}")
            # edges added + deleted on the path to each state, summed over visits
            cum, acc = [], 0
            for state in tr.order:
                iv = compose(tr.registry, state)
                acc += (sum(c.numel() for c in iv.del_cols.values() if c is not None)
                        + sum(a.size(1) for a in iv.add_edges.values() if a is not None))
                cum.append(acc)
            self.mem.clear()
            self.mem[key] = (tr, s, cum)
        return self.mem[key]


def part_generate(cfg, settings, traces, cis):
    out = dict(traces.saved)
    for ds in cfg["datasets"]:
        st = settings(ds)
        for ci in cis:
            tr, s, cum = traces.get(st, ci)
            depths = [len(x) for x in tr.order]
            out[f"{ds}/{cname(cfg['configs'][ci])}"] = {
                "structure": s, "depth_hist": {str(d): depths.count(d) for d in sorted(set(depths))},
                "edges_applied_total": cum[-1]}
            print(f"trace {ds} {cname(cfg['configs'][ci])}: {s}", flush=True)
    json.dump(out, open(os.path.join(OUT, "traces.json"), "w"), indent=1)
    traces.saved = out


def _done(rows, n):
    cnt = {}
    for r in rows:
        k = (r["dataset"], r["arch"], r["config"], r["method"], r["policy"])
        cnt[k] = cnt.get(k, 0) + 1
    return {k for k, c in cnt.items() if c >= n}


def _save(rows, new, key):
    rows[:] = [r for r in rows if (r["dataset"], r["arch"], r["config"], r["method"],
                                   r["policy"]) != key] + new
    write_csv(EXP, rows, "states.csv")


def _base(ds, arch, c, ci, method, policy):
    return {"dataset": ds, "arch": arch, "config": cname(c), "config_idx": ci,
            "method": method, "policy": policy}


def _abort_on_fail(rows, what):
    for r in rows:
        if r.get("exact_pass") is False:
            raise SystemExit(f"EXACTNESS FAILURE {what} state {r['state_idx']} "
                             f"(depth {r['depth']}): " + ", ".join(f"{k}={v}" for k, v in r.items()
                                                                  if "err" in k or "pass" in k))


STAT_COLS = ("ancestor_depth", "residual_size", "nodes_recomputed", "edges_touched",
             "edges_scanned", "sign_flips")


def part_ours(cfg, settings, traces, cis, n):
    rows = read_rows("states.csv")
    done = _done(rows, n)
    builds = read_rows("build.csv")
    for ds in cfg["datasets"]:
        for ci in cis:                       # trace outermost: generated once
            c = cfg["configs"][ci]
            for arch in cfg["archs"]:
                todo = [p for p in cfg["policies"]
                        if (ds, arch, cname(c), "ours", str(p)) not in done]
                if not todo or not has_ckpt(ds, arch, cfg["layers"], cfg["task"]):
                    continue
                st = settings(ds)
                tr, _, cum_e = traces.get(st, ci)
                model = load_model(ckpt_path(ds, cfg["task"], arch, cfg["layers"]))
                index_mb = get_index(build_cache(model, st.gs.base_view())).nbytes() / 2 ** 20
                _ours_setting(cfg, st, tr, cum_e, model, ds, arch, ci, todo, n, rows, builds,
                              index_mb)


def _ours_setting(cfg, st, tr, cum_e, model, ds, arch, ci, todo, n, rows, builds, index_mb):
    c = cfg["configs"][ci]
    for pol in todo:
        mc = BIG if pol == "all" else int(pol)
        run = lambda **kw: run_trace(model, st.gs, st.schema, tr, "reuse",
                                     max_caches=mc, impl=cfg["impl"], **kw)
        run(n_states=1)                                    # warm-up, discarded
        reps = []
        for r in range(cfg["reps_ours"].get(pol, 1)):   # policies not in the config: 1 rep
            reps.append(run(n_states=n, account=(r == 0),
                            check_every=cfg["check_every"] if r == 0 else 0))
            if r == 0:
                _abort_on_fail(reps[0], f"{ds} {arch} {cname(c)} ours/{pol}")
        build = [x[0]["base_cache_wall_s"] for x in reps]
        walls = np.array([[row["wall_s"] for row in x] for x in reps])
        cums = np.array(build)[:, None] + np.cumsum(walls, axis=1)
        new = []
        for i, row in enumerate(reps[0]):
            new.append({**_base(ds, arch, c, ci, "ours", pol),
                        "state_idx": i, "depth": row["depth"],
                        **{k: row.get(k, "") for k in STAT_COLS},
                        "inc_method": row.get("method", ""),
                        "wall_s": float(np.median(walls[:, i])),
                        "wall_min_s": float(walls[:, i].min()),
                        "wall_max_s": float(walls[:, i].max()),
                        "cum_s": float(np.median(cums[:, i])),
                        "build_s": float(np.median(build)) if i == 0 else "",
                        "retained_bytes": row["retained_bytes"],
                        "query_bytes": row.get("query_bytes", ""),
                        "edges_applied_cum": cum_e[i],
                        "exact_max_abs_err": row.get("exact_max_abs_err", ""),
                        "exact_pass": row.get("exact_pass", "")})
        _save(rows, new, (ds, arch, cname(c), "ours", str(pol)))
        # in place: the caller's list carries the rows of the earlier settings
        builds[:] = [b for b in builds if not (b["dataset"] == ds and b["arch"] == arch
                                               and b["backend"] == "memory")]
        builds.append({"dataset": ds, "arch": arch, "backend": "memory",
                       "build_s": float(np.median(build)), "download_s": 0.0,
                       "tensors_s": 0.0, "index_MB": index_mb})
        write_csv(EXP, builds, "build.csv")
        checks = [r for r in new if r["exact_pass"] != ""]
        print(f"ours {ds:8} {arch} {cname(c):12} keep={pol:>3}: build "
              f"{np.median(build):.3f}s, median/state {np.median(walls):.4f}s, "
              f"cum[{n}] {new[-1]['cum_s']:.2f}s, retained "
              f"{new[-1]['retained_bytes'] / 2**20:.1f} MB, exact "
              f"{sum(r['exact_pass'] is True for r in checks)}/{len(checks)} "
              f"(max {max(r['exact_max_abs_err'] for r in checks):.2g})", flush=True)


def part_scratch(cfg, settings, traces, cis, n):
    rows = read_rows("states.csv")
    done = _done(rows, n)
    for ds in cfg["datasets"]:
        for ci in cis:
            c = cfg["configs"][ci]
            for arch in cfg["archs"]:
                if ((ds, arch, cname(c), "scratch", "") in done
                        or not has_ckpt(ds, arch, cfg["layers"], cfg["task"])):
                    continue
                st = settings(ds)
                tr, _, cum_e = traces.get(st, ci)
                model = load_model(ckpt_path(ds, cfg["task"], arch, cfg["layers"]))
                whatif_forward(model, st.gs, compose(tr.registry, tr.order[0]))  # warm-up
                out = run_trace(model, st.gs, st.schema, tr, "scratch", n_states=n)
                cum = np.cumsum([r["wall_s"] for r in out])
                new = [{**_base(ds, arch, c, ci, "scratch", ""), "state_idx": i,
                        "depth": r["depth"], "nodes_recomputed": r["nodes_recomputed"],
                        "edges_touched": r["edges_touched"], "wall_s": r["wall_s"],
                        "cum_s": float(cum[i]), "edges_applied_cum": cum_e[i]}
                       for i, r in enumerate(out)]
                _save(rows, new, (ds, arch, cname(c), "scratch", ""))
                print(f"scratch {ds:8} {arch} {cname(c):12}: median/state "
                      f"{np.median([r['wall_s'] for r in out]):.4f}s, cum[{n}] {cum[-1]:.2f}s",
                      flush=True)


def _edge_key(ei):
    """Sorted edge multiset of one relation (order-free comparison)."""
    return torch.sort(ei[0] * (int(ei[1].max()) + 1 if ei.numel() else 1) + ei[1]).values


def _tx_bytes(store, tag):
    with store._session() as s:
        rs = s.run("SHOW TRANSACTIONS YIELD transactionId, metaData, allocatedDirectBytes, "
                   "estimatedUsedHeapMemory").data()
    me = [r for r in rs if (r["metaData"] or {}).get("rcf") == tag]
    if not me:
        return "", ""
    return (me[0]["allocatedDirectBytes"] or 0), (me[0]["estimatedUsedHeapMemory"] or 0)


def part_store(cfg, settings, traces, cis, n_cap):
    from src.db.roundtrip import (BoltStore, _apply_delta, _build_tensors,
                                  _download_raw, _edit_pairs)
    sc = cfg["store"]
    ci = sc["config"]
    if ci not in cis:
        return
    n = min(sc["n_states"], n_cap)
    c = cfg["configs"][ci]
    dbc = yaml.safe_load(open(cfg["db_config"]))[sc["backend"]]
    # Neo4j rows keep the original method name; other backends get their own.
    # Undo-memory columns come from Neo4j's SHOW TRANSACTIONS only.
    meth = "store_scratch" if sc["backend"] == "neo4j" else f"store_scratch_{sc['backend']}"
    rows, builds = read_rows("states.csv"), read_rows("build.csv")
    done = _done(rows, n)
    try:
        store = BoltStore(dbc["uri"], dbc.get("user", ""), dbc.get("password", ""),
                          dbc.get("database"))
        with store._session() as s:
            s.run("RETURN 1").consume()
    except Exception as e:  # noqa: BLE001
        print(f"store: {sc['backend']} not reachable ({type(e).__name__}); part skipped. "
              "Start it with `docker compose up -d neo4j`.", flush=True)
        return
    T = time.perf_counter
    for ds in [d for d in sc["datasets"] if d in cfg["datasets"]]:
        archs = [a for a in cfg["archs"] if has_ckpt(ds, a, cfg["layers"], cfg["task"])
                 and (ds, a, cname(c), meth, "") not in done]
        if not archs:
            continue
        st = settings(ds)
        pfx = f"{ds}_"
        tr, _, cum_e = traces.get(st, ci)

        def fetch():
            with store._session() as s:
                tx = s.begin_transaction()
                raw = _download_raw(tx, st.schema, pfx)
                tx.rollback()
            return raw

        fetch()                                                   # warm-up
        t0 = T(); raw = fetch(); dl = T() - t0
        t0 = T(); x, ei = _build_tensors(raw, st.schema); tb = T() - t0
        bad = [et for et in st.gs.edge_types if ei[et].size(1) != st.gs.edge_index[et].size(1)]
        xmax = max(float((x[nt] - st.gs.x[nt]).abs().max()) for nt in st.gs.node_types)
        if bad or xmax > 1e-4:
            raise SystemExit(f"{ds}: store graph != local graph (edges differ on {bad}, "
                             f"feature max diff {xmax:.3g})")
        view = GraphView(x, ei, {})
        for arch in archs:
            model = load_model(ckpt_path(ds, cfg["task"], arch, cfg["layers"]))
            t0 = T(); cache = build_cache(model, view); get_index(cache); bs = T() - t0
            builds = [b for b in builds if not (b["dataset"] == ds and b["arch"] == arch
                                                and b["backend"] == sc["backend"])]
            builds.append({"dataset": ds, "arch": arch, "backend": sc["backend"],
                           "build_s": bs, "download_s": dl, "tensors_s": tb,
                           "index_MB": get_index(cache).nbytes() / 2 ** 20})
            write_csv(EXP, builds, "build.csv")
            del cache

            def one(i, check_it):
                iv = compose(tr.registry, tr.order[i])
                adds, dels, coh = _edit_pairs(st.gs, iv)
                tag = f"{ds}-{arch}-{i}-{time.time_ns()}"
                with store._session() as s:
                    tx = s.begin_transaction(metadata={"rcf": tag})
                    t0 = T(); _apply_delta(tx, adds, dels, coh, pfx); ta = T() - t0
                    u_apply = _tx_bytes(store, tag) if meth == "store_scratch" else ("", "")                   # untimed
                    t0 = T(); r = _download_raw(tx, st.schema, pfx); td = T() - t0
                    t0 = T()
                    xx, ee = _build_tensors(r, st.schema)
                    for nt, d in iv.feat_deltas.items():          # see module docstring
                        xx[nt] = xx[nt] + d
                    tt = T() - t0
                    t0 = T(); h = encode(model, GraphView(xx, ee, {})); tf = T() - t0
                    u_end = _tx_bytes(store, tag) if meth == "store_scratch" else ("", "")                     # untimed
                    t0 = T(); tx.rollback(); trb = T() - t0
                rec = {"apply_s": ta, "download_s": td, "tensors_s": tt, "forward_s": tf,
                       "rollback_s": trb, "wall_s": ta + td + tt + tf + trb,
                       "undo_direct_bytes": u_end[0], "undo_heap_bytes": u_end[1],
                       "undo_bytes_measured": (u_end[0] + u_end[1]) if u_end[0] != "" else "",
                       "undo_bytes_after_apply": (sum(u_apply) if u_apply[0] != "" else ""),
                       "db_edges": sum(t.size(1) for t in ee.values())}
                if check_it:      # gate: the store computed on the same G^Δ (see docstring)
                    ref = st.gs.apply(iv)
                    same = (all(torch.equal(xx[nt], ref.x_dict[nt]) for nt in xx)
                            and all(torch.equal(_edge_key(ee[et]), _edge_key(ref.edge_index_dict[et]))
                                    for et in ee))
                    rep = check(h, whatif_forward(model, st.gs, iv))
                    rec.update(exact_pass=same, out_vs_memory_max_abs_err=rep.max_abs_err,
                               out_vs_memory_pass=rep.passed)
                return rec

            one(0, False)                                         # warm-up
            new, acc = [], 0.0
            for i in range(n):
                rec = one(i, i % cfg["check_every"] == 0)
                acc += rec["wall_s"]
                new.append({**_base(ds, arch, c, ci, meth, ""), "state_idx": i,
                            "depth": len(tr.order[i]), "backend": sc["backend"], **rec,
                            "cum_s": acc, "edges_applied_cum": cum_e[i]})
                _abort_on_fail(new[-1:], f"{ds} {arch} {meth}")
            _save(rows, new, (ds, arch, cname(c), meth, ""))
            ph = {k: statistics.median(r[k] for r in new) for k in
                  ("apply_s", "download_s", "tensors_s", "forward_s", "rollback_s")}
            print(f"store {ds:8} {arch}: per state " +
                  " ".join(f"{k[:-2]}={v:.3f}" for k, v in ph.items()) +
                  f" | cum[{n}] {acc:.1f}s | undo heap median "
                  f"{statistics.median(r['undo_heap_bytes'] or 0 for r in new) / 2**20:.1f} MiB",
                  flush=True)
    store.close()


def machine_info():
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return ""
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
            "ncpu": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "loadavg": os.getloadavg(),
            "top_cpu": sh("sh", "-c", "ps -Ao pcpu,comm -r | head -6")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sessions.yaml")
    ap.add_argument("--parts", nargs="+",
                    default=["generate", "ours", "scratch", "store", "figures"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", type=int, default=None,
                    help="indices into configs (default: all)")
    ap.add_argument("--n-states", type=int, default=None,
                    help="evaluate this prefix of the trace (trace is always n_states long)")
    ap.add_argument("--exp", default="sessions", help="results/<exp>/")
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--policies", nargs="+", default=None,
                    help="override policies (max_caches; 'all' or an integer)")
    ap.add_argument("--store-backend", default=None, help="override store.backend (db.yaml key)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.archs:
        cfg["archs"] = args.archs
    if args.policies:
        cfg["policies"] = [p if p == "all" else int(p) for p in args.policies]
    if args.store_backend:
        cfg["store"]["backend"] = args.store_backend
    global EXP, OUT
    EXP, OUT = args.exp, os.path.join(RESULTS, args.exp)
    os.makedirs(OUT, exist_ok=True)
    set_determinism(cfg["seed"])
    cis = args.configs if args.configs is not None else list(range(len(cfg["configs"])))
    n = min(args.n_states or cfg["n_states"], cfg["n_states"])

    loaded = {}

    def settings(ds):     # one dataset in memory at a time
        if ds not in loaded:
            loaded.clear()
            traces.mem.clear()
            gc.collect()
            loaded[ds] = Setting(ds)
        return loaded[ds]

    traces = Traces(cfg)
    info_start = machine_info()
    tpath = os.path.join(OUT, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}
    steps = {"generate": lambda: part_generate(cfg, settings, traces, cis),
             "ours": lambda: part_ours(cfg, settings, traces, cis, n),
             "scratch": lambda: part_scratch(cfg, settings, traces, cis, n),
             "store": lambda: part_store(cfg, settings, traces, cis, n),
             "figures": lambda: __import__("src.eval.figures", fromlist=["x"])
             .main_sessions(EXP)}
    for p in args.parts:
        t0 = time.perf_counter()
        steps[p]()
        timings[p] = timings.get(p, 0.0) + time.perf_counter() - t0
        print(f"part {p} done in {time.perf_counter() - t0:.1f}s", flush=True)
        json.dump(timings, open(tpath, "w"), indent=2)

    write_manifest(EXP, {**cfg, "n_states_evaluated": n, "configs_run": cis},
                   {d: dataset_card(d) for d in cfg["datasets"]})
    mpath = os.path.join(OUT, "manifest.json")
    m = json.load(open(mpath))
    m["machine"] = {"start": info_start, "end": machine_info()}
    m["traces"] = traces.saved
    json.dump(m, open(mpath, "w"), indent=2, default=str)
    st = read_rows("states.csv")
    n_fail = sum(r.get("exact_pass") == "False" for r in st)
    n_chk = sum(r.get("exact_pass") in ("True", "False") for r in st)
    return summary_line(EXP, n_fail == 0, f"{len(st)} state rows; exactness "
                        f"{n_chk - n_fail}/{n_chk} checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
