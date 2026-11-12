"""Profile incremental_forward: where does the wall time go?

For each setting, torch.profiler (CPU, record_shapes, with_stack) attributes
every aten op's self time to the line of src/incremental/ (or src/data/) that
issued it, and classifies the op as WHOLE-GRAPH (some input has a dimension
equal to a relation's |E_r| — base or edited — or a node type's |V_t|) or
AFFECTED (everything else). cProfile gives the Python-level breakdown.
Whatever the profiler does not see as op time is Python/dispatch overhead.

  python -m src.eval.profile_overhead --impl v1 --out results/overhead_fix/profile_v1.json
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import statistics
import sys
import time
from collections import defaultdict

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile

from src.eval.exp_revision import Setting
from src.eval.runner import ckpt_path
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.models.model import load_model

SETTINGS = [("faers", "rgcn", "topo", 0.05), ("eicu", "rgcn", "topo", 0.05),
            ("faers", "hgat", "feature", 0.05), ("snb", "rgcn", "feature", 0.05),
            ("finbench", "hgat", "topo", 0.30)]
L = 2


def _site(ev):
    """Innermost stack frame inside the repo's src/ as 'src/...py:line', else 'other'."""
    for fr in ev.stack or []:
        if "src/" in fr and "site-packages" not in fr and "(" in fr:
            path, rest = fr.split("(", 1)
            return "src/" + path.split("src/", 1)[1] + ":" + rest.split(")")[0]
    return "other"


_TRACED = ("src/incremental/", "src/models/model.py", "src/data/graph_state.py")


def line_times(fn):
    """Wall time per source line (src/incremental, the model cells, apply):
    the time between consecutive line events is charged to the earlier line,
    so a line that launches a big op gets that op's time."""
    acc, last = defaultdict(float), [None, 0.0]

    def tracer(frame, event, arg):
        f = frame.f_code.co_filename
        if not any(t in f for t in _TRACED):
            return None
        if event == "line":
            now = time.perf_counter()
            if last[0] is not None:
                acc[last[0]] += now - last[1]
            last[0] = "src/" + f.split("/src/", 1)[1] + f":{frame.f_lineno}"
            last[1] = now
        return tracer

    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(None)
    if last[0] is not None:
        acc[last[0]] += time.perf_counter() - last[1]
    return sorted(acc.items(), key=lambda kv: -kv[1])


def profile_setting(ds, arch, kind, frac, impl, seed=42, reps=5):
    st = Setting(ds)
    gs = st.gs
    model = load_model(ckpt_path(ds, "nodelabel", arch, L))
    cache = build_cache(model, gs.base_view())
    _, iv = st.iv(kind, frac, seed)
    run = lambda: incremental_forward(model, gs, cache, iv, impl=impl)
    for _ in range(2):
        run()                                   # warm-up (and lazy index build)
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter(); run(); walls.append(time.perf_counter() - t0)
    wall = statistics.median(walls)
    whatif_forward(model, gs, iv)
    ws = []
    for _ in range(3):
        t0 = time.perf_counter(); whatif_forward(model, gs, iv); ws.append(time.perf_counter() - t0)
    scratch = statistics.median(ws)

    # whole-graph sizes: every |E_r| (base and edited) and every |V_t|
    big = set(st.schema.num_nodes.values())
    for et, ei in gs.edge_index.items():
        e = ei.size(1)
        big |= {e, e - iv.del_cols.get(et, torch.empty(0)).numel()
                + (iv.add_edges[et].size(1) if et in iv.add_edges else 0),
                e + (iv.add_edges[et].size(1) if et in iv.add_edges else 0)}
    big.discard(0)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True,
                 experimental_config=_ExperimentalConfig(verbose=True)) as prof:
        res = run()
    by_site = defaultdict(lambda: {"global_us": 0.0, "local_us": 0.0, "ops": defaultdict(float)})
    tot = {"global_us": 0.0, "local_us": 0.0}
    for ev in prof.events():
        if not ev.name.startswith("aten::") or ev.cpu_parent is not None and \
                ev.cpu_parent.name.startswith("aten::"):
            continue                            # top-level aten ops only (no double count)
        dims = {d for shp in (ev.input_shapes or []) for d in (shp or [])}
        cls = "global_us" if dims & big else "local_us"
        us = ev.cpu_time_total
        site = _site(ev)
        by_site[site][cls] += us
        by_site[site]["ops"][ev.name] += us
        tot[cls] += us
    prof_wall_us = max(ev.time_range.end for ev in prof.events()) - \
        min(ev.time_range.start for ev in prof.events())

    lines = [{"line": k, "ms": v * 1e3} for k, v in line_times(run)[:15]]

    pr = cProfile.Profile()
    pr.enable(); run(); pr.disable()
    ps = pstats.Stats(pr).sort_stats("tottime")
    py_top = []
    for (f, line, fn), (cc, nc, tt, ct, _) in sorted(ps.stats.items(),
                                                     key=lambda kv: -kv[1][2])[:12]:
        py_top.append({"func": f"{os.path.relpath(f) if f.startswith('/') else f}:{line}({fn})",
                       "calls": nc, "tottime_ms": tt * 1e3})

    sites = sorted(({"site": s, **{k: v for k, v in d.items() if k != "ops"},
                     "top_ops": sorted(d["ops"].items(), key=lambda kv: -kv[1])[:4]}
                    for s, d in by_site.items()),
                   key=lambda r: -(r["global_us"] + r["local_us"]))
    return {"setting": f"{ds} {arch} {kind} {int(frac * 100)}% L{L}", "impl": impl,
            "wall_ms": wall * 1e3, "scratch_ms": scratch * 1e3,
            "profiled_span_ms": prof_wall_us / 1e3,
            "op_global_ms": tot["global_us"] / 1e3, "op_local_ms": tot["local_us"] / 1e3,
            "stats": {k: v for k, v in res.stats.items() if k != "dirty_frontier"},
            "lines": lines, "sites": sites[:15], "python_top": py_top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_determinism(42)
    out = [profile_setting(*s, impl=args.impl) for s in SETTINGS]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for r in out:
        span = r["profiled_span_ms"]
        print(f"{r['setting']:32} {r['impl']} wall={r['wall_ms']:8.1f}ms scratch={r['scratch_ms']:8.1f}ms "
              f"| profiled span {span:8.1f}ms: whole-graph ops {r['op_global_ms']/span:5.1%} "
              f"affected ops {r['op_local_ms']/span:5.1%} "
              f"other(python/dispatch) {1-(r['op_global_ms']+r['op_local_ms'])/span:5.1%}")


if __name__ == "__main__":
    main()
