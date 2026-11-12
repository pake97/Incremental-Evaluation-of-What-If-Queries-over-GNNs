"""Tables T-a..T-f for results/overhead_fix/README.md: v1 (results/revision)
vs v2 (results/revision_v2), measured medians only. Rows failing the
exactness gate are dropped before anything is tabulated.

  python -m src.eval.report_overhead_fix
"""
from __future__ import annotations

import os

import pandas as pd

from src.paths import RESULTS

OUT = os.path.join(RESULTS, "overhead_fix")
KEY = ["dataset", "arch", "kind", "frac", "layers"]


def md(df):
    f = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def R(exp, name):
    p = os.path.join(RESULTS, exp, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    # v1 timed under the same machine conditions as v2 if available (see README)
    v1src = "revision_v1_same" if R("revision_v1_same", "latency.csv") is not None \
        else "revision"
    v1, v2 = R(v1src, "latency.csv"), R("revision_v2", "latency.csv")
    v1 = v1[v1.set_index(KEY).index.isin(v2.set_index(KEY).index)]
    n_fail = int((~v2.passed).sum())
    v1, v2 = v1[v1.passed], v2[v2.passed]
    m = v2.merge(v1[KEY + ["speedup_measured", "ours_median_ms", "scratch_median_ms"]],
                 on=KEY, suffixes=("_v2", "_v1"))
    out = [f"v1 timings from results/{v1src}/latency.csv ({len(v1)} settings).",
           f"Exactness (Part D, v2): {len(v2)} rows passed, {n_fail} failed; "
           f"max |v2 - scratch| = {v2.max_abs_err.max():.3g}, "
           f"max |v2 - v1| = {v2.vs_v1_max_abs_err.max():.3g}; "
           f"v1/v2 stats equal in {int(v2.v1_stats_equal.sum())}/{len(v2)} rows."]

    a = m[(m.layers == 2) & m.frac.isin([0.05, 0.30])][
        ["dataset", "arch", "kind", "frac", "closure_frac", "speedup_measured_v1",
         "speedup_measured_v2", "ours_median_ms_v1", "ours_median_ms_v2",
         "scratch_median_ms_v2"]].sort_values(["dataset", "arch", "kind", "frac"])
    out.append("### T-a — measured speedup at L=2 (5% and 30%), v1 vs v2\n\n" + md(a))

    slow1 = v1[v1.ours_median_ms > v1.scratch_median_ms]
    slow2 = v2[v2.ours_median_ms > v2.scratch_median_ms]
    cnt = pd.DataFrame({"layers": [1, 2, 3, 4],
                        "v1_slower": [int((slow1.layers == L).sum()) for L in range(1, 5)],
                        "v2_slower": [int((slow2.layers == L).sum()) for L in range(1, 5)],
                        "settings": [int((v2.layers == L).sum()) for L in range(1, 5)]})
    out.append(f"### T-b — settings still slower than from-scratch: v1 {len(slow1)}, "
               f"v2 {len(slow2)} of {len(v2)}\n\n" + md(cnt) + "\n\nv2 list:\n\n" +
               md(slow2[KEY + ["inc_method", "scratch_median_ms", "ours_median_ms",
                               "speedup_measured", "closure_frac"]]
                  .sort_values(["dataset", "arch", "layers", "kind", "frac"])))

    c = m[m.closure_frac >= 0.95]
    cs = c.groupby(["dataset", "arch", "kind"]).agg(
        n=("frac", "size"), v1_median=("speedup_measured_v1", "median"),
        v2_median=("speedup_measured_v2", "median"), v2_min=("speedup_measured_v2", "min"),
        v2_max=("speedup_measured_v2", "max")).reset_index()
    out.append(f"### T-c — speedup at closure_frac >= 0.95 ({len(c)} settings; overall median "
               f"v1 {c.speedup_measured_v1.median():.3g}, v2 {c.speedup_measured_v2.median():.3g})"
               "\n\n" + md(cs))

    v2 = v2.assign(r_v1=v2.v1_edges_scanned / v2.edges_touched,
                   r_v2=v2.edges_scanned / v2.edges_touched)
    d = v2.groupby(["dataset", "arch", "kind"]).agg(
        v1_ratio_median=("r_v1", "median"), v2_ratio_median=("r_v2", "median"),
        v2_ratio_max=("r_v2", "max")).reset_index()
    out.append("### T-d — edges_scanned / edges_touched (median over L and fracs)\n\n"
               f"Overall median: v1 {v2.r_v1.median():.3g}, v2 {v2.r_v2.median():.3g}.\n\n" + md(d))

    q1, q2 = R("revision", "query_memory.csv"), R("revision_v2", "query_memory.csv")
    keep = ["dataset", "arch", "kind", "retained_hz_MB"]
    q = q1[(q1.method == "ours") & (q1.frac == 0.10)][keep].merge(
        q2[(q2.method == "ours") & (q2.frac == 0.10)][keep + ["retained_graph_MB"]],
        on=["dataset", "arch", "kind"], suffixes=("_v1", "_v2"))
    s2 = R("revision_v2", "cache_size_summary.csv")
    s2 = s2[s2.L == 2][["dataset", "arch", "cache_MB", "index_MB"]]
    s1 = R("revision", "cache_size_summary.csv")
    s1 = s1[s1.L == 2][["dataset", "arch", "extras_MB"]].rename(columns={"extras_MB": "v1_extras_MB"})
    b = R("revision_v2", "build_cost.csv")
    bi = b[(b.phase == "index_build") & (b.status == "ok")]
    bim = bi[bi.backend == "memory"].groupby("dataset").agg(
        index_build_first_s=("first_s", "first"), index_build_warm_s=("warm_median_s", "first"))
    bc = b[(b.backend == "memory") & (b.phase == "build_cache")][["dataset", "arch", "warm_median_s"]] \
        .rename(columns={"warm_median_s": "build_cache_s"})
    e = s2.merge(s1, on=["dataset", "arch"]).merge(bim.reset_index(), on="dataset") \
        .merge(bc, on=["dataset", "arch"])
    out.append("### T-e — retained state of one query (10%, L=2) and the session index\n\n"
               "Retained h/z bytes of the derived state (v1: cloned node-type tensors; "
               "v2: row overlays + the dense output layer); retained_graph_MB_v2 = the "
               "intervention it references (edit list + dense feature delta):\n\n" + md(q) +
               "\n\nIndex (v2) vs lazy extras (v1), L=2:\n\n" + md(e))
    st = bi[bi.backend != "memory"][["dataset", "backend", "first_s", "warm_median_s"]]
    if len(st):
        out.append("Index build on DB-built tensors (added to the store one-time cost):\n\n" + md(st))

    cf = R("revision_v2", "compiled.csv")
    if cf is not None:
        cf = cf[cf.passed]
        f = v2.merge(cf[KEY + ["scratch_compiled_median_ms"]], on=KEY)
        f["v2_vs_compiled"] = f.scratch_compiled_median_ms / f.ours_median_ms
        f["compile_gain"] = f.scratch_median_ms / f.scratch_compiled_median_ms
        t = f[(f.layers == 2) & f.frac.isin([0.05, 0.30])][
            ["dataset", "arch", "kind", "frac", "scratch_median_ms", "scratch_compiled_median_ms",
             "ours_median_ms", "compile_gain", "v2_vs_compiled"]] \
            .sort_values(["dataset", "arch", "kind", "frac"])
        out.append(f"### T-f — v2 vs compiled from-scratch (REFERENCE; not plotted)\n\n"
                   f"v2 faster than compiled from-scratch in {int((f.v2_vs_compiled > 1).sum())} "
                   f"of {len(f)} settings; median compile gain on from-scratch "
                   f"{f.compile_gain.median():.3g}x. L=2, 5%/30%:\n\n" + md(t))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n\n".join(out) + "\n")
    print("wrote", os.path.join(OUT, "tables.md"))


if __name__ == "__main__":
    main()
