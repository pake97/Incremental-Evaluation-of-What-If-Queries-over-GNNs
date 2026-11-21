"""Supplementary-material figures for the session experiment (no new measurements):
  supp_sessions_store.pdf  FinBench / SNB / FAERS, RGCN, first 100 states, graph in a store:
                           cumulative time, memory held, per-query latency vs path depth
  supp_sessions_deep.pdf   deep tree (b2_d50_bt0.1), in memory, 4 datasets, RGCN:
                           per-query latency vs path depth
Series follow the definitions of figures.fig_sessions_eicu (same styles and labels).
  python scripts/supp_session_figures.py <output dir>
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.lines as ml
import src.eval.figures as F

OUT = sys.argv[1]; MB = 2 ** 20; N = 100
POLS = ("all", "64", "32", "16", "8", "4", "1")
NAME = {"eicu": "eICU", "finbench": "FinBench", "snb": "SNB", "faers": "FAERS"}

def load(exp):
    df = pd.read_csv(f"results/{exp}/states.csv", low_memory=False)
    assert not (df.exact_pass == False).any(), exp
    df["k"] = df.state_idx + 1
    return df, pd.read_csv(f"results/{exp}/build.csv")

sz = pd.read_csv("results/revision_v2/store_size.csv")
gsz = pd.read_csv("results/sessions/graph_size.csv")

def store_fig():
    runs = [("finbench", "sessions"), ("snb", "sessions_snbfaers"), ("faers", "sessions_snbfaers")]
    fig, axes = plt.subplots(len(runs), 3, figsize=(10.5, 7.2), squeeze=False)
    for i, (d, exp) in enumerate(runs):
        df, b = load(exp)
        sub = df[(df.config_idx == 0) & (df.dataset == d) & (df.arch == "rgcn") & (df.k <= N)]
        pols = [p for p in POLS if p in set(sub[sub.method == "ours"].policy.astype(str))]
        bb = b[(b.dataset == d) & (b.arch == "rgcn") & (b.backend == "neo4j")].iloc[0]
        one = float(bb.download_s + bb.tensors_s + bb.build_s)
        g = gsz[gsz.dataset == d].iloc[0]
        ix = float(b[(b.dataset == d) & (b.arch == "rgcn") & (b.backend == "memory")].index_MB.iloc[0])
        cache0 = sub[(sub.method == "ours") & (sub.policy.astype(str) == "1")].retained_bytes.min() / MB
        neo = sz[(sz.dataset == d) & (sz.backend == "neo4j")].iloc[0]
        peak = neo.store_MB + g.x_MB + g.edges_MB + cache0
        k = np.arange(N + 1)
        aT, aM, aD = axes[i]
        for p in pols:
            o = sub[(sub.method == "ours") & (sub.policy.astype(str) == p)].sort_values("k")
            c, ls, _ = F._SESS_LINE[p]
            aT.plot(k, np.r_[one, one + o.cum_s.values - float(o.build_s.iloc[0])], ls=ls, color=c, lw=1.8)
            aM.plot(k, np.r_[peak, g.x_MB + o.retained_bytes.values / MB], ls=ls, color=c, lw=1.8)
            med = o.groupby("depth").wall_s.median() * 1e3
            aD.plot(med.index, med.values, ls=ls, color=c, lw=1.8, marker=".", ms=4)
        ref = sub[sub.method == "store_scratch"].sort_values("k")
        for be, meth in (("neo4j", "store_scratch"), ("memgraph", "store_scratch_memgraph")):
            st = sub[sub.method == meth].sort_values("k")
            c, ls, _ = F._SESS_LINE[be]
            aT.plot(k[1:], st.cum_s.values, ls=ls, color=c, lw=1.8)
            r = sz[(sz.dataset == d) & (sz.backend == be)].iloc[0]
            aM.plot(k, r.store_MB + g.x_MB + g.edges_MB + cache0 - ix
                    + np.r_[0.0, r.log_bytes / r.edges * ref.edges_applied_cum.values / MB],
                    ls=ls, color=c, lw=1.8)
            med = st.groupby("depth").wall_s.median() * 1e3
            aD.plot(med.index, med.values, ls=ls, color=c, lw=1.8, marker=".", ms=4)
        for a in (aT, aM, aD):
            a.set_yscale("log")
        aT.set_ylabel(f"{NAME[d]}\ncumulative time (s)"); aM.set_ylabel("memory held (MB)")
        aD.set_ylabel("per-query latency (ms)")
        if i == len(runs) - 1:
            aT.set_xlabel("states visited k"); aM.set_xlabel("states visited k")
            aD.set_xlabel("path depth |D| from the base")
    fig.legend(handles=[F._sess_handle(p) for p in (*POLS, "neo4j", "memgraph")], frameon=False,
               fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(os.path.join(OUT, "supp_sessions_store.pdf")); plt.close(fig)

def deep_fig():
    df, _ = load("sessions")
    ds = ["eicu", "finbench", "snb", "faers"]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6), squeeze=False)
    for a, d in zip(axes[0], ds):
        sub = df[(df.config == "b2_d50_bt0.1") & (df.dataset == d) & (df.arch == "rgcn")]
        for p in ("all", "16", "1"):
            c, ls, _ = F._SESS_LINE[p]
            med = sub[(sub.method == "ours") & (sub.policy.astype(str) == p)].groupby("depth").wall_s.median() * 1e3
            a.plot(med.index, med.values, ls=ls, color=c, lw=1.6)
        med = sub[sub.method == "scratch"].groupby("depth").wall_s.median() * 1e3
        a.plot(med.index, med.values, "-", color="#333333", lw=1.6)
        a.set_yscale("log"); a.set_title(NAME[d]); a.set_xlabel("path depth |D| from the base")
    axes[0][0].set_ylabel("per-query latency (ms)")
    h = [F._sess_handle(p) for p in ("all", "16", "1")]
    h.append(ml.Line2D([], [], color="#333333", ls="-", lw=2, label="naive-GNN in memory"))
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(os.path.join(OUT, "supp_sessions_deep.pdf")); plt.close(fig)

store_fig(); deep_fig(); print("ok")
