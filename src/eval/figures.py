"""Generate the paper figures from results/ CSVs.

Static matplotlib figures for the paper; skips any figure whose experiment
hasn't produced results yet. Follows the dataviz method: form by the data's
job, one axis per chart, fixed categorical hue order (validated palette), thin
marks, recessive grids, direct labels where they earn their place.

  python -m src.eval.figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.paths import FIGURES, RESULTS

# validated categorical palette (dataviz reference instance, light mode),
# assigned in fixed order: scratch, incremental, reuse / rgcn, hgat
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
GRID = dict(color="#dddddd", linewidth=0.6)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID["color"],
    "grid.linewidth": GRID["linewidth"], "axes.axisbelow": True,
    "font.size": 9,
})


def _load(exp, name="results.csv"):
    path = os.path.join(RESULTS, exp, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fig_overhead():
    """Proxy (algorithmic) vs wall-clock (Python prototype), and why they differ:
    the framework tax that tiny kernels pay and the bulk forward does not."""
    df = _load("overhead")
    if df is None:
        return None
    df = df[df.kind == "feature"]
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(3.1 * len(datasets), 5.2), squeeze=False)
    for j, ds in enumerate(datasets):
        sub = df[(df.dataset == ds) & (df.arch == "rgcn")].sort_values("size")
        # (top) proxy vs wall speedup — the gap is the overhead tax
        ax = axes[0][j]
        ax.plot(sub["size"], sub.proxy_speedup_edges, "o-", color=C[0],
                linewidth=2, markersize=4, label="algorithmic (edges avoided)")
        ax.plot(sub["size"], sub.wall_speedup, "s--", color=C[2],
                linewidth=2, markersize=4, label="wall-clock (Python prototype)")
        ax.axhline(1.0, color="#999999", linewidth=1, linestyle=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ds)
        if j == 0:
            ax.set_ylabel("speedup vs from-scratch")
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        # (bottom) per-edge wall cost: incremental amortizes toward the flat
        # compiled-bulk rate; the gap at small Δ is pure framework overhead
        ax = axes[1][j]
        ax.plot(sub["size"], sub.inc_wall_per_edge_us, "o-", color=C[0],
                linewidth=2, markersize=4, label="INQ-RGCN")
        ax.plot(sub["size"], sub.scratch_wall_per_edge_us, "s--", color=C[1],
                linewidth=2, markersize=4, label="from-scratch (compiled bulk)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("intervention size (cohort nodes)")
        if j == 0:
            ax.set_ylabel("wall-clock per edge-message (µs)")
            ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Wall-clock understates the method: the algorithmic advantage "
                 "(top, blue) is\nlargest exactly where tiny-kernel overhead (bottom) "
                 "hides it — RGCN feature Δ", y=1.06, fontsize=9)
    fig.tight_layout()
    return fig, "overhead_proxy_vs_wall.png"


def _overhead_kinds(df):
    return [k for k in ["feature", "topo"] if k in set(df.kind.unique())]


# fixed color per (arch, method) so every grid panel reads the same way
_GRID_STYLE = {
    ("rgcn", "incremental"): (C[0], "o-", "INQ-RGCN"),
    ("rgcn", "scratch"):     (C[0], "o--", "from-scratch (RGCN)"),
    ("hgat", "incremental"): (C[5], "s-", "INQ-hGAT"),
    ("hgat", "scratch"):     (C[5], "s--", "from-scratch (HGAT)"),
}
_GRID_KINDS = ["feature", "edge_add", "edge_del", "edge_swap"]
_GRID_KIND_LABEL = {"feature": "feature", "edge_add": "edge add",
                    "edge_del": "edge remove", "edge_swap": "edge swap"}


def _fig_grid(dataset, value, title, ylabel, fname, logy=True):
    """rows = intervention kind, cols = size; x = layers L; four lines
    (incremental & from-scratch, for RGCN & HGAT). `value` returns (inc, scr)
    series in the plotted unit for a per-(arch) sub-frame."""
    df = _load("grid")
    if df is None:
        return None
    d = df[df.dataset == dataset]
    if d.empty:
        return None
    kinds = [k for k in _GRID_KINDS if k in set(d.kind.unique())]
    sizes = sorted(d["size"].unique())
    fig, axes = plt.subplots(len(kinds), len(sizes),
                             figsize=(3.0 * len(sizes), 2.3 * len(kinds)),
                             squeeze=False)
    for i, kind in enumerate(kinds):
        for j, size in enumerate(sizes):
            ax = axes[i][j]
            for arch in ("rgcn", "hgat"):
                s = d[(d.kind == kind) & (d["size"] == size) & (d.arch == arch)]
                s = s.sort_values("layers")
                if s.empty:
                    continue
                inc, scr = value(s)
                col, _, lbl = _GRID_STYLE[(arch, "incremental")]
                ax.plot(s.layers, inc, "o-", color=col, linewidth=2, markersize=4,
                        label=lbl)
                col2, _, lbl2 = _GRID_STYLE[(arch, "scratch")]
                ax.plot(s.layers, scr, "o--", color=col2, linewidth=1.3,
                        markersize=3, alpha=0.6, label=lbl2)
            ax.set_xticks([1, 2, 3, 4])
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"size = {size}")
            if i == len(kinds) - 1:
                ax.set_xlabel("layers L")
            if j == 0:
                ax.set_ylabel(f"{_GRID_KIND_LABEL[kind]} Δ\n{ylabel}", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6)
    fig.suptitle(f"{title} — {dataset}", y=1.005, fontsize=10)
    fig.tight_layout()
    return fig, fname


def fig_grid_latency_all():
    figs = []
    for ds in ("finbench", "faers", "eicu", "snb"):
        out = _fig_grid(
            ds,
            value=lambda s: (s.wall_incremental_best_s * 1e3, s.wall_scratch_s * 1e3),
            title="Single-query latency: INQ (best impl.) vs from-scratch",
            ylabel="latency (ms)",
            fname=f"grid_latency_{ds}.png")
        if out:
            figs.append(out)
    return figs


def fig_grid_operations_all():
    figs = []
    for ds in ("finbench", "faers", "eicu", "snb"):
        out = _fig_grid(
            ds,
            value=lambda s: (s.inc_edges, s.scratch_edges),
            title="Single-query operations: INQ vs from-scratch",
            ylabel="edge-messages",
            fname=f"grid_operations_{ds}.png")
        if out:
            figs.append(out)
    return figs


def _fig_grid_signflips_one(dataset):
    """Feature Δ sign-flips (activations whose ReLU state the intervention
    changes) vs depth L and vs size, for BOTH architectures. rows = arch,
    cols = {vs L, vs size}."""
    df = _load("grid")
    if df is None or "sign_flips" not in df.columns:
        return None
    d = df[(df.dataset == dataset) & (df.kind == "feature")].copy()
    d = d[d.sign_flips.astype(str) != ""]
    if d.empty:
        return None
    d["sign_flips"] = d.sign_flips.astype(float)
    archs = [a for a in ("rgcn", "hgat") if a in set(d.arch.unique())]
    sizes = sorted(d["size"].unique())
    layers = sorted(d.layers.unique())
    arch_lbl = {"rgcn": "RGCN (delta path)", "hgat": "HGAT (closure path)"}
    fig, axes = plt.subplots(len(archs), 2, figsize=(8.0, 2.9 * len(archs)),
                             squeeze=False)
    for r, arch in enumerate(archs):
        da = d[d.arch == arch]
        # (left) vs L, one line per size
        ax = axes[r][0]
        for k, size in enumerate(sizes):
            s = da[da["size"] == size].sort_values("layers")
            ax.plot(s.layers, s.sign_flips.clip(lower=0.5), "o-",
                    color=C[k % len(C)], linewidth=2, markersize=5,
                    label=f"size {size}")
        ax.set_xticks([1, 2, 3, 4]); ax.set_yscale("log")
        ax.set_ylabel(f"{arch_lbl[arch]}\nsign-flips", fontsize=8)
        if r == 0:
            ax.set_title("vs depth L")
        if r == len(archs) - 1:
            ax.set_xlabel("layers L")
        ax.legend(frameon=False, fontsize=7)
        # (right) vs size, one line per L
        ax = axes[r][1]
        for L in layers:
            s = da[da.layers == L].sort_values("size")
            ax.plot(s["size"], s.sign_flips.clip(lower=0.5), "o-",
                    color=C[(L - 1) % len(C)], linewidth=2, markersize=5,
                    label=f"L={L}")
        ax.set_xscale("log"); ax.set_yscale("log")
        if r == 0:
            ax.set_title("vs cohort size")
        if r == len(archs) - 1:
            ax.set_xlabel("feature Δ cohort size")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Feature-Δ sign-flips vs depth and size — {dataset} "
                 f"(clipped at 0.5 for log scale)", y=1.01, fontsize=9.5)
    fig.tight_layout()
    return fig, f"grid_signflips_{dataset}.png"


def fig_grid_signflips_all():
    figs = []
    for ds in ("finbench", "faers", "eicu", "snb"):
        out = _fig_grid_signflips_one(ds)
        if out:
            figs.append(out)
    return figs


def fig_signflips_combined():
    """One figure: columns = datasets, rows = {vs depth L, vs cohort size},
    both architectures overlaid — RGCN solid + circles, HGAT dashed + triangles;
    color encodes size (top row) / depth L (bottom row)."""
    import matplotlib.lines as mlines

    df = _load("grid")
    if df is None or "sign_flips" not in df.columns:
        return None
    d = df[df.kind == "feature"].copy()
    d = d[d.sign_flips.astype(str) != ""]
    if d.empty:
        return None
    d["sign_flips"] = d.sign_flips.astype(float)
    datasets = [x for x in ("finbench", "faers", "eicu", "snb")
                if x in set(d.dataset.unique())]
    sizes = sorted(d["size"].unique())
    layers = sorted(d.layers.unique())
    ARCH = {"rgcn": ("-", "o"), "hgat": ("--", "^")}

    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(3.0 * len(datasets), 6.0), squeeze=False)
    for j, ds in enumerate(datasets):
        dd = d[d.dataset == ds]
        # top: vs depth L, colour per size
        ax = axes[0][j]
        for k, size in enumerate(sizes):
            for arch, (ls, mk) in ARCH.items():
                s = dd[(dd["size"] == size) & (dd.arch == arch)].sort_values("layers")
                if s.empty:
                    continue
                ax.plot(s.layers, s.sign_flips.clip(lower=0.5), ls, marker=mk,
                        color=C[k % len(C)], linewidth=1.8, markersize=5)
        ax.set_xticks([1, 2, 3, 4]); ax.set_yscale("log")
        ax.set_title(ds)
        if j == 0:
            ax.set_ylabel("sign-flips\n(vs depth L)", fontsize=9)
        ax.set_xlabel("layers L", fontsize=8)
        # bottom: vs size, colour per L
        ax = axes[1][j]
        for L in layers:
            for arch, (ls, mk) in ARCH.items():
                s = dd[(dd.layers == L) & (dd.arch == arch)].sort_values("size")
                if s.empty:
                    continue
                ax.plot(s["size"], s.sign_flips.clip(lower=0.5), ls, marker=mk,
                        color=C[(L - 1) % len(C)], linewidth=1.8, markersize=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        if j == 0:
            ax.set_ylabel("sign-flips\n(vs cohort size)", fontsize=9)
        ax.set_xlabel("cohort size", fontsize=8)

    # colour legends (size / L) on the first column of each row
    axes[0][0].legend(handles=[mlines.Line2D([], [], color=C[k % len(C)], lw=3,
                      label=f"size {s}") for k, s in enumerate(sizes)],
                      frameon=False, fontsize=7, loc="lower right")
    axes[1][0].legend(handles=[mlines.Line2D([], [], color=C[(L - 1) % len(C)], lw=3,
                      label=f"L={L}") for L in layers],
                      frameon=False, fontsize=7, loc="lower right")
    # architecture legend (texture + marker), figure-level
    arch_handles = [
        mlines.Line2D([], [], color="#444", ls="-", marker="o", label="RGCN (delta path)"),
        mlines.Line2D([], [], color="#444", ls="--", marker="^", label="HGAT (closure path)")]
    fig.legend(handles=arch_handles, frameon=False, fontsize=8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Feature-Δ sign-flips grow with depth and size, both architectures "
                 "(clipped at 0.5 for log scale)", y=1.05, fontsize=10)
    fig.tight_layout()
    return fig, "signflips_combined.png"


def fig_single_query_best():
    """Projected BEST-implementation latency: the method's measured work
    (edge-messages touched) costed at the measured per-edge rate of the compiled
    bulk forward — i.e. overhead_ratio -> 1, what a fused kernel would achieve.
    This is a roofline projection from measured quantities, NOT the Python
    prototype's wall-clock (that is single_query_latency.png)."""
    df = _load("overhead")
    if df is None:
        return None
    df = df[df.arch == "rgcn"]
    datasets = sorted(df.dataset.unique())
    kinds = _overhead_kinds(df)
    fig, axes = plt.subplots(len(kinds), len(datasets),
                             figsize=(3.2 * len(datasets), 2.7 * len(kinds)),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for i, kind in enumerate(kinds):
            ax = axes[i][j]
            s = df[(df.dataset == ds) & (df.kind == kind)].sort_values("size")
            rate = s.scratch_wall_per_edge_us / 1e3  # ms per edge-message (bulk kernel)
            inc_best_ms = s.inc_edges * rate
            scratch_ms = s.wall_scratch_s * 1e3
            proto_ms = s.wall_incremental_s * 1e3
            ax.plot(s["size"], inc_best_ms, "o-", color=C[0], linewidth=2,
                    markersize=4, label="INQ-RGCN (projected best)")
            ax.plot(s["size"], scratch_ms, "s--", color=C[1], linewidth=1.4,
                    markersize=3, label="from-scratch (measured)")
            ax.plot(s["size"], proto_ms, "^:", color=C[2], linewidth=1,
                    markersize=3, alpha=0.5, label="INQ-RGCN (Python prototype)")
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(ds)
            if i == len(kinds) - 1:
                ax.set_xlabel("intervention size (%s)"
                              % ("cohort nodes" if kind == "feature" else "edges"))
            if j == 0:
                ax.set_ylabel(f"{kind} Δ — latency (ms)", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Single what-if query — PROJECTED best implementation "
                 "(work × compiled-bulk per-edge rate);\nfrom-scratch measured, "
                 "Python prototype shown faded for reference — RGCN", y=1.04, fontsize=8.5)
    fig.tight_layout()
    return fig, "single_query_latency_best.png"


def fig_single_query_operations():
    """The same comparison in the implementation-INDEPENDENT currency: edge-
    messages aggregated (the work proxy), incremental vs from-scratch."""
    df = _load("overhead")
    if df is None:
        return None
    df = df[df.arch == "rgcn"]
    datasets = sorted(df.dataset.unique())
    kinds = _overhead_kinds(df)
    fig, axes = plt.subplots(len(kinds), len(datasets),
                             figsize=(3.2 * len(datasets), 2.7 * len(kinds)),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for i, kind in enumerate(kinds):
            ax = axes[i][j]
            s = df[(df.dataset == ds) & (df.kind == kind)].sort_values("size")
            ax.plot(s["size"], s.inc_edges, "o-", color=C[0], linewidth=2,
                    markersize=4, label="INQ-RGCN")
            ax.plot(s["size"], s.scratch_edges, "s--", color=C[1], linewidth=1.4,
                    markersize=3, label="from-scratch (all edges × L)")
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(ds)
            if i == len(kinds) - 1:
                ax.set_xlabel("intervention size (%s)"
                              % ("cohort nodes" if kind == "feature" else "edges"))
            if j == 0:
                ax.set_ylabel(f"{kind} Δ — edge-messages\naggregated", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Single what-if query — OPERATIONS (edge-messages aggregated): "
                 "the implementation-independent work measure — RGCN", y=1.04, fontsize=8.5)
    fig.tight_layout()
    return fig, "single_query_operations.png"


def fig_sign_flips(sample="finbench"):
    """Sparse ReLU correction: how many activation sign-flips the low-rank delta
    path must correct, vs the nodes it recomputes. One dataset (sample)."""
    df = _load("overhead")
    if df is None or "sign_flips" not in df.columns:
        return None
    df = df[(df.dataset == sample) & (df.kind == "feature") & (df.arch == "rgcn")].copy()
    df = df[df.sign_flips.astype(str) != ""]
    if df.empty:
        return None
    df["sign_flips"] = df.sign_flips.astype(float)
    df = df.sort_values("size")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axes[0]
    ax.plot(df["size"], df.sign_flips, "o-", color=C[5], linewidth=2,
            markersize=5, label="sign-flips (nonlinear correction)")
    ax.plot(df["size"], df.inc_nodes, "s--", color=C[0], linewidth=1.6,
            markersize=4, label="node-rows recomputed")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("feature Δ cohort size")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Sign-flips vs recomputed rows")
    ax = axes[1]
    frac = df.sign_flips / df.inc_nodes.clip(lower=1)
    ax.plot(df["size"], frac, "o-", color=C[5], linewidth=2, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("feature Δ cohort size")
    ax.set_ylabel("sign-flips per recomputed row")
    ax.set_title("Correction sparsity")
    ax.set_ylim(bottom=0)
    fig.suptitle(f"Low-rank delta path: the nonlinear correction is sparse "
                 f"({sample}, RGCN feature Δ)", y=1.03, fontsize=9)
    fig.tight_layout()
    return fig, "sign_flips.png"


_BASELINE_STYLE = {
    "from_scratch": (C[3], "x:", "from-scratch"),
    "ripple":       (C[2], "s-", "Ripple"),
    "neutronrt":    (C[1], "D-", "NeutronRT"),
    "ours":         (C[0], "o-", "INQ"),
}
_BASELINE_ORDER = ["from_scratch", "ripple", "neutronrt", "ours"]


def _fig_baselines(value, ylabel, fname_tmpl, logy=True, df_override=None):
    """Per dataset: one figure, rows = experiment kind {feature, topo}, cols =
    architecture, x = intervention size, four method lines. `value(s)` picks the
    plotted quantity from the per-method sub-frame."""
    df = df_override if df_override is not None else _load("baselines")
    if df is None:
        return []
    figs = []
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        archs = sorted(d.arch.unique())
        kinds = ["feature", "topo"]
        fig, axes = plt.subplots(len(kinds), len(archs),
                                 figsize=(3.4 * len(archs), 2.6 * len(kinds)),
                                 squeeze=False)
        for i, kind in enumerate(kinds):
            for j, arch in enumerate(archs):
                ax = axes[i][j]
                for nm in _BASELINE_ORDER:
                    s = d[(d.kind == kind) & (d.arch == arch) & (d.method == nm)]
                    s = s.sort_values("size")
                    if s.empty:
                        continue
                    col, style, lbl = _BASELINE_STYLE[nm]
                    ax.plot(s["size"], value(s), style, color=col, linewidth=2,
                            markersize=5, label=lbl,
                            alpha=0.9 if nm != "from_scratch" else 0.6)
                ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                if i == 0:
                    ax.set_title(arch.upper())
                if i == len(kinds) - 1:
                    ax.set_xlabel("intervention size (%s)"
                                  % ("cohort nodes" if kind == "feature" else "edges"))
                if j == 0:
                    ax.set_ylabel(f"{kind} Δ\n{ylabel}", fontsize=8)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"Baselines — {ylabel} — {ds}", y=1.01, fontsize=10)
        fig.tight_layout()
        figs.append((fig, fname_tmpl.format(ds=ds)))
    return figs


def fig_baselines_operations():
    return _fig_baselines(lambda s: s.edges, "edge-messages (ops)",
                          "baselines_ops_{ds}.png")


def fig_baselines_latency():
    """Projected best-implementation latency (consistent with
    single_query_latency_best): every method's op-count costed at the measured
    per-edge rate of the compiled from-scratch forward (overhead_ratio -> 1).
    Raw Python-prototype wall-clock lives in results/baselines/results.csv
    (wall_s) but is not the headline — it carries the framework tax and is only
    fair among the three prototype methods, not vs the compiled baseline."""
    df = _load("baselines")
    if df is None:
        return []
    # per (dataset,arch,kind,size): rate = from_scratch wall / from_scratch edges
    rate = {}
    for _, r in df[df.method == "from_scratch"].iterrows():
        rate[(r.dataset, r.arch, r.kind, r["size"])] = (
            r.wall_s / max(r.edges, 1))
    df = df.copy()
    df["proj_ms"] = df.apply(
        lambda r: r.edges * rate.get((r.dataset, r.arch, r.kind, r["size"]), 0) * 1e3,
        axis=1)
    # stash on a temp global via closure
    def val(s):
        return s.proj_ms
    return _fig_baselines(val, "latency (ms, projected best impl.)",
                          "baselines_latency_{ds}.png", df_override=df)


def fig_baselines_exactness():
    """Exactness matrix: max-abs distance from full recompute, per method ×
    (dataset, arch, kind, size). Confirms ours & NeutronRT exact; Ripple
    inexact on the cases baselines.md predicts."""
    df = _load("baselines")
    if df is None:
        return None
    import numpy as np
    df = df.copy()
    df["cell"] = (df.dataset + "\n" + df.arch + " " + df.kind + " " + df["size"].astype(str))
    order = _BASELINE_ORDER
    cells = sorted(df.cell.unique())
    mat = np.full((len(order), len(cells)), np.nan)
    for ci, c in enumerate(cells):
        for mi, nm in enumerate(order):
            v = df[(df.cell == c) & (df.method == nm)].max_abs
            if len(v):
                mat[mi, ci] = float(v.iloc[0])
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(cells)), 2.8))
    with np.errstate(divide="ignore"):
        logm = np.log10(np.clip(mat, 1e-12, None))
    im = ax.imshow(logm, aspect="auto", cmap="magma_r", vmin=-12, vmax=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_BASELINE_STYLE[n][2] for n in order])
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=90, fontsize=5)
    for mi in range(len(order)):
        for ci in range(len(cells)):
            if not np.isnan(mat[mi, ci]):
                v = mat[mi, ci]
                txt = "~0" if v < 1e-4 else (f"{v:.0e}")
                ax.text(ci, mi, txt, ha="center", va="center", fontsize=4,
                        color="white" if logm[mi, ci] > -5 else "black")
    fig.colorbar(im, ax=ax, label="log10 max-abs error", fraction=0.015)
    ax.set_title("Exactness: max-abs distance from full recompute "
                 "(~0 = exact; nonzero = approximate)", fontsize=9)
    fig.tight_layout()
    return fig, "baselines_exactness.png"


def fig_db_roundtrip():
    """Store-backed per-hypothetical DB cost (Neo4j/Memgraph): from-scratch
    re-downloads the full graph every query; streaming reads the L-hop closure;
    ours touches the DB 0 times (one-time full download shown as a reference
    line). One figure per dataset: rows = DB, cols = intervention kind."""
    df = _load("db_roundtrip")
    if df is None:
        return []
    figs = []
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        dbs = sorted(d.db.unique())
        kinds = ["feature", "topo"]
        fig, axes = plt.subplots(len(dbs), len(kinds),
                                 figsize=(3.4 * len(kinds), 2.6 * len(dbs)),
                                 squeeze=False)
        for i, db in enumerate(dbs):
            for j, kind in enumerate(kinds):
                ax = axes[i][j]
                s = d[(d.db == db) & (d.kind == kind)].sort_values("size")
                if s.empty:
                    continue
                ax.plot(s["size"], s.from_scratch_db_ms, "x:", color=C[3],
                        linewidth=1.6, markersize=7, alpha=0.7,
                        label="from-scratch (full download, every query)")
                ax.plot(s["size"], s.streaming_db_ms, "s-", color=C[2],
                        linewidth=2, markersize=5, label="streaming (Δ + L-hop read)")
                onetime = s.one_time_load_ms.iloc[0]
                ax.axhline(onetime, color=C[0], linestyle="--", linewidth=1.4,
                           label="INQ: one-time load (then 0/query)")
                ax.set_xscale("log"); ax.set_yscale("log")
                if i == 0:
                    ax.set_title(f"{kind} Δ")
                if i == len(dbs) - 1:
                    ax.set_xlabel("intervention size")
                if j == 0:
                    ax.set_ylabel(f"{db}\nDB time per query (ms)", fontsize=8)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=6)
        fig.suptitle(f"Store-backed what-if cost — {ds}: from-scratch pays a full "
                     f"download every query;\nours pays it once (break-even ~1)",
                     y=1.02, fontsize=9)
        fig.tight_layout()
        figs.append((fig, f"db_roundtrip_{ds}.png"))
    return figs


def _edges_of(dataset):
    """Total |E| for a dataset, read from its db_export schema.json."""
    import json
    from src.paths import ROOT
    p = os.path.join(ROOT, "db_export", dataset, "schema.json")
    if not os.path.exists(p):
        return None
    return sum(r["count"] for r in json.load(open(p))["relations"])


def _db_pipeline_bars(mode, kind="feature"):
    """Grouped bars of store-backed step-times. `mode`:
      'absolute'     — full+update+gnn STACKED, linear (download dominates)
      'proportional' — each stacked bar normalised to 100% (composition)
      'zoom'         — stacked absolute, y-limited to update+gnn (full clipped)
      'nodownload'   — drop full; stack only update+gnn (baselines) / gnn (ours)
      'overlay'      — the 3 steps as OVERLAID (layered) bars on a LOG axis, each
                       from the floor to its own value (widest=full at back,
                       narrowest=gnn in front) — shows all 3 despite the 1000x
                       range; ours has only gnn.
    Colour = step; texture = arch (RGCN solid, hGAT hatched); x groups =
    intervention size; bars within a group = approach × arch."""
    df = _load("db_pipeline")
    if df is None or "arch" not in df.columns:
        return None
    APPROACH = [("from_scratch", "from-scratch"), ("ripple", "Ripple"),
                ("neutronrt", "NeutronRT"), ("ours", "INQ")]
    STEP_ALL = [("full", C[3]), ("update", C[2]), ("gnn", C[0])]
    STEP = STEP_ALL[1:] if mode == "nodownload" else STEP_ALL
    HATCH = {"rgcn": "", "hgat": "///"}
    archs = ["rgcn", "hgat"]
    dbs = [d for d in ("neo4j", "memgraph") if d in set(df.db.unique())]
    datasets = sorted(set(df.dataset.unique()), key=lambda x: _edges_of(x) or 0)
    fig, axes = plt.subplots(len(dbs), len(datasets),
                             figsize=(4.3 * len(datasets), 3.2 * len(dbs)),
                             squeeze=False)

    def steps_of(row):
        return {"full": row.t_download_ms + row.t_tensor_ms,
                "update": row.t_apply_ms + row.t_rollback_ms,
                "gnn": row.t_gnn_ms}

    for i, db in enumerate(dbs):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            base = df[(df.db == db) & (df.dataset == ds) & (df.kind == kind)]
            if base.empty:
                continue
            sizes = sorted(base["size"].unique())
            nbar = len(APPROACH) * len(archs)
            bw = 1.0
            group_span = nbar * bw + 2.0
            xticks, xlabels = [], []
            zoom_max = 0.0
            for gi, size in enumerate(sizes):
                g0 = gi * group_span
                for ai, (m, mlbl) in enumerate(APPROACH):
                    for hi, arch in enumerate(archs):
                        x = g0 + (ai * len(archs) + hi) * bw
                        r = base[(base.method == m) & (base.arch == arch)
                                 & (base["size"] == size)]
                        if r.empty:
                            continue
                        st = steps_of(r.iloc[0])
                        vals = [st[name] for name, _ in STEP]
                        if mode == "proportional":
                            tot = sum(vals) or 1.0
                            vals = [v / tot * 100 for v in vals]
                        zoom_max = max(zoom_max, st["update"] + st["gnn"])
                        if mode == "overlay":
                            # layered bars (not stacked): each from the floor to
                            # its own value, decreasing width so all are visible.
                            widths = [bw * 0.92, bw * 0.58, bw * 0.30]
                            for val, (_, col), w in zip(vals, STEP, widths):
                                if val <= 0:
                                    continue
                                ax.bar(x, val, width=w, color=col, hatch=HATCH[arch],
                                       edgecolor="white", linewidth=0.3, zorder=3)
                        else:
                            bottom = 0.0
                            for val, (_, col) in zip(vals, STEP):
                                if val > 0:   # skip zeros (ours; and log-safe)
                                    ax.bar(x, val, bottom=bottom, width=bw * 0.9,
                                           color=col, hatch=HATCH[arch],
                                           edgecolor="white", linewidth=0.3)
                                bottom += val
                    xticks.append(g0 + (ai * len(archs) + 0.5) * bw)
                    xlabels.append(mlbl)
                grp = base[base["size"] == size]
                glabel = (f"{int(grp.pct.iloc[0])}%" if "pct" in base.columns
                          and not grp.empty else f"size {size}")
                ax.text(g0 + (nbar * bw) / 2 - bw / 2, -0.30, glabel,
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8, fontweight="bold")
            if mode == "zoom" and zoom_max > 0:
                ax.set_ylim(0, zoom_max * 1.25)   # full/download overflows (clipped)
            if mode in ("overlay", "log"):
                ax.set_yscale("log"); ax.set_ylim(bottom=0.05)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=45, fontsize=6, ha="right")
            ax.margins(x=0.02)
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                unit = ("% of query time" if mode == "proportional"
                        else "time per query (ms, log)" if mode in ("overlay", "log")
                        else "time per query (ms)")
                ax.set_ylabel(f"{db}\n{unit}", fontsize=9)
    import matplotlib.patches as mp
    handles = [mp.Patch(color=col, label=name) for name, col in STEP]
    handles += [mp.Patch(facecolor="#ccc", label="RGCN (solid)"),
                mp.Patch(facecolor="#ccc", hatch="///", label="hGAT (hatched)")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    TITLE = {
        "absolute": "absolute (linear): download dominates; INQ' bar is gnn only",
        "proportional": "proportional (100%): composition — full ~99% for baselines, INQ = 100% gnn",
        "zoom": "zoomed to update+gnn range: full (download) overflows/clipped",
        "nodownload": "download removed: only update+gnn (baselines) / gnn (ours)",
        "overlay": "overlaid (layered) bars on log axis: full/update/gnn each from floor; INQ = gnn",
        "log": "stacked on LOG axis (segment sizes distorted; makes INQ' total visible)",
    }[mode]
    fig.suptitle(f"Store-backed what-if step-times ({kind} Δ) — {TITLE}", y=1.04, fontsize=9)
    fig.tight_layout()
    return (fig, f"db_pipeline_steps{'' if mode == 'absolute' else '_' + mode}.png")


def fig_db_pipeline():
    figs = []
    for mode in ("absolute", "proportional", "zoom", "nodownload", "overlay", "log"):
        out = _db_pipeline_bars(mode)
        if out:
            figs.append(out)
    return figs


def fig_db_finbench_2x2(only=None):
    """FinBench focus: 2x2 panels (rows=DB neo4j/memgraph, cols=kind feature/topo),
    download removed (update+gnn only, like nodownload), with the UPDATE step (orange)
    stacked ON TOP of the GNN step (blue). ours = gnn only (no DB update)."""
    df = _load("db_pipeline")
    if df is None or "arch" not in df.columns:
        return []
    df = df[df.dataset == "finbench"]
    if df.empty:
        return []
    APPROACH = [("from_scratch", "from-scratch"), ("ripple", "Ripple"),
                ("neutronrt", "NeutronRT"), ("ours", "INQ")]
    # gnn first (bottom), update second (top) — update orange sits above gnn blue.
    STEP = [("gnn", C[0]), ("update", C[2])]
    HATCH = {"rgcn": "", "hgat": "///"}
    archs = ["rgcn", "hgat"]
    dbs = [d for d in ("neo4j", "memgraph") if d in set(df.db.unique())
           and (only is None or d == only)]
    kinds = ["feature", "topo"]
    fig, axes = plt.subplots(len(dbs), len(kinds),
                             figsize=(4.6 * len(kinds), 3.2 * len(dbs)), squeeze=False)

    def steps_of(row):
        return {"update": row.t_apply_ms + row.t_rollback_ms, "gnn": row.t_gnn_ms}

    for i, db in enumerate(dbs):
        for j, kind in enumerate(kinds):
            ax = axes[i][j]
            base = df[(df.db == db) & (df.kind == kind)]
            if base.empty:
                continue
            sizes = sorted(base["size"].unique())
            nbar = len(APPROACH) * len(archs)
            bw = 1.0
            group_span = nbar * bw + 2.0
            xticks, xlabels = [], []
            for gi, size in enumerate(sizes):
                g0 = gi * group_span
                for ai, (m, mlbl) in enumerate(APPROACH):
                    for hi, arch in enumerate(archs):
                        x = g0 + (ai * len(archs) + hi) * bw
                        r = base[(base.method == m) & (base.arch == arch)
                                 & (base["size"] == size)]
                        if r.empty:
                            continue
                        st = steps_of(r.iloc[0])
                        bottom = 0.0
                        for name, col in STEP:
                            val = st[name]
                            if val > 0:
                                ax.bar(x, val, bottom=bottom, width=bw * 0.9,
                                       color=col, hatch=HATCH[arch],
                                       edgecolor="white", linewidth=0.3)
                            bottom += val
                    xticks.append(g0 + (ai * len(archs) + 0.5) * bw)
                    xlabels.append(mlbl)
                grp = base[base["size"] == size]
                glabel = f"{int(grp.pct.iloc[0])}%" if "pct" in base.columns else f"size {size}"
                ax.text(g0 + (nbar * bw) / 2 - bw / 2, -0.30, glabel,
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8, fontweight="bold")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=45, fontsize=6, ha="right")
            ax.margins(x=0.02)
            klabel = {"feature": "attribute", "topo": "structure"}.get(kind, kind)
            ax.set_title(f"{klabel} Δ")
            if j == 0:
                ax.set_ylabel(f"{db}\ntime per query (ms)", fontsize=9)
    import matplotlib.patches as mp
    handles = [mp.Patch(facecolor=C[0], label="RGCN inference"),
               mp.Patch(facecolor=C[0], hatch="///", label="hGAT inference"),
               mp.Patch(facecolor=C[2], label="RGCN update"),
               mp.Patch(facecolor=C[2], hatch="///", label="hGAT update")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Latency vs intervention size against the baselines for FinBench",
                 y=1.03, fontsize=10)
    fig.tight_layout()
    if only:
        return [(fig, f"db_finbench_{only}.png")]
    return [(fig, "db_finbench_2x2.png"), *fig_db_finbench_2x2("neo4j")]


def fig_db_full_appendix():
    """Appendix: the FULL store-backed step-time picture — every dataset (cols) x
    every DB/kind combination (rows), download removed, update (orange) stacked on
    gnn (blue), ours = gnn only. One comprehensive figure."""
    df = _load("db_pipeline")
    if df is None or "arch" not in df.columns:
        return []
    APPROACH = [("from_scratch", "from-scratch"), ("ripple", "Ripple"),
                ("neutronrt", "NeutronRT"), ("ours", "INQ")]
    STEP = [("gnn", C[0]), ("update", C[2])]      # gnn bottom, update on top
    HATCH = {"rgcn": "", "hgat": "///"}
    archs = ["rgcn", "hgat"]
    dbs = [d for d in ("neo4j", "memgraph") if d in set(df.db.unique())]
    kinds = ["feature", "topo"]
    rows = [(db, k) for db in dbs for k in kinds]
    datasets = sorted(set(df.dataset.unique()), key=lambda x: _edges_of(x) or 0)
    fig, axes = plt.subplots(len(rows), len(datasets),
                             figsize=(4.2 * len(datasets), 2.7 * len(rows)),
                             squeeze=False)

    def steps_of(r):
        return {"update": r.t_apply_ms + r.t_rollback_ms, "gnn": r.t_gnn_ms}

    for i, (db, kind) in enumerate(rows):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            base = df[(df.db == db) & (df.dataset == ds) & (df.kind == kind)]
            if base.empty:
                continue
            sizes = sorted(base["size"].unique())
            nbar = len(APPROACH) * len(archs)
            bw = 1.0
            group_span = nbar * bw + 2.0
            xticks, xlabels = [], []
            for gi, size in enumerate(sizes):
                g0 = gi * group_span
                for ai, (m, mlbl) in enumerate(APPROACH):
                    for hi, arch in enumerate(archs):
                        x = g0 + (ai * len(archs) + hi) * bw
                        r = base[(base.method == m) & (base.arch == arch)
                                 & (base["size"] == size)]
                        if r.empty:
                            continue
                        st = steps_of(r.iloc[0])
                        bottom = 0.0
                        for name, col in STEP:
                            val = st[name]
                            if val > 0:
                                ax.bar(x, val, bottom=bottom, width=bw * 0.9,
                                       color=col, hatch=HATCH[arch],
                                       edgecolor="white", linewidth=0.3)
                            bottom += val
                    xticks.append(g0 + (ai * len(archs) + 0.5) * bw)
                    xlabels.append(mlbl)
                grp = base[base["size"] == size]
                glabel = f"{int(grp.pct.iloc[0])}%" if "pct" in base.columns else f"size {size}"
                ax.text(g0 + (nbar * bw) / 2 - bw / 2, -0.32, glabel,
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=7, fontweight="bold")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=45, fontsize=5.5, ha="right")
            ax.margins(x=0.02)
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(f"{db} · {kind} Δ\ntime/query (ms)", fontsize=8)
    import matplotlib.patches as mp
    handles = [mp.Patch(color=C[0], label="gnn (compute)"),
               mp.Patch(color=C[2], label="update (DB apply+rollback)"),
               mp.Patch(facecolor="#ccc", label="RGCN (solid)"),
               mp.Patch(facecolor="#ccc", hatch="///", label="hGAT (hatched)")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Store-backed what-if step-times (all datasets) — update (orange) stacked on "
                 "gnn (blue); download removed; INQ = gnn only", y=1.005, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return [(fig, "db_full_appendix.png")]


# --- fractional-size scaling figures (1-3) --------------------------------
# color = approach, linestyle = arch — consistent across all three.
_SCALE_APPROACH = {"from_scratch": (C[3], "from-scratch"), "ours": (C[0], "ours")}
_SCALE_LS = {"rgcn": "-", "hgat": "--"}
_SCALE_DATASETS = ["eicu", "finbench", "snb", "faers"]


def _scaling_df():
    df = _load("scaling")
    if df is None:
        return None
    df = df.copy()
    df["pct"] = (df["frac"] * 100).round().astype(int)
    return df


def fig_scaling_L2():
    """Figure 1: L=2, x = intervention size (% of dataset), incremental vs
    from-scratch, all datasets, 4 lines (approach × arch). Two figures: one for
    operations (edge-messages), one for latency (best-impl)."""
    df = _scaling_df()
    if df is None:
        return []
    from matplotlib.ticker import FuncFormatter, NullFormatter
    import matplotlib.lines as ml
    d2 = df[df.layers == 2]
    kinds = ["feature", "topo"]
    klab = {"feature": "attribute", "topo": "structure"}
    datasets = [x for x in _SCALE_DATASETS if x in set(d2.dataset.unique())]
    out = []
    for ycol, ylab, fname, title in (
            ("flops", "FLOPs", "scaling_L2_operations.png",
             "FLOPs comparison between from-scratch and INQ evaluation"),
            ("latency_best_ms", "latency (ms)", "scaling_L2_latency.png",
             "Latency comparison between from-scratch and INQ evaluation")):
        # plain (non-scientific) tick labels; ints for FLOPs, ms for latency
        if ycol == "flops":
            fmt = lambda v, _: f"{v:,.0f}"
        else:
            fmt = lambda v, _: (f"{v:,.0f}" if v >= 100 else f"{v:.0f}"
                                if v >= 10 else f"{v:.1f}")
        fig, axes = plt.subplots(len(kinds), len(datasets),
                                 figsize=(4.2 * len(datasets), 2.7 * len(kinds)),
                                 squeeze=False)
        for i, kind in enumerate(kinds):
            for j, ds in enumerate(datasets):
                ax = axes[i][j]
                sub = d2[(d2.dataset == ds) & (d2.kind == kind)]
                yvals = []
                # color = architecture; linestyle = method (INQ dashed, from-scratch solid)
                arch_color = {"rgcn": C[0], "hgat": C[3]}
                method_ls = {"from_scratch": "-", "ours": "--"}
                for method, ls in method_ls.items():
                    for arch, color in arch_color.items():
                        s = sub[(sub.method == method) & (sub.arch == arch)].sort_values("pct")
                        if s.empty:
                            continue
                        ax.plot(s.pct, s[ycol], ls, color=color, lw=2, marker="o", ms=4)
                        yvals.extend(s[ycol].tolist())
                ax.set_yscale("log")
                # a y-tick at every data point value; plain numbers, no 10^x.
                # collapse ticks within ~4% (indistinguishable on a log axis) so
                # labels don't overprint where lines bunch together.
                if yvals:
                    # bottom-right panel = faers, structure Δ. For FLOPs (near-
                    # coincident RGCN/hGAT values) keep every tick but nudge close
                    # labels apart; for latency it over-crowds, so thin the ticks.
                    is_br = (i == len(kinds) - 1) and (j == len(datasets) - 1)
                    if ycol == "flops":
                        thresh, sep_labels = (1.001, True) if is_br else (1.11, False)
                    else:  # latency: thin the faers-structure ticks
                        thresh, sep_labels = (1.06, False) if is_br else (1.11, False)
                    ticks = []
                    for v in sorted(set(round(v, 1) for v in yvals)):
                        if not ticks or v > ticks[-1] * thresh:
                            ticks.append(v)
                    ax.set_yticks(ticks)
                    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.tick_params(axis="y", which="major", labelsize=5.5)
                    ax.tick_params(axis="y", which="minor", length=0)
                    if sep_labels:
                        # push labels of near-coincident ticks apart: lower one
                        # below its tick, upper one above, so both stay readable.
                        labs = ax.get_yticklabels()
                        for k in range(1, len(ticks)):
                            if ticks[k] <= ticks[k - 1] * 1.10:
                                labs[k - 1].set_verticalalignment("top")
                                labs[k].set_verticalalignment("bottom")
                if i == 0:
                    ax.set_title(ds)
                if i == len(kinds) - 1:
                    ax.set_xlabel("update size")
                if j == 0:
                    ax.set_ylabel(f"{klab[kind]} Δ\n{ylab}", fontsize=8)
        h = [ml.Line2D([], [], color=C[0], ls="--", lw=2, marker="o", ms=4, label="INQ-RGCN"),
             ml.Line2D([], [], color=C[0], ls="-", lw=2, marker="o", ms=4, label="from-scratch-RGCN"),
             ml.Line2D([], [], color=C[3], ls="--", lw=2, marker="o", ms=4, label="INQ-hGAT"),
             ml.Line2D([], [], color=C[3], ls="-", lw=2, marker="o", ms=4, label="from-scratch-hGAT")]
        fig.legend(handles=h, frameon=False, fontsize=8, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, y=1.03, fontsize=10)
        fig.tight_layout()
        out.append((fig, fname))
    return out


def _scaling_legend(fig):
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=C[0], ls="-", lw=2, label="INQ-RGCN"),
         ml.Line2D([], [], color=C[0], ls="--", lw=2, label="INQ-hGAT"),
         ml.Line2D([], [], color=C[3], ls="-", lw=2, label="from-scratch RGCN"),
         ml.Line2D([], [], color=C[3], ls="--", lw=2, label="from-scratch hGAT")]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))


def fig_scaling_layers():
    """Figure 2: one figure per intervention size; x = layers (1-4), y = latency,
    4 lines (approach × arch), all datasets as columns."""
    df = _scaling_df()
    if df is None:
        return []
    datasets = [x for x in _SCALE_DATASETS if x in set(df.dataset.unique())]
    out = []
    for pct in sorted(df.pct.unique()):
        for kind in ["feature", "topo"]:
            dp = df[(df.pct == pct) & (df.kind == kind)]
            if dp.empty:
                continue
            fig, axes = plt.subplots(1, len(datasets),
                                     figsize=(3.2 * len(datasets), 3.0), squeeze=False)
            for j, ds in enumerate(datasets):
                ax = axes[0][j]
                sub = dp[dp.dataset == ds]
                for method, (color, _) in _SCALE_APPROACH.items():
                    for arch, ls in _SCALE_LS.items():
                        s = sub[(sub.method == method) & (sub.arch == arch)].sort_values("layers")
                        if s.empty:
                            continue
                        ax.plot(s.layers, s.latency_best_ms, ls, color=color, lw=2,
                                marker="o", ms=4)
                ax.set_xticks([1, 2, 3, 4]); ax.set_yscale("log")
                ax.set_title(ds); ax.set_xlabel("layers L")
                if j == 0:
                    ax.set_ylabel("latency (ms, best impl.)")
            _scaling_legend(fig)
            fig.suptitle(f"Scalability vs depth — {kind} Δ at {pct}% — latency",
                         y=1.04, fontsize=9)
            fig.tight_layout()
            out.append((fig, f"scaling_layers_{kind}_{pct}pct.png"))
    return out


def fig_scaling_signflips():
    """Figure 3: one figure per intervention size; x = layers (1-4), y = sign-
    flips (ours only — from-scratch has none), 2 lines (RGCN, hGAT), datasets as
    columns. Feature Δ (delta path)."""
    df = _scaling_df()
    if df is None:
        return []
    datasets = [x for x in _SCALE_DATASETS if x in set(df.dataset.unique())]
    d = df[(df.method == "ours") & (df.kind == "feature")]
    out = []
    for pct in sorted(d.pct.unique()):
        dp = d[d.pct == pct]
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(3.2 * len(datasets), 3.0), squeeze=False)
        for j, ds in enumerate(datasets):
            ax = axes[0][j]
            sub = dp[dp.dataset == ds]
            for arch, ls in _SCALE_LS.items():
                s = sub[sub.arch == arch].sort_values("layers")
                if s.empty:
                    continue
                ax.plot(s.layers, s.sign_flips.clip(lower=0.5), ls, color=C[0],
                        lw=2, marker="^", ms=5, label=("INQ-RGCN" if arch=="rgcn" else "INQ-hGAT"))
            ax.set_xticks([1, 2, 3, 4]); ax.set_yscale("log")
            ax.set_title(ds); ax.set_xlabel("layers L")
            if j == 0:
                ax.set_ylabel("sign-flips (nonlinear corrections)")
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Scalability vs depth — sign-flips (INQ), feature Δ at {pct}%",
                     y=1.04, fontsize=9)
        fig.tight_layout()
        out.append((fig, f"scaling_signflips_{pct}pct.png"))
    return out


_SCALE_ARCH_C = {"rgcn": C[0], "hgat": C[1]}


def fig_scaling_speedup_flips():
    """Twin-axis: for each dataset (feature Δ), latency SPEEDUP (from-scratch/ours,
    left axis, solid) and SIGN-FLIPS (right axis, log, dashed) vs depth L. Shows
    the inverse relationship — flips climb as speedup decays. One figure per size;
    the 10% one is the paper's."""
    df = _scaling_df()
    if df is None:
        return []
    datasets = [x for x in _SCALE_DATASETS if x in set(df.dataset.unique())]
    out = []
    for kind in ["feature", "topo"]:
      for pct in sorted(df.pct.unique()):
        dp = df[(df.pct == pct) & (df.kind == kind)]
        if dp.empty:
            continue
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(4.4 * len(datasets), 2.5), squeeze=False)
        # color = (metric x arch), all solid lines. speedup: INQ-RGCN blue,
        # INQ-hGAT green. sign-flips: RGCN orange, hGAT pink.
        SPD_C = {"rgcn": C[0], "hgat": C[3]}
        BROWN = "#8c564b"
        FLIP_C = {"rgcn": C[2], "hgat": BROWN}
        for j, ds in enumerate(datasets):
            axL = axes[0][j]
            axR = axL.twinx()
            sub = dp[dp.dataset == ds]
            for arch in ("rgcn", "hgat"):
                sa = sub[sub.arch == arch]
                s = sa[sa.method == "ours"].sort_values("layers")
                p = sa.pivot_table(index="layers", columns="method",
                                   values="latency_best_ms").sort_index()
                if "ours" not in p or "from_scratch" not in p:
                    continue
                spd = (p["from_scratch"] / p["ours"])
                axL.plot(spd.index, spd.values, "-", color=SPD_C[arch], lw=2, marker="o", ms=4)
                axR.plot(s.layers, s.sign_flips.clip(lower=0.5), "-", color=FLIP_C[arch],
                         lw=1.6, marker="^", ms=5, alpha=0.9)
            axL.set_xticks([1, 2, 3, 4]); axL.set_xlabel("layers L")
            axL.axhline(1.0, color="#999", lw=0.8, ls=":")
            axR.set_yscale("log")
            axL.set_title(ds)
            if j == 0:
                axL.set_ylabel("latency speedup")
            if j == len(datasets) - 1:
                axR.set_ylabel("sign-flips")
        import matplotlib.lines as ml
        h = [ml.Line2D([], [], color=C[0], ls="-", lw=2, marker="o", ms=4, label="INQ-RGCN"),
             ml.Line2D([], [], color=C[3], ls="-", lw=2, marker="o", ms=4, label="INQ-hGAT"),
             ml.Line2D([], [], color=C[2], ls="-", lw=2, marker="^", ms=5, label="RGCN sign-flips"),
             ml.Line2D([], [], color=BROWN, ls="-", lw=2, marker="^", ms=5, label="hGAT sign-flips")]
        fig.legend(handles=h, frameon=False, fontsize=8, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.5, 1.0))
        klabel = {"feature": "attribute", "topo": "structure"}[kind]
        fig.suptitle(f"Speedup and sign flips against the number of layers for {klabel} delta",
                     y=1.03, fontsize=10)
        fig.tight_layout()
        out.append((fig, f"scaling_speedup_flips_{kind}_{pct}pct.png"))
    return out


def fig_interactive_eicu(out):
    """Two-column (RGCN left, hGAT right) figure for the composing what-if session
    on eICU, Neo4j-backed. ROW 1: query Q per step (from-scratch=ground truth,
    ours overlays, Ripple drifts). ROW 2: cumulative end-to-end latency (ms) —
    from-scratch & Ripple pay DB apply+download every state (+ compute); ours pays
    only its resident compute (+ one-time S0 load). Backtrack step marked. PDF+PNG."""
    cols = ["rgcn", "hgat"]                       # RGCN left, hGAT right
    titles = {"rgcn": "RGCN", "hgat": "hGAT"}
    thr = out["Q_threshold"]
    bt = out["backtrack_step"]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.0), sharex=True,
                             gridspec_kw=dict(hspace=0.22, wspace=0.32,
                                              height_ratios=[1.2, 1]))
    for j, arch in enumerate(cols):
        rows = out["archs"][arch]["states"]
        steps = [r["step"] for r in rows]
        labels = [f"S{i}" for i in steps]
        Qa = [r["Q_scratch"] for r in rows]
        Qb = [r["Q_ours"] for r in rows]
        Qc = [r["Q_ripple"] for r in rows]
        Qd = [r.get("Q_neutronrt") for r in rows]
        has_neu = out.get("show_neutronrt", True) and all(v is not None for v in Qd)
        ax0, ax1 = axes[0][j], axes[1][j]
        for ax in (ax0, ax1):
            ax.axvline(bt, color="#bbb", lw=6, alpha=0.35, zorder=0)
        ax0.annotate("backtrack", (bt, 1.0), xytext=(0, 2), textcoords="offset points",
                     xycoords=("data", "axes fraction"), ha="center", va="bottom",
                     fontsize=6.5, color="#777")
        ax0.set_title(titles[arch], fontsize=9)

        # ROW 1: query value
        ax0.plot(steps, Qa, "-", color=C[3], lw=2, marker="o", ms=4,
                 label="from-scratch (truth)", zorder=3)
        if has_neu:   # NeutronRT is exact — also overlays truth
            ax0.plot(steps, Qd, ":", color=C[4], lw=1.4, marker="D", ms=4, mfc="none",
                     label="NeutronRT", zorder=3)
        ax0.plot(steps, Qb, "--", color=C[0], lw=1.6, marker="s", ms=5, mfc="none",
                 label=f"INQ-{titles[arch]}", zorder=4)
        ax0.plot(steps, Qc, "-", color=C[5], lw=2, marker="^", ms=4, label="Ripple", zorder=3)
        if min(min(Qa), min(Qc)) - 0.05 <= thr <= max(max(Qa), max(Qc)) + 0.05:
            ax0.axhline(thr, color="#999", ls=":", lw=1)
            ax0.annotate(f"≤{int(thr*100)}%", (steps[-1], thr), fontsize=6, color="#777",
                         va="bottom", ha="right")
        # signed Ripple deviation (Q_ripple - truth) as a percentage, at EVERY
        # step (S1..S4). Placed below each Ripple point, except the lowest point
        # in the panel (placed above so it doesn't clip the axis). RGCN's lines
        # are tightly spaced, so nudge its labels a little further down.
        allmin = min(Qa + Qb + Qc)
        for si in range(1, len(rows)):
            dev = (rows[si]["Q_ripple"] - rows[si]["Q_scratch"]) * 100
            last = (si == len(rows) - 1)
            below = Qc[si] > allmin + 1e-9
            dy = (-13 if arch == "rgcn" else -10) if below else 8
            ax0.annotate(f"{dev:.1f} %", (steps[si], Qc[si]), xytext=(0, dy),
                         textcoords="offset points", fontsize=6, color=C[5],
                         ha="center", va=("top" if below else "bottom"),
                         fontweight=("bold" if last else "normal"))
        if j == 0:
            ax0.set_ylabel(out.get("q_ylabel", "high-risk\nfraction Q"), fontsize=8)
        ax0.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.6)  # per-column (INQ-<arch>)
        ax0.tick_params(labelsize=7)

        # ROW 2: cumulative end-to-end latency (ms)
        ls = [r["cum_lat_ms_scratch"] for r in rows]
        lr = [r["cum_lat_ms_ripple"] for r in rows]
        lo = [r["cum_lat_ms_ours"] for r in rows]
        ax1.plot(steps, ls, "-", color=C[3], lw=2, marker="o", ms=4, label="from-scratch")
        ax1.plot(steps, lr, "-", color=C[5], lw=2, marker="^", ms=4, label="Ripple")
        if has_neu:   # NeutronRT re-fetches every state too — climbs with baselines
            ln = [r["cum_lat_ms_neutronrt"] for r in rows]
            ax1.plot(steps, ln, ":", color=C[4], lw=1.6, marker="D", ms=4, label="NeutronRT")
        ax1.plot(steps, lo, "-", color=C[0], lw=2, marker="s", ms=4, label=f"INQ-{titles[arch]}")
        if j == 0:
            ax1.set_ylabel("cum. latency\n(ms)", fontsize=8)
        ax1.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.6)  # per-column (INQ-<arch>)
        ax1.set_xlabel("session step", fontsize=8)
        ax1.set_xticks(steps); ax1.set_xticklabels(labels)
        ax1.tick_params(labelsize=7)

    fig.suptitle(out.get("fig_title", "Interactive what-if session — eICU (Neo4j-backed)"),
                 fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fname = out.get("fig_name", "interactive_eicu")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{fname}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_db_cumulative(size=100, n_queries=25):
    """The store-backed cost the RIGHT way: cumulative DB time over a session of
    N what-if queries. from-scratch re-downloads every query (steep line);
    streaming reads the affected region every query (shallower line); ours pays
    one full download then 0/query (flat after the start). Makes 'pay once vs
    pay every query' obvious, which the per-query figure cannot."""
    df = _load("db_roundtrip")
    if df is None:
        return []
    figs = []
    for db in sorted(df.db.unique()):
        datasets = sorted(df[df.db == db].dataset.unique())
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(3.2 * len(datasets), 3.0), squeeze=False)
        k = list(range(0, n_queries + 1))
        for j, ds in enumerate(datasets):
            ax = axes[0][j]
            s = df[(df.db == db) & (df.dataset == ds) & (df.kind == "feature")
                   & (df["size"] == size)]
            if s.empty:
                continue
            r = s.iloc[0]
            load = r.one_time_load_ms / 1e3
            scr = r.from_scratch_db_ms / 1e3
            strm = r.streaming_db_ms / 1e3
            ax.plot(k, [q * scr for q in k], "-", color=C[3], linewidth=2,
                    label="from-scratch (download every query)")
            ax.plot(k, [q * strm for q in k], "-", color=C[2], linewidth=2,
                    label="streaming (read region every query)")
            ax.plot(k, [load + q * 0 for q in k], "-", color=C[0], linewidth=2.5,
                    label="INQ (load once, then 0/query)")
            ax.set_title(ds)
            ax.set_xlabel("what-if queries in session")
            if j == 0:
                ax.set_ylabel("cumulative DB time (s)")
                ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        fig.suptitle(f"Store-backed session cost — {db} (feature Δ, size {size}): "
                     f"ours flattens, others grow every query", y=1.02, fontsize=9)
        fig.tight_layout()
        figs.append((fig, f"db_cumulative_{db}.png"))
    return figs


def fig_single_query():
    df = _load("single_query_efficiency")
    if df is None:
        return None
    datasets = sorted(df.dataset.unique())
    # one row per intervention kind; edges split into add-only/del-only/swap
    kinds = [k for k in ["feature", "edge_add", "edge_del", "edge_swap"]
             if k in set(df.kind.unique())]
    labels = {"feature": "feature Δ", "edge_add": "edge add-only Δ",
              "edge_del": "edge delete-only Δ", "edge_swap": "edge swap Δ"}
    units = {"feature": "cohort nodes"}
    fig, axes = plt.subplots(len(kinds), len(datasets),
                             figsize=(3.2 * len(datasets), 1.5 * len(kinds) + 0.9),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for i, kind in enumerate(kinds):
            ax = axes[i][j]
            sub = df[(df.dataset == ds) & (df.kind == kind)]
            for a, arch in enumerate(sorted(sub.arch.unique())):
                s = sub[sub.arch == arch].sort_values("size")
                ax.plot(s["size"], s.wall_incremental_s * 1e3, "o-", color=C[a],
                        linewidth=2, markersize=4, label=f"INQ-{'RGCN' if arch=='rgcn' else 'hGAT'}")
                ax.plot(s["size"], s.wall_scratch_s * 1e3, "--", color=C[a],
                        linewidth=1.2, alpha=0.55, label=f"from-scratch ({arch})")
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(ds)
            if i == len(kinds) - 1:
                ax.set_xlabel("intervention size (%s)"
                              % units.get(kind, "edges"))
            if j == 0:
                ax.set_ylabel(f"{labels[kind]}\nlatency (ms)", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Single what-if query — WALL-CLOCK (unoptimized Python prototype; "
                 "a lower bound.\nSee overhead_proxy_vs_wall for the "
                 "implementation-independent algorithmic measure)", y=1.02, fontsize=8.5)
    fig.tight_layout()
    return fig, "single_query_latency.png"


def fig_timing_model():
    """The three measured scenarios (warm / cold one-off / cold pipeline vs
    resident system) and the session amortization that reconciles them."""
    df = _load("single_query_efficiency")
    if df is None or "speedup_cold_oneoff" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) median speedup per scenario, one row per dataset/arch/kind
    ax = axes[0]
    med = (df.groupby(["dataset", "arch", "kind"])
           [["speedup", "speedup_cold_oneoff", "speedup_cold_vs_resident"]]
           .median().reset_index())
    med = med.sort_values(["dataset", "arch", "kind"]).reset_index(drop=True)
    ys = range(len(med))
    ax.scatter(med["speedup"], ys, s=22, color=C[0], marker="o",
               label="warm (both resident)", zorder=3)
    ax.scatter(med["speedup_cold_oneoff"], ys, s=22, color=C[2], marker="s",
               label="cold one-off", zorder=3)
    # nearly coincides with warm (the finding) — hollow so both stay visible
    ax.scatter(med["speedup_cold_vs_resident"], ys, s=60, facecolors="none",
               edgecolors=C[1], marker="D", label="cold pipeline vs resident",
               zorder=4, linewidths=1.4)
    ax.axvline(1.0, color="#999999", linewidth=1, linestyle=":")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r.dataset} {r.arch} {r.kind}" for r in med.itertuples()],
                       fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("median speedup vs from-scratch (log)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # (b) cumulative session cost: cache build amortizes within ~2 queries
    ax = axes[1]
    picks = [("finbench", "rgcn", "topo"), ("faers", "rgcn", "feature"),
             ("eicu", "rgcn", "feature")]
    k = list(range(0, 11))
    for i, (ds, arch, kind) in enumerate(picks):
        sub = df[(df.dataset == ds) & (df.arch == arch) & (df.kind == kind)]
        if sub.empty:
            continue
        scr = sub.wall_scratch_s.median()
        inc = sub.wall_incremental_s.median()
        cache = sub.cache_build_s.iloc[0]
        ax.plot(k, [q * scr for q in k], "--", color=C[i], linewidth=1.2,
                alpha=0.6)
        ax.plot(k, [cache + q * inc for q in k], "-", color=C[i], linewidth=2,
                label=f"{ds} ({kind} Δ)")
    ax.set_xlabel("queries in the session")
    ax.set_ylabel("cumulative cost (s)")
    ax.legend(frameon=False, fontsize=7, title="solid = cached session\ndashed = per-query scratch",
              title_fontsize=7)
    fig.suptitle("Timing model: setup costs, scenarios, and session amortization",
                 y=1.03)
    return fig, "timing_model.png"


def fig_layer_scaling():
    df = _load("layer_scaling")
    if df is None:
        return None
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.9),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        sub = df[df.dataset == ds]
        for a, arch in enumerate(sorted(sub.arch.unique())):
            for kind, style in (("feature", "o-"), ("topo", "s--")):
                s = sub[(sub.arch == arch) & (sub.kind == kind)].sort_values("layers")
                ax.plot(s.layers, s.speedup, style, color=C[a], linewidth=2,
                        markersize=4, label=f"{arch}, {kind} Δ")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(ds)
        ax.set_xlabel("layers L")
        if j == 0:
            ax.set_ylabel("speedup vs from-scratch")
            ax.legend(frameon=False, fontsize=7)
        ax.set_yscale("log")
    fig.suptitle("Layer scaling of INQ speedup", y=1.05)
    return fig, "layer_scaling.png"


def fig_exploration():
    df = _load("exploration_reuse", "summary.csv")
    if df is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # headline: recomputation avoided vs the sharing knob
    ax = axes[0]
    for a, arch in enumerate(sorted(df.arch.unique())):
        s = (df[df.arch == arch].groupby("share_rate")["reuse_ratio_edges"]
             .mean().reset_index())
        ax.plot(s.share_rate, 1 - s.reuse_ratio_edges, "o-", color=C[a],
                linewidth=2, markersize=4, label=arch)
    ax.set_xlabel("prefix-sharing rate (trace knob)")
    ax.set_ylabel("recomputation avoided\n(1 − reuse/no-reuse edge work)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)

    # realized sharing vs benefit (scatter, one dot per trace config)
    ax = axes[1]
    for a, arch in enumerate(sorted(df.arch.unique())):
        s = df[df.arch == arch]
        ax.scatter(s.sharing, 1 - s.reuse_ratio_edges, s=18, color=C[a],
                   label=arch, alpha=0.8)
    ax.set_xlabel("realized sharing (multi-parent states / states)")
    ax.set_ylabel("recomputation avoided")
    ax.set_ylim(bottom=0)
    fig.suptitle("Exploration reuse: benefit rises with prefix sharing", y=1.03)
    return fig, "exploration_reuse.png"


def fig_exploration_totals():
    df = _load("exploration_reuse", "summary.csv")
    if df is None:
        return None
    agg = df.groupby("share_rate")[["scratch_wall_s", "incremental_wall_s",
                                    "reuse_wall_s"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels = ["from-scratch", "INQ (no reuse)", "INQ + reuse"]
    for i, col in enumerate(["scratch_wall_s", "incremental_wall_s", "reuse_wall_s"]):
        ax.plot(agg.share_rate, agg[col], "o-", color=C[i], linewidth=2,
                markersize=4, label=labels[i])
    ax.set_xlabel("prefix-sharing rate")
    ax.set_ylabel("total trace wall-clock (s)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Whole-trace cost under the three rungs")
    return fig, "exploration_totals.png"


# ------------------------------------------------------------------ revision
# Single-column (3.4 in) PDFs, every text >= 9 pt at print size. Only rows that
# passed the exactness gate are plotted; latency is MEASURED (never the
# latency_best_ms projection).
_REV_W = 3.4
_REV_RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
           "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}
_REV_DS = ["eicu", "finbench", "snb", "faers"]
_REV_DS_C = {"eicu": C[0], "finbench": C[1], "snb": C[2], "faers": C[4]}
_REV_ARCH_C = {"rgcn": C[0], "hgat": C[3]}
_REV_ARCH = {"rgcn": "RGCN", "hgat": "hGAT"}
_REV_KIND = {"feature": "attribute Δ", "topo": "structure Δ"}


def _rev_lat(exp="revision"):
    df = _load(exp, "latency.csv")
    if df is None:
        return None
    df = df[df.passed == True].copy()   # exactness gate
    df["pct"] = (df.frac * 100).round().astype(int)
    return df


def _rev_datasets(df):
    return [d for d in _REV_DS if d in set(df.dataset)]


def fig_revision_memory():
    s = _load("revision", "cache_size_summary.csv")
    b = _load("revision", "build_cost.csv")
    if s is None:
        return None
    s = s[s.L == 2]
    build = {}
    if b is not None:
        bm = b[(b.backend == "memory") & (b.phase == "build_cache")]
        build = {(r.dataset, r.arch): r.warm_median_s for r in bm.itertuples()}
    ds = _rev_datasets(s)
    fig, ax = plt.subplots(figsize=(_REV_W, 2.6))
    w = 0.38
    for k, arch in enumerate(("rgcn", "hgat")):
        for i, d in enumerate(ds):
            r = s[(s.dataset == d) & (s.arch == arch)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (k - 0.5) * w
            ax.bar(x, r.cache_MB, w, color=_REV_ARCH_C[arch])
            ax.bar(x, r.extras_MB, w, bottom=r.cache_MB, color="white",
                   edgecolor=_REV_ARCH_C[arch], hatch="////", lw=0.6)
            if (d, arch) in build:
                ax.text(x, (r.cache_MB + r.extras_MB) * 1.12, f"{build[(d, arch)]:.2g} s",
                        ha="center", va="bottom", rotation=90, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ds)), ds)
    ax.set_ylabel("cache size (MB)")
    ax.set_ylim(1, ax.get_ylim()[1] * 8)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=C[0], label="RGCN"), mp.Patch(color=C[3], label="hGAT"),
                       mp.Patch(facecolor="white", edgecolor="#555", hatch="////",
                                label="lazy extras")],
              frameon=False, ncol=3, loc="upper left")
    ax.set_title("Cache footprint at L=2 (label: in-memory build time)")
    return fig, "revision_memory_L2.pdf"


def fig_revision_latency_L2(exp="revision", fname="revision_latency_L2.pdf"):
    df = _rev_lat(exp)
    if df is None:
        return None
    d2 = df[df.layers == 2]
    ds = _rev_datasets(d2)
    fig, axes = plt.subplots(len(ds), 2, figsize=(_REV_W, 1.55 * len(ds) + 0.6),
                             squeeze=False, sharex=True)
    for i, d in enumerate(ds):
        for j, kind in enumerate(("feature", "topo")):
            ax = axes[i][j]
            sub = d2[(d2.dataset == d) & (d2.kind == kind)]
            for arch, col in _REV_ARCH_C.items():
                s = sub[sub.arch == arch].sort_values("pct")
                for m, ls in (("scratch", "-"), ("ours", "--")):
                    ax.plot(s.pct, s[f"{m}_median_ms"], ls, color=col, lw=1.5,
                            marker="o", ms=3)
                    ax.fill_between(s.pct, s[f"{m}_p10_ms"], s[f"{m}_p90_ms"],
                                    color=col, alpha=0.18, lw=0)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(_REV_KIND[kind])
            if j == 0:
                ax.set_ylabel(f"{d}\nms")
            if i == len(ds) - 1:
                ax.set_xlabel("update size (%)")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_ARCH_C[a], ls=ls, lw=1.5, label=f"{lab}-{_REV_ARCH[a]}")
         for a in ("rgcn", "hgat") for ls, lab in (("--", "INQ"), ("-", "scratch"))]
    fig.legend(handles=h, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    return fig, fname


def fig_revision_latency_layers(pct=10):
    df = _rev_lat()
    if df is None:
        return None
    dp = df[df.pct == pct]
    ds = _rev_datasets(dp)
    FLIP_C = {"rgcn": C[2], "hgat": "#8c564b"}
    fig, axes = plt.subplots(len(ds), 2, figsize=(_REV_W, 1.55 * len(ds) + 0.7),
                             squeeze=False, sharex=True)
    for i, d in enumerate(ds):
        for j, kind in enumerate(("feature", "topo")):
            axL = axes[i][j]
            axR = axL.twinx()
            sub = dp[(dp.dataset == d) & (dp.kind == kind)]
            for arch in ("rgcn", "hgat"):
                s = sub[sub.arch == arch].sort_values("layers")
                axL.plot(s.layers, s.speedup_measured, "-", color=_REV_ARCH_C[arch],
                         lw=1.5, marker="o", ms=3)
                axR.plot(s.layers, s.sign_flips.clip(lower=0.5), ":", color=FLIP_C[arch],
                         lw=1.3, marker="^", ms=3)
            axL.axhline(1.0, color="#999", lw=0.8, ls=":")
            axR.set_yscale("log")
            axR.spines["right"].set_visible(True)
            axL.set_xticks([1, 2, 3, 4])
            if i == 0:
                axL.set_title(_REV_KIND[kind])
            if j == 0:
                axL.set_ylabel(f"{d}\nspeedup")
            if j == 1:
                axR.set_ylabel("flips")
            if i == len(ds) - 1:
                axL.set_xlabel("layers L")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_ARCH_C[a], lw=1.5, marker="o", ms=3,
                   label=f"speedup {_REV_ARCH[a]}") for a in ("rgcn", "hgat")]
    h += [ml.Line2D([], [], color=FLIP_C[a], ls=":", lw=1.3, marker="^", ms=3,
                    label=f"flips {_REV_ARCH[a]}") for a in ("rgcn", "hgat")]
    fig.legend(handles=h, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    return fig, "revision_latency_layers.pdf"


def fig_revision_breakeven(frac=0.10):
    be = _load("revision", "breakeven.csv")
    if be is None:
        return None
    be = be[(be.frac - frac).abs() < 1e-9]
    ds = _rev_datasets(be)
    backends = [b for b in ("memory", "neo4j", "memgraph") if b in set(be.backend)]
    BC = {"memory": C[0], "neo4j": C[1], "memgraph": C[2]}
    MK = {"feature": "o", "topo": "s"}
    finite = be.k_star[be.k_star != float("inf")]
    top = (finite.max() if len(finite) else 10) * 4
    fig, axes = plt.subplots(2, 1, figsize=(_REV_W, 4.2), sharex=True)
    for ax, arch in zip(axes, ("rgcn", "hgat")):
        for bi, b in enumerate(backends):
            for ki, kind in enumerate(("feature", "topo")):
                s = be[(be.arch == arch) & (be.backend == b) & (be.kind == kind)]
                for r in s.itertuples():
                    x = ds.index(r.dataset) + (bi - (len(backends) - 1) / 2) * 0.22 \
                        + (ki - 0.5) * 0.08
                    inf = r.k_star == float("inf")
                    ax.scatter(x, top if inf else r.k_star, marker="x" if inf else MK[kind],
                               color=BC[b], s=28, zorder=3,
                               facecolors=BC[b] if inf or kind == "feature" else "none")
        ax.axhline(top, color="#999", lw=0.8, ls=":")
        ax.text(len(ds) - 0.5, top, "no break-even", ha="right", va="bottom", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(0.7, top * 3)
        ax.set_ylabel(f"{_REV_ARCH[arch]}: k*")
    axes[-1].set_xticks(range(len(ds)), ds)
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=BC[b], marker="o", ls="", label=b) for b in backends]
    h += [ml.Line2D([], [], color="#555", marker="o", ls="", label="attribute Δ"),
          ml.Line2D([], [], color="#555", marker="s", mfc="none", ls="", label="structure Δ")]
    fig.suptitle(f"Break-even queries k* at {int(frac * 100)}%, L=2", fontsize=9)
    fig.legend(handles=h, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_figheight()))
    return fig, "revision_breakeven.pdf"


def fig_revision_speedup_vs_closure():
    df = _rev_lat()
    if df is None:
        return None
    MK = {"feature": "o", "topo": "s"}
    fig, ax = plt.subplots(figsize=(_REV_W, 2.8))
    for d in _rev_datasets(df):
        for kind, m in MK.items():
            s = df[(df.dataset == d) & (df.kind == kind)]
            ax.scatter(s.closure_frac, s.speedup_measured, marker=m, s=12,
                       color=_REV_DS_C[d], alpha=0.75,
                       facecolors=_REV_DS_C[d] if kind == "feature" else "none")
    ax.axhline(1.0, color="#999", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"closure fraction $|S_L|/|V|$")
    ax.set_ylabel("measured speedup")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_DS_C[d], marker="o", ls="", label=d)
         for d in _rev_datasets(df)]
    h += [ml.Line2D([], [], color="#555", marker="o", ls="", label="attribute"),
          ml.Line2D([], [], color="#555", marker="s", mfc="none", ls="", label="structure")]
    fig.legend(handles=h, frameon=False, ncol=3, loc="upper center",
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    return fig, "revision_speedup_vs_closure.pdf"


def fig_revision_rows_vs_flips():
    df = _rev_lat()
    if df is None:
        return None
    f = df[df.kind == "feature"]
    MK = {"rgcn": "o", "hgat": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(_REV_W, 2.5), sharey=True)
    for d in _rev_datasets(f):
        for arch, m in MK.items():
            s = f[(f.dataset == d) & (f.arch == arch)]
            axes[0].scatter(s.rows_dirty_sum, s.ours_median_ms, marker=m, s=12,
                            color=_REV_DS_C[d], alpha=0.75)
            s = s[s.sign_flips > 0]       # L=1 has no hidden ReLU -> 0 flips
            axes[1].scatter(s.sign_flips, s.ours_median_ms, marker=m, s=12,
                            color=_REV_DS_C[d], alpha=0.75)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
    axes[0].set_xlabel("affected rows\n(sum over layers)")
    axes[1].set_xlabel("sign flips")
    axes[0].set_ylabel("INQ measured (ms)")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_DS_C[d], marker="o", ls="", label=d)
         for d in _rev_datasets(f)]
    h += [ml.Line2D([], [], color="#555", marker=MK[a], ls="", label=_REV_ARCH[a])
          for a in MK]
    fig.legend(handles=h, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    return fig, "revision_rows_vs_flips.pdf"


def fig_v2_scaling_L2_latency():
    """Original scaling_L2_latency, from MEASURED v2 medians (p10-p90 band)."""
    return fig_revision_latency_L2("revision_v2", "v2_scaling_L2_latency.pdf")


def fig_v2_scaling_L2_operations():
    """Original scaling_L2_operations (FLOP proxy; v2 performs the same
    operations as v1, so the proxy is unchanged) in the single-column layout."""
    df = _scaling_df()
    if df is None:
        return None
    d2 = df[df.layers == 2]
    ds = [d for d in _REV_DS if d in set(d2.dataset)]
    fig, axes = plt.subplots(len(ds), 2, figsize=(_REV_W, 1.55 * len(ds) + 0.6),
                             squeeze=False, sharex=True)
    for i, d in enumerate(ds):
        for j, kind in enumerate(("feature", "topo")):
            ax = axes[i][j]
            sub = d2[(d2.dataset == d) & (d2.kind == kind)]
            for arch, col in _REV_ARCH_C.items():
                for m, ls in (("from_scratch", "-"), ("ours", "--")):
                    s = sub[(sub.arch == arch) & (sub.method == m)].sort_values("pct")
                    ax.plot(s.pct, s.flops, ls, color=col, lw=1.5, marker="o", ms=3)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(_REV_KIND[kind])
            if j == 0:
                ax.set_ylabel(f"{d}\nFLOPs (proxy)")
            if i == len(ds) - 1:
                ax.set_xlabel("update size (%)")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_ARCH_C[a], ls=ls, lw=1.5, label=f"{lab}-{_REV_ARCH[a]}")
         for a in ("rgcn", "hgat") for ls, lab in (("--", "INQ"), ("-", "scratch"))]
    fig.legend(handles=h, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    return fig, "v2_scaling_L2_operations.pdf"


def fig_v2_scaling_L2_memory(exp="revision_v2", fname="v2_scaling_L2_memory.png"):
    """Memory row of the revised Fig. 3, in the ORIGINAL wide layout of
    fig_scaling_L2 (2 rows = attribute / structure, 4 columns = datasets):
    peak memory of ONE query (MB, log) vs update size at L=2, INQ vs from-scratch,
    both archs; dotted grey = cache size (identical for both archs). Peak =
    maxrss after the query minus RSS before it (`peak_over_current_MB`, median
    over 5 fresh processes). Medians only, as in the original figure."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    import matplotlib.lines as ml
    q = _load(exp, "query_memory.csv")
    c = _load(exp, "cache_size_summary.csv")
    if q is None or c is None:
        return None
    q = q[q.layers == 2].copy()
    q["pct"] = (q.frac * 100).round().astype(int)
    cache = c[(c.L == 2) & (c.arch == "rgcn")].set_index("dataset")["cache_MB"]
    kinds = ["feature", "topo"]
    klab = {"feature": "attribute", "topo": "structure"}
    datasets = [x for x in _SCALE_DATASETS if x in set(q.dataset.unique())]
    fmt = lambda v, _: (f"{v:,.0f}" if v >= 100 else f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    fig, axes = plt.subplots(len(kinds), len(datasets),
                             figsize=(4.2 * len(datasets), 2.7 * len(kinds)), squeeze=False)
    arch_color = {"rgcn": C[0], "hgat": C[3]}
    method_ls = {"from_scratch": "-", "ours": "--"}
    for i, kind in enumerate(kinds):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            sub = q[(q.dataset == ds) & (q.kind == kind)]
            yvals = []
            cache_note = ""
            for method, ls in method_ls.items():
                for arch, color in arch_color.items():
                    s_ = sub[(sub.method == method) & (sub.arch == arch)].sort_values("pct")
                    if s_.empty:
                        continue
                    ax.plot(s_.pct, s_.peak_over_current_MB, ls, color=color, lw=2,
                            marker="o", ms=4)
                    yvals.extend(s_.peak_over_current_MB.tolist())
            if ds in cache.index and yvals:
                cv = float(cache[ds])
                if min(yvals) / 6 <= cv <= max(yvals) * 6:
                    ax.axhline(cv, color="0.45", ls=":", lw=1.6)
                    yvals.append(cv)
                else:
                    # far outside the query range (faers: 25 MB cache vs ~2 GB queries);
                    # a line there would crush the curves, so it goes into the title.
                    cache_note = f" (cache {cv:,.0f} MB)"
            ax.set_yscale("log")
            if yvals:
                ticks = []
                thresh = 1.04 if (max(yvals) / min(yvals) < 2) else 1.11
                for v in sorted(set(round(v, 1) for v in yvals)):
                    if not ticks or v > ticks[-1] * thresh:
                        ticks.append(v)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(FuncFormatter(fmt))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis="y", which="major", labelsize=5.5)
                ax.tick_params(axis="y", which="minor", length=0)
            ax.grid(True, which="major", lw=0.3, alpha=0.5)
            if i == 0:
                ax.set_title(ds + cache_note)
            if i == len(kinds) - 1:
                ax.set_xlabel("update size")
            if j == 0:
                ax.set_ylabel(f"{klab[kind]} Δ\npeak memory (MB)", fontsize=8)
    h = [ml.Line2D([], [], color=C[0], ls="--", lw=2, marker="o", ms=4, label="INQ-RGCN"),
         ml.Line2D([], [], color=C[0], ls="-", lw=2, marker="o", ms=4, label="from-scratch-RGCN"),
         ml.Line2D([], [], color=C[3], ls="--", lw=2, marker="o", ms=4, label="INQ-hGAT"),
         ml.Line2D([], [], color=C[3], ls="-", lw=2, marker="o", ms=4, label="from-scratch-hGAT"),
         ml.Line2D([], [], color="0.45", ls=":", lw=1.6, label="cache size")]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Peak memory comparison between from-scratch and INQ evaluation",
                 y=1.03, fontsize=10)
    fig.tight_layout()
    return fig, fname

def _v2_speedup_flips(kind, pct=10):
    df = _rev_lat("revision_v2")
    if df is None:
        return None
    dp = df[(df.pct == pct) & (df.kind == kind)]
    ds = [d for d in _REV_DS if d in set(dp.dataset)]
    FLIP_C = {"rgcn": C[2], "hgat": "#8c564b"}
    fig, axes = plt.subplots(2, 2, figsize=(_REV_W, 3.9), squeeze=False)
    for k, d in enumerate(ds):
        axL = axes[k // 2][k % 2]
        axR = axL.twinx()
        sub = dp[dp.dataset == d]
        for arch in ("rgcn", "hgat"):
            s = sub[sub.arch == arch].sort_values("layers")
            axL.plot(s.layers, s.speedup_measured, "-", color=_REV_ARCH_C[arch], lw=1.5,
                     marker="o", ms=3)
            axR.plot(s.layers, s.sign_flips.clip(lower=0.5), ":", color=FLIP_C[arch],
                     lw=1.3, marker="^", ms=3)
        axL.axhline(1.0, color="#999", lw=0.8, ls=":")
        axR.set_yscale("log")
        axR.spines["right"].set_visible(True)
        axL.set_xticks([1, 2, 3, 4])
        axL.set_title(d)
        if k % 2 == 0:
            axL.set_ylabel("measured speedup")
        else:
            axR.set_ylabel("sign flips")
        if k // 2 == 1:
            axL.set_xlabel("layers L")
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=_REV_ARCH_C[a], lw=1.5, marker="o", ms=3,
                   label=f"speedup {_REV_ARCH[a]}") for a in ("rgcn", "hgat")]
    h += [ml.Line2D([], [], color=FLIP_C[a], ls=":", lw=1.3, marker="^", ms=3,
                    label=f"flips {_REV_ARCH[a]}") for a in ("rgcn", "hgat")]
    fig.legend(handles=h, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    return fig, f"v2_scaling_speedup_flips_{kind}_{pct}pct.pdf"


def fig_v2_speedup_flips_feature():
    return _v2_speedup_flips("feature")


def fig_v2_speedup_flips_topo():
    return _v2_speedup_flips("topo")


def fig_v2_speedup_flips_eicu(pct=10):
    """eICU only, structure | attribute delta side by side, in the layout and
    panel size of the v1 scaling_speedup_flips_*.png (4.4 x 2.5 per panel), with
    the measured v2 speedup instead of the latency_best_ms projection."""
    import matplotlib.lines as ml
    df = _rev_lat("revision_v2")
    if df is None:
        return None
    BROWN = "#8c564b"
    SPD_C, FLIP_C = {"rgcn": C[0], "hgat": C[3]}, {"rgcn": C[2], "hgat": BROWN}
    kinds = {"topo": "structure", "feature": "attribute"}
    fig, axes = plt.subplots(1, 2, figsize=(4.4 * 2, 2.5), squeeze=False)
    for j, (kind, label) in enumerate(kinds.items()):
        axL = axes[0][j]
        axR = axL.twinx()
        sub = df[(df.pct == pct) & (df.kind == kind) & (df.dataset == "eicu")]
        for arch in ("rgcn", "hgat"):
            s = sub[sub.arch == arch].sort_values("layers")
            axL.plot(s.layers, s.speedup_measured, "-", color=SPD_C[arch], lw=2,
                     marker="o", ms=4)
            axR.plot(s.layers, s.sign_flips.clip(lower=0.5), "-", color=FLIP_C[arch],
                     lw=1.6, marker="^", ms=5, alpha=0.9)
        axL.set_xticks([1, 2, 3, 4]); axL.set_xlabel("layers L")
        axL.axhline(1.0, color="#999", lw=0.8, ls=":")
        axR.set_yscale("log")
        axR.spines["right"].set_visible(True)
        axL.set_title(f"eicu, {label} delta")
        if j == 0:
            axL.set_ylabel("measured speedup")
        else:
            axR.set_ylabel("sign-flips")
    h = [ml.Line2D([], [], color=C[0], ls="-", lw=2, marker="o", ms=4, label="INQ-RGCN"),
         ml.Line2D([], [], color=C[3], ls="-", lw=2, marker="o", ms=4, label="INQ-hGAT"),
         ml.Line2D([], [], color=C[2], ls="-", lw=2, marker="^", ms=5, label="RGCN sign-flips"),
         ml.Line2D([], [], color=BROWN, ls="-", lw=2, marker="^", ms=5, label="hGAT sign-flips")]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    return fig, f"v2_scaling_speedup_flips_eicu_{pct}pct.pdf"


V2 = (fig_v2_scaling_L2_latency, fig_v2_scaling_L2_operations, fig_v2_scaling_L2_memory,
      fig_v2_speedup_flips_feature, fig_v2_speedup_flips_topo, fig_v2_speedup_flips_eicu)


REVISION = (fig_revision_memory, fig_revision_latency_L2, fig_revision_latency_layers,
            fig_revision_breakeven, fig_revision_speedup_vs_closure,
            fig_revision_rows_vs_flips)


def main_revision(fns=REVISION, tag="figures_revision"):
    os.makedirs(FIGURES, exist_ok=True)
    made = 0
    with plt.rc_context(_REV_RC):
        for fn in fns:
            out = fn()
            if out is None:
                print(f"skip {fn.__name__}: results not present yet")
                continue
            fig, name = out
            fig.savefig(os.path.join(FIGURES, name))
            plt.close(fig)
            made += 1
            print(f"wrote figures/{name}")
    print(f"{'PASS' if made else 'FAIL'} {tag}: {made} written")
    return 0 if made else 1


# ------------------------------------------------------------------ sessions
# R2.O2 (results/sessions): x = states visited k, one panel per dataset,
# colour = arch (RGCN C[0], hGAT C[3]), ours dashed, from-scratch solid.
_SESS_ARCH_C = {"rgcn": C[0], "hgat": C[3]}


def _sess(exp):
    df = _load(exp, "states.csv")
    if df is None:
        return None, None
    if (df.exact_pass == False).any():          # exactness gate: never plot a failure
        raise SystemExit(f"{exp}: states.csv has exactness failures; not plotting")
    df["k"] = df.state_idx + 1
    return df, _load(exp, "build.csv")


def _sess_panels(df, nrows=1):
    ds = [d for d in _SCALE_DATASETS if d in set(df.dataset)]
    fig, axes = plt.subplots(nrows, len(ds), figsize=(4.2 * len(ds), 2.7 * nrows),
                             squeeze=False)
    return ds, fig, axes


def _sess_legend(fig, handles):
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=min(len(handles), 4),
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.18 * ((len(handles) + 3) // 4)
                           / fig.get_figheight()))


def _pts(x, ks):
    return x if ks is None else x[x.k.isin(ks)]


def _crossing(k, ours, base):
    """Smallest k with cumulative ours <= cumulative baseline (None if never)."""
    hit = [kk for kk, o, b in zip(k, ours, base) if o <= b]
    return hit[0] if hit else None


def fig_sessions_time(exp="sessions", ks=None):
    import matplotlib.lines as ml
    df, b = _sess(exp)
    if df is None:
        return None
    cfg0 = df[df.config_idx == 0]
    ds, fig, axes = _sess_panels(cfg0)
    kstar = []
    for j, d in enumerate(ds):
        ax = axes[0][j]
        for arch, col in _SESS_ARCH_C.items():
            o = cfg0[(cfg0.dataset == d) & (cfg0.arch == arch) & (cfg0.method == "ours")
                     & (cfg0.policy.astype(str) == "all")].sort_values("k")
            s = cfg0[(cfg0.dataset == d) & (cfg0.arch == arch)
                     & (cfg0.method == "scratch")].sort_values("k")
            if o.empty or s.empty:
                continue
            b0 = float(o.build_s.iloc[0])
            mk = dict(marker="." if ks else None, ms=5)
            op, sp = _pts(o, ks), _pts(s, ks)
            ax.plot([0] + op.k.tolist(), [b0] + op.cum_s.tolist(), "--", color=col, lw=2, **mk)
            ax.plot([0] + sp.k.tolist(), [0] + sp.cum_s.tolist(), "-", color=col, lw=2, **mk)
            k = _crossing(o.k, o.cum_s, s.cum_s)
            kstar.append({"dataset": d, "arch": arch, "backend": "memory", "k_star": k or "",
                          "n_states": len(o), "note": "" if k else f"> {len(o)}"})
            if k:
                ax.plot(k, float(o.cum_s[o.k == k].iloc[0]), "o", color=col, ms=6,
                        mfc="white", mew=1.5)
            st = cfg0[(cfg0.dataset == d) & (cfg0.arch == arch)
                      & (cfg0.method == "store_scratch")].sort_values("k")
            if b is not None and not st.empty:
                bb = b[(b.dataset == d) & (b.arch == arch) & (b.backend == "neo4j")]
                if not bb.empty:
                    one = float(bb.download_s.iloc[0] + bb.tensors_s.iloc[0] + bb.build_s.iloc[0])
                    oo = o[o.k <= st.k.max()]
                    ours_st = one + (oo.cum_s - b0)
                    keep = oo.k.isin(ks) if ks else oo.k > 0
                    ax.plot([0] + oo.k[keep].tolist(), [one] + ours_st[keep].tolist(), "-.",
                            color=col, lw=1.5, **mk)
                    stp = _pts(st, ks)
                    ax.plot([0] + stp.k.tolist(), [0] + stp.cum_s.tolist(), ":", color=col,
                            lw=1.5, **mk)
                    k = _crossing(oo.k, ours_st, st.cum_s)
                    kstar.append({"dataset": d, "arch": arch, "backend": "neo4j",
                                  "k_star": k or "", "n_states": len(st),
                                  "note": "" if k else f"> {len(st)}"})
                    if k:
                        ax.plot(k, float(ours_st[oo.k == k].iloc[0]), "s", color=col, ms=5,
                                mfc="white", mew=1.5)
        ax.set_yscale("symlog", linthresh=0.1)
        if ks:
            ax.set_xlim(0, max(ks) * 1.03)
        ax.set_title(d)
        ax.set_xlabel("states visited k")
        if j == 0:
            ax.set_ylabel("cumulative time (s)")
    if ks is None:      # k* from every state, written by the full figure only
        pd.DataFrame(kstar).to_csv(os.path.join(RESULTS, exp, "kstar.csv"), index=False)
    h = [ml.Line2D([], [], color=C[0], ls="--", lw=2, label="INQ-RGCN (retain all)"),
         ml.Line2D([], [], color=C[0], ls="-", lw=2, label="from-scratch-RGCN"),
         ml.Line2D([], [], color=C[3], ls="--", lw=2, label="INQ-hGAT (retain all)"),
         ml.Line2D([], [], color=C[3], ls="-", lw=2, label="from-scratch-hGAT"),
         ml.Line2D([], [], color="#555", ls="-.", lw=1.5, label="INQ, store-backed (Neo4j)"),
         ml.Line2D([], [], color="#555", ls=":", lw=1.5, label="from-scratch, store-backed"),
         ml.Line2D([], [], color="#555", ls="", marker="o", mfc="white", label="k* (crossing)")]
    _sess_legend(fig, h)
    return fig, "sessions_time.pdf"


def fig_sessions_depth(exp="sessions", ks=None):
    import matplotlib.lines as ml
    df, _ = _sess(exp)
    if df is None:
        return None
    deep = df[df.config_idx == 2]
    if ks:
        deep = deep[deep.k <= max(ks)]
    if deep.empty:
        return None
    ds, fig, axes = _sess_panels(deep, 2)
    pol_ls = {"all": "--", "1": (0, (5, 1, 1, 1))}
    for j, d in enumerate(ds):
        for arch, col in _SESS_ARCH_C.items():
            sub = deep[(deep.dataset == d) & (deep.arch == arch)]
            for pol, ls in pol_ls.items():
                o = sub[(sub.method == "ours") & (sub.policy.astype(str) == pol)]
                if o.empty:
                    continue
                g = o.groupby("depth")
                med = g.wall_s.median() * 1e3
                axes[0][j].plot(med.index, med, ls=ls, color=col, lw=2)
                axes[0][j].fill_between(med.index, g.wall_s.quantile(0.1) * 1e3,
                                        g.wall_s.quantile(0.9) * 1e3, color=col, alpha=0.12,
                                        lw=0)
                axes[1][j].plot(med.index, g.sign_flips.median(), ls=ls, color=col, lw=2)
            s = sub[sub.method == "scratch"]
            if not s.empty:
                m = s.groupby("depth").wall_s.median() * 1e3
                axes[0][j].plot(m.index, m, "-", color=col, lw=2)
        axes[0][j].set_yscale("log")
        axes[0][j].set_title(d)
        axes[1][j].set_xlabel("path depth |D|")
        axes[1][j].set_yscale("symlog", linthresh=1)
    axes[0][0].set_ylabel("per-query latency (ms)")
    axes[1][0].set_ylabel("sign flips")
    h = [ml.Line2D([], [], color=C[0], ls="--", lw=2, label="INQ-RGCN retain all"),
         ml.Line2D([], [], color=C[0], ls=pol_ls["1"], lw=2, label="INQ-RGCN base only"),
         ml.Line2D([], [], color=C[0], ls="-", lw=2, label="from-scratch-RGCN"),
         ml.Line2D([], [], color=C[3], ls="--", lw=2, label="INQ-hGAT retain all"),
         ml.Line2D([], [], color=C[3], ls=pol_ls["1"], lw=2, label="INQ-hGAT base only"),
         ml.Line2D([], [], color=C[3], ls="-", lw=2, label="from-scratch-hGAT")]
    _sess_legend(fig, h)
    return fig, "sessions_depth.pdf"


def fig_sessions_memory(exp="sessions", ks=None):
    """Memory held at k, both sides counted in full (RGCN; hGAT within ~2%).
    Ours: the session starts from the base graph in the DB (Neo4j, as in the
    store-backed runs), downloads it once and builds cache + index: k = 0 is
    that peak (db + graph + cache + index). After it the DB copy and the edge
    lists are not read again: node features + cache + index + retained states. From-scratch on a DB: the
    stored graph + the log of the edits so far + per query the downloaded graph
    and one forward pass (the base cache without the index)."""
    import matplotlib.lines as ml
    import numpy as np
    df, b = _sess(exp)
    if df is None:
        return None
    sz = _load("revision_v2", "store_size.csv")
    gsz = _load(exp, "graph_size.csv")
    if sz is None or gsz is None or b is None:
        return None
    cfg0 = df[(df.config_idx == 0) & (df.arch == "rgcn")]
    ds, fig, axes = _sess_panels(cfg0)
    pol_ls = {"all": "--", "16": ":", "1": (0, (5, 1, 1, 1))}
    db_c = {"neo4j": "#333333", "memgraph": "#999999"}
    MB = 2 ** 20
    for j, d in enumerate(ds):
        ax = axes[0][j]
        sub = cfg0[cfg0.dataset == d]
        g = gsz[gsz.dataset == d].iloc[0]
        ix = float(b[(b.dataset == d) & (b.arch == "rgcn")
                     & (b.backend == "memory")].index_MB.iloc[0])
        base = sub[(sub.method == "ours") & (sub.policy.astype(str) == "1")]
        cache0 = base.retained_bytes.min() / MB             # base cache + index
        neo = sz[(sz.dataset == d) & (sz.backend == "neo4j")].iloc[0]
        peak = neo.store_MB + g.x_MB + g.edges_MB + cache0  # build: db + downloaded graph
        # every line as a full series over k = 0..n (crossings need every state)
        ref = sub[sub.method == "scratch"].sort_values("k")
        k = np.r_[0, ref.k.values]
        ours = {pol: np.r_[peak, g.x_MB + sub[(sub.method == "ours")
                                              & (sub.policy.astype(str) == pol)]
                           .sort_values("k").retained_bytes.values / MB]
                for pol in pol_ls}
        scr = {}
        for be in db_c:
            r = sz[(sz.dataset == d) & (sz.backend == be)].iloc[0]
            # k = 0: what the first query needs (db + downloaded graph + GNN), empty log
            scr[be] = (r.store_MB + g.x_MB + g.edges_MB + cache0 - ix
                       + np.r_[0.0, r.log_bytes / r.edges * ref.edges_applied_cum.values / MB])
        # crossings per (policy, DB): the drop at k = 1 once the build is done, then
        # the first crossing after it; later wobbles around the line are not marked
        cross, kmax = [], max(ks) if ks else k[-1]
        for pol, o in ours.items():
            for be, sc in scr.items():
                above = o > sc
                flips = [i for i in range(1, len(k)) if above[i] != above[i - 1]
                         and k[i] <= kmax]
                mark = [i for i in flips if i == 1][:1] + [i for i in flips if i > 1][:1]
                cross += [(i, o[i], db_c[be]) for i in mark]
        sel = np.ones(len(k), bool) if ks is None else np.isin(k, (0, *ks, *(c[0] for c in cross)))
        for pol, ls in pol_ls.items():
            ax.plot(k[sel], ours[pol][sel], ls=ls, color=C[0], lw=2)
        for be, colr in db_c.items():
            ax.plot(k[sel], scr[be][sel], "-", color=colr, lw=2)
        if ks:
            grid = np.isin(k, (0, *ks))
            for o in (*ours.values(), *scr.values()):
                ax.plot(k[grid], o[grid], ".", color="#666", ms=3)
        # crossing k in red next to the crossing; on the same DB line, ks closer than
        # 8% of the axis share one label ("6,7")
        groups = []
        for i, y, colr in sorted(cross, key=lambda c: (c[2], c[0])):
            g_ = groups[-1] if groups else None
            if g_ and g_["c"] == colr and k[i] - g_["k"][-1] < 0.08 * kmax:
                g_["k"].append(int(k[i])); g_["y"].append(y)
            elif not g_ or int(k[i]) not in g_["k"] or g_["c"] != colr:
                groups.append({"c": colr, "k": [int(k[i])], "y": [y]})
        for g_ in groups:
            ax.annotate(",".join(map(str, dict.fromkeys(g_["k"]))),
                        (np.mean(g_["k"]), np.mean(g_["y"])), xytext=(4, 5),
                        textcoords="offset points", fontsize=7, color="#c0392b",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax.set_yscale("linear" if ks else "log")
        if ks:
            ax.set_xlim(-2, max(ks) * 1.03)
            ax.set_ylim(bottom=0)
        ax.set_title(d)
        ax.set_xlabel("states visited k")
        if j == 0:
            ax.set_ylabel("memory held (MB)")
    h = [ml.Line2D([], [], color=C[0], ls=pol_ls["all"], lw=2, label="INQ retain all"),
         ml.Line2D([], [], color=C[0], ls=pol_ls["16"], lw=2, label="INQ retain 16"),
         ml.Line2D([], [], color=C[0], ls=pol_ls["1"], lw=2, label="INQ base only"),
         ml.Line2D([], [], color=db_c["neo4j"], lw=2,
                   label="from-scratch on Neo4j (db + log + graph + GNN)"),
         ml.Line2D([], [], color=db_c["memgraph"], lw=2,
                   label="from-scratch on Memgraph (db + WAL + graph + GNN)")]
    _sess_legend(fig, h)
    return fig, "sessions_memory.pdf"


SESSIONS = (fig_sessions_time, fig_sessions_depth, fig_sessions_memory)


SESSIONS_K100 = (*range(5, 31, 5), 50, 75, 100)   # coarse prefix grid, k <= 100


def main_sessions(exp="sessions", ks=None):
    os.makedirs(FIGURES, exist_ok=True)
    made = 0
    for fn in SESSIONS:
        out = fn(exp, ks)
        if out is None:
            print(f"skip {fn.__name__}: results not present yet")
            continue
        fig, name = out
        name = name if exp == "sessions" else f"{exp}_{name}"
        if ks:
            name = name.replace(".pdf", f"_k{max(ks)}.pdf")
        fig.savefig(os.path.join(FIGURES, name))
        plt.close(fig)
        made += 1
        print(f"wrote figures/{name}")
    print(f"{'PASS' if made else 'FAIL'} figures_sessions: {made} written")
    return 0 if made else 1


def main():
    import sys
    if "--revision" in sys.argv:
        return main_revision()
    if "--v2" in sys.argv:
        return main_revision(V2, "figures_v2")
    if "--sessions-k100" in sys.argv:
        return main_sessions(ks=SESSIONS_K100)
    if "--sessions" in sys.argv:
        return main_sessions()
    os.makedirs(FIGURES, exist_ok=True)
    made = []
    # functions returning one (fig, name), or a list of them
    single = (fig_overhead, fig_single_query, fig_single_query_best,
              fig_single_query_operations, fig_signflips_combined,
              fig_baselines_exactness,
              fig_timing_model, fig_layer_scaling, fig_exploration,
              fig_exploration_totals)
    multi = (fig_grid_latency_all, fig_grid_operations_all,
             fig_baselines_operations, fig_baselines_latency,
             fig_db_roundtrip, fig_db_cumulative, fig_db_pipeline, fig_db_finbench_2x2,
             fig_db_full_appendix,
             fig_scaling_L2, fig_scaling_layers, fig_scaling_signflips,
             fig_scaling_speedup_flips)
    outs = []
    for fn in single:
        out = fn()
        if out is None:
            print(f"skip {fn.__name__}: results not present yet")
        else:
            outs.append(out)
    for fn in multi:
        got = fn()
        if not got:
            print(f"skip {fn.__name__}: results not present yet")
        else:
            outs.extend(got)
    for fig, name in outs:
        path = os.path.join(FIGURES, name)
        fig.savefig(path)
        plt.close(fig)
        made.append(name)
        print(f"wrote figures/{name}")
    print(f"PASS figures: {len(made)} written" if made
          else "FAIL figures: nothing to plot")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
