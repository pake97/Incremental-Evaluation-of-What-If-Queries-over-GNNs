"""Shared harness for the forward-counterfactual baselines (see baselines.md).

Every method (ours, Ripple, NeutronRT, from-scratch) plugs into `evaluate` and
increments the SAME global `Counters` at the SAME granularity, so operation
counts are directly comparable — the clean signal, since CPU wall-clock is noisy.

An `edge` op = one per-relation edge message lin_msg(h_u) (or attention message)
computed and aggregated. A `vert` op = one node embedding (aggregate + self +
activation) computed. From-scratch therefore costs |E_directed|*L edges and
|V|*L verts; incremental methods cost only what they touch.

Rollback is free here: our interventions are copy-on-write (GraphState never
mutates the base graph or the cached tensors), so snapshot/restore is a no-op —
the branch-evaluate-drop loop is uniform for all methods by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from src.data.graph_state import GraphState
from src.models.model import _key

EdgeType = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
class Counters:
    """Global op counter. All methods call add_edges/add_verts so the counts
    are defined at one granularity."""
    edges = 0
    verts = 0

    @classmethod
    def reset(cls):
        cls.edges = 0
        cls.verts = 0

    @classmethod
    def add_edges(cls, n):
        cls.edges += int(n)

    @classmethod
    def add_verts(cls, n):
        cls.verts += int(n)


# ---------------------------------------------------------------------------
# Recording cache: base per-relation message sums + degree + per-layer state
# ---------------------------------------------------------------------------
class BaselineCache:
    """Full forward on the base graph, recording everything the linear-delta
    baselines need to reuse:

      H[l][nt]      post-activation embeddings; H[0] = proj(x)   (l = 0..L)
      num[l][et]    per-relation UN-normalized message sum at dst  (l = 0..L-1)
      deg[et]       per-relation in-degree at dst (topology-fixed) clamp 1e-8
      att[l][et]    hGAT per-edge attention weight alpha (l = 0..L-1)
      att_sum[l][et] hGAT per-dst attention normalizer (softmax denom)
    """

    def __init__(self, model, gs: GraphState):
        self.model = model
        self.enc = model.encoder
        self.gs = gs
        self.mt = model.cfg["model_type"]
        self.L = model.cfg["num_layers"]
        self.node_types = list(self.enc.node_types)
        self.all_types = list(self.enc.all_types)
        self.N = {nt: gs.num_nodes[nt] for nt in self.node_types}
        self._ei = self._build_ei(gs.edge_index)
        self._record()

    def _build_ei(self, ei_dict) -> Dict[EdgeType, torch.Tensor]:
        out = {}
        for (s, r, d) in self.enc.fwd_types:
            ei = ei_dict[(s, r, d)]
            out[(s, r, d)] = ei
            out[(d, "rev_" + r, s)] = ei.flip(0)
        return out

    @torch.no_grad()
    def _record(self):
        enc = self.enc
        H = {nt: enc.proj[nt](self.gs.x[nt]) for nt in self.node_types}
        self.H = [H]
        self.num: List[Dict[EdgeType, torch.Tensor]] = []
        self.agg: List[Dict[EdgeType, torch.Tensor]] = []  # aggregated message (post-norm)
        self.att: List[Dict[EdgeType, torch.Tensor]] = []
        self.att_sum: List[Dict[EdgeType, torch.Tensor]] = []
        self.deg: Dict[EdgeType, torch.Tensor] = {}
        for et in self.all_types:
            d = et[2]
            ei = self._ei[et]
            ones = torch.ones(ei.size(1))
            self.deg[et] = torch.zeros(self.N[d]).index_add_(0, ei[1], ones).clamp(min=1e-8)

        h = H
        for li in range(self.L):
            layer = enc.layers[li]
            num_l, agg_l, att_l, asum_l = {}, {}, {}, {}
            acc = {nt: [] for nt in self.node_types}
            for et in self.all_types:
                s, _, d = et
                cell = layer[_key(et)]
                ei = self._ei[et]
                src, dst = ei[0], ei[1]
                if self.mt == "rgcn":
                    msg = cell.lin_msg(h[s])[src]
                    num = torch.zeros(self.N[d], enc.hidden).index_add_(0, dst, msg)
                    num_l[et] = num
                    agg = num / self.deg[et].unsqueeze(-1)
                    agg_l[et] = agg
                    acc[d].append(agg + cell.lin_self(h[d]))
                else:  # hgat
                    hs, hd = cell.lin(h[s]), cell.lin(h[d])
                    e = (hs * cell.att_src).sum(-1)[src] + (hd * cell.att_dst).sum(-1)[dst]
                    e = torch.nn.functional.leaky_relu(e, 0.2)
                    from torch_geometric.utils import softmax as sm
                    alpha = sm(e, dst, num_nodes=self.N[d])
                    att_l[et] = alpha
                    asum = torch.zeros(self.N[d]).index_add_(0, dst, torch.exp(e).clamp(min=1e-12))
                    asum_l[et] = asum
                    msg = hs[src] * alpha.unsqueeze(-1)
                    agg = torch.zeros(self.N[d], enc.hidden).index_add_(0, dst, msg)
                    agg_l[et] = agg
                    acc[d].append(agg + cell.lin_self(h[d]))
            self.num.append(num_l)
            self.agg.append(agg_l)
            self.att.append(att_l)
            self.att_sum.append(asum_l)
            h_new = {}
            for nt in self.node_types:
                z = sum(acc[nt]) if acc[nt] else h[nt]
                h_new[nt] = torch.relu(z) if li < self.L - 1 else z
            self.H.append(h_new)
            h = h_new

    @property
    def h_final(self):
        return self.H[-1]


# ---------------------------------------------------------------------------
# Metrics + evaluate
# ---------------------------------------------------------------------------
@dataclass
class Result:
    method: str
    wall_s: float
    edges: int
    verts: int
    mse: float
    max_abs: float
    affected_frac: float
    degenerate: bool = False


def _distance(h_hat, h_true) -> Tuple[float, float]:
    sq, n, mx = 0.0, 0, 0.0
    for nt in h_true:
        d = (h_hat[nt] - h_true[nt])
        sq += float((d * d).sum())
        n += d.numel()
        mx = max(mx, float(d.abs().max()) if d.numel() else 0.0)
    return sq / max(n, 1), mx


@torch.no_grad()
def evaluate(method, cache: BaselineCache, iv, h_true, affected_frac: float) -> Result:
    import time
    Counters.reset()
    t0 = time.perf_counter()
    h_hat = method.apply(cache, iv)
    wall = time.perf_counter() - t0
    mse, max_abs = _distance(h_hat, h_true)
    return Result(method.name, wall, Counters.edges, Counters.verts, mse, max_abs,
                  affected_frac, degenerate=affected_frac > 0.5)
