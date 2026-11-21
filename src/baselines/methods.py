"""The four baseline methods (see baselines.md Docs 1-2).

All operate on a BaselineCache and an Intervention, return the final-layer
embedding dict for G^Delta, and increment the shared Counters at the common
granularity (one edge op per message, one vert op per recomputed node).

  FromScratch  reference: full recompute (distance 0 by definition)
  Ours         our engine (delta for RGCN feature, closure otherwise)
  NeutronRT    RGCN: exact linear delta with NEW-degree renormalization;
               hGAT: exact constrained closure recompute
  Ripple       linear look-forward delta with FROZEN degree (RGCN) and FROZEN
               attention (hGAT): exact for RGCN+feature (static degree),
               nonzero distance for RGCN+topology (mean renormalization) and
               for hGAT (data-dependent attention) — the point of the baseline
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.baselines.harness import BaselineCache, Counters
from src.data.graph_state import GraphState, Intervention
from src.forward.forward import whatif_forward
from src.incremental.cache import build_cache
from src.incremental.closure import closure_forward
from src.incremental.engine import (feat_dirty_rows, incremental_forward,
                                    is_topological, struct_dirty_rows)
from src.models.model import _key

EdgeType = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# helpers shared by the linear-delta methods
# ---------------------------------------------------------------------------
def _topology_edits(cache: BaselineCache, iv: Intervention):
    """Per relation (fwd + derived rev): kept base edges (deletions removed),
    added edges, deleted edges, and net degree change at each dst."""
    edits = {}
    for (s, r, d) in cache.enc.fwd_types:
        fwd, rev = (s, r, d), (d, "rev_" + r, s)
        base = cache.gs.edge_index[fwd]
        del_cols = iv.del_cols.get(fwd)
        add = iv.add_edges.get(fwd)
        keep = torch.ones(base.size(1), dtype=torch.bool)
        if del_cols is not None and del_cols.numel():
            keep[del_cols] = False
        kept = base[:, keep]
        deleted = base[:, ~keep] if (del_cols is not None and del_cols.numel()) \
            else base[:, :0]
        added = add if (add is not None and add.numel()) else base[:, :0]
        for et, ei_kept, ei_add, ei_del in (
                (fwd, kept, added, deleted),
                (rev, kept.flip(0), added.flip(0), deleted.flip(0))):
            ddeg = torch.zeros(cache.N[et[2]])
            if ei_add.numel():
                ddeg.index_add_(0, ei_add[1], torch.ones(ei_add.size(1)))
            if ei_del.numel():
                ddeg.index_add_(0, ei_del[1], -torch.ones(ei_del.size(1)))
            edits[et] = (ei_kept, ei_add, ei_del, ddeg)
    return edits


def _linprop(cache: BaselineCache, iv: Intervention, degree_mode: str):
    """Shared linear-aggregation delta propagation.

    degree_mode: 'new' (NeutronRT, exact renormalization) or 'frozen' (Ripple).
    Attention (hGAT) is handled here as FROZEN (Ripple) via cached alpha; the
    exact hGAT path is NeutronRT's closure, not this function.
    """
    enc = cache.enc
    L, hid = cache.L, enc.hidden
    rgcn = cache.mt == "rgcn"
    edits = _topology_edits(cache, iv)

    dH = {nt: torch.zeros(cache.N[nt], hid) for nt in cache.node_types}
    for nt, delta in iv.feat_deltas.items():
        rows = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if rows.numel():
            dH[nt][rows] = F.linear(delta[rows], enc.proj[nt].weight)

    Hcur = {nt: cache.H[0][nt] + dH[nt] for nt in cache.node_types}
    for li in range(L):
        layer = enc.layers[li]
        # per-node-type accumulator of z' over affected rows
        dz = {nt: torch.zeros(cache.N[nt], hid) for nt in cache.node_types}
        touched = {nt: torch.zeros(cache.N[nt], dtype=torch.bool)
                   for nt in cache.node_types}

        for et in cache.all_types:
            s, _, d = et
            cell = layer[_key(et)]
            ei_kept, ei_add, ei_del, ddeg = edits[et]
            lin_msg = cell.lin_msg if rgcn else cell.lin

            dnum = torch.zeros(cache.N[d], hid)
            aff_dst = torch.zeros(cache.N[d], dtype=torch.bool)

            # (1) embedding-changed sources along kept base edges
            src_changed = (dH[s].abs().sum(-1) > 0)
            if ei_kept.numel() and bool(src_changed.any()):
                m = src_changed[ei_kept[0]]
                if bool(m.any()):
                    e = ei_kept[:, m]
                    if rgcn:
                        msg = lin_msg(dH[s][e[0]])
                    else:  # frozen attention on kept edges
                        alpha = _kept_alpha(cache, li, et, iv)[m]
                        msg = cell.lin(dH[s][e[0]]) * alpha.unsqueeze(-1)
                    dnum.index_add_(0, e[1], msg)
                    aff_dst[torch.unique(e[1])] = True
                    Counters.add_edges(e.size(1))
            # (2) added edges: full new message
            if ei_add.numel():
                if rgcn:
                    msg = lin_msg(Hcur[s][ei_add[0]])
                else:
                    alpha = _frozen_alpha_for(cache, li, et, ei_add, Hcur)
                    msg = cell.lin(Hcur[s][ei_add[0]]) * alpha.unsqueeze(-1)
                dnum.index_add_(0, ei_add[1], msg)
                aff_dst[torch.unique(ei_add[1])] = True
                Counters.add_edges(ei_add.size(1))
            # (3) deleted edges: remove base message
            if ei_del.numel():
                if rgcn:
                    msg = lin_msg(cache.H[li][s][ei_del[0]])
                else:
                    alpha = _frozen_alpha_for(cache, li, et, ei_del, cache.H[li])
                    msg = cell.lin(cache.H[li][s][ei_del[0]]) * alpha.unsqueeze(-1)
                dnum.index_add_(0, ei_del[1], -msg)
                aff_dst[torch.unique(ei_del[1])] = True
                Counters.add_edges(ei_del.size(1))

            # nodes whose self-term changed (dH[d] != 0) are also affected
            self_changed = (dH[d].abs().sum(-1) > 0)
            aff = aff_dst | self_changed
            if not bool(aff.any()):
                continue
            rows = aff.nonzero(as_tuple=False).squeeze(1)

            if rgcn:
                deg_base = cache.deg[et][rows]
                deg_eff = deg_base + ddeg[rows] if degree_mode == "new" else deg_base
                deg_eff = deg_eff.clamp(min=1e-8)
                agg_new = (cache.num[li][et][rows] + dnum[rows]) / deg_eff.unsqueeze(-1)
                agg_base = cache.agg[li][et][rows]
                dz[d][rows] += (agg_new - agg_base)
            else:
                dz[d][rows] += dnum[rows]  # frozen-attention aggregated delta
            # self term delta: weight only — the bias cancels in a delta
            dz[d][rows] += F.linear(dH[d][rows], cell.lin_self.weight)
            touched[d][rows] = True

        # form H'[li+1] on touched rows
        dH_next = {nt: torch.zeros(cache.N[nt], hid) for nt in cache.node_types}
        for nt in cache.node_types:
            if not bool(touched[nt].any()):
                continue
            rows = touched[nt].nonzero(as_tuple=False).squeeze(1)
            z_base = _z_base(cache, li, nt, rows)
            z_new = z_base + dz[nt][rows]
            h_new = F.relu(z_new) if li < L - 1 else z_new
            dH_next[nt][rows] = h_new - cache.H[li + 1][nt][rows]
            Counters.add_verts(rows.numel())
        dH = dH_next
        Hcur = {nt: cache.H[li + 1][nt] + dH[nt] for nt in cache.node_types}

    return {nt: cache.H[-1][nt] + dH[nt] for nt in cache.node_types}


def _z_base(cache, li, nt, rows):
    """Recompose base z (pre-activation) for the given dst rows from cached
    aggregates + self terms (avoids storing z explicitly)."""
    enc = cache.enc
    layer = enc.layers[li]
    z = torch.zeros(rows.numel(), enc.hidden)
    for et in cache.all_types:
        if et[2] != nt:
            continue
        cell = layer[_key(et)]
        z += cache.agg[li][et][rows] + cell.lin_self(cache.H[li][nt][rows])
    return z


# --- frozen-attention helpers (Ripple hGAT only) --------------------------
def _kept_alpha(cache, li, et, iv):
    """Base per-edge alpha for kept edges of relation et (deletions removed)."""
    (s, r, d) = et
    if r.startswith("rev_"):
        fwd = (d, r[4:], s)
    else:
        fwd = et
    del_cols = iv.del_cols.get(fwd)
    alpha = cache.att[li][et]
    if del_cols is not None and del_cols.numel():
        keep = torch.ones(alpha.size(0), dtype=torch.bool)
        keep[del_cols] = False
        return alpha[keep]
    return alpha


def _frozen_alpha_for(cache, li, et, ei, H):
    """Frozen alpha for a set of edges: exp(leaky(e)) / base att_sum[dst]."""
    s, _, d = et
    cell = cache.enc.layers[li][_key(et)]
    hs = cell.lin(H[s][ei[0]])
    hd = cell.lin(H[d][ei[1]])
    e = (hs * cell.att_src).sum(-1) + (hd * cell.att_dst).sum(-1)
    e = F.leaky_relu(e, 0.2)
    denom = cache.att_sum[li][et][ei[1]].clamp(min=1e-12)
    return torch.exp(e) / denom


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------
class FromScratch:
    name = "from_scratch"

    def __init__(self, model, gs):
        self.model, self.gs = model, gs
        self.L = model.cfg["num_layers"]
        self._edges = sum(t.size(1) for t in gs.edge_index.values()) * 2 * self.L
        self._verts = sum(gs.num_nodes[nt] for nt in gs.node_types) * self.L

    def apply(self, cache, iv):
        Counters.add_edges(self._edges)
        Counters.add_verts(self._verts)
        return whatif_forward(self.model, self.gs, iv)


class Ours:
    name = "ours"

    def __init__(self, model, gs, cache):
        self.model, self.gs, self._cache = model, gs, cache

    def apply(self, cache, iv):
        res = incremental_forward(self.model, self.gs, self._cache, iv)
        Counters.add_edges(res.stats["edges_touched"])
        Counters.add_verts(res.stats["nodes_recomputed"])
        return res.h


class NeutronRT:
    name = "neutronrt"

    def __init__(self, model, gs, cache, impl="reference"):
        """impl='optimised': same arithmetic on the v2 primitives (linprop_v2 for
        RGCN, the v2 closure for hGAT); `cache` must be the base ForwardCache."""
        self.model, self.gs, self._cache, self.impl = model, gs, cache, impl
        self.rgcn = model.cfg["model_type"] == "rgcn"

    def apply(self, cache, iv):
        if self.impl == "optimised":
            if self.rgcn:
                from src.baselines.linprop_v2 import linprop_v2
                from src.incremental.index import get_index
                return linprop_v2(cache, iv, "new", get_index(self._cache))
            res = incremental_forward(self.model, self.gs, self._cache, iv,
                                      method="closure", impl="v2")
            Counters.add_edges(res.stats["edges_touched"])
            Counters.add_verts(res.stats["nodes_recomputed"])
            return res.h
        if self.rgcn:
            return _linprop(cache, iv, degree_mode="new")
        # hGAT: exact constrained closure recompute
        view = self.gs.apply(iv)
        h, _, stats = closure_forward(
            self.model, self._cache, view,
            feat_dirty=feat_dirty_rows(iv),
            struct_dirty=struct_dirty_rows(self.gs, iv))
        Counters.add_edges(stats["edges_touched"])
        Counters.add_verts(stats["nodes_recomputed"])
        return h


class Ripple:
    name = "ripple"

    def __init__(self, model, gs, cache, impl="reference"):
        self.model, self.gs, self._cache, self.impl = model, gs, cache, impl

    def apply(self, cache, iv):
        if self.impl == "optimised":
            from src.baselines.linprop_v2 import linprop_v2
            from src.incremental.index import get_index
            return linprop_v2(cache, iv, "frozen", get_index(self._cache))
        return _linprop(cache, iv, degree_mode="frozen")
