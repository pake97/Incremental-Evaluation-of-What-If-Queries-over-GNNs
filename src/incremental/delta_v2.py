"""Delta maintenance, v2: the same low-rank propagation as delta.py (Alg. 1),
with per-query cost ~ support rows + the edges leaving them.

Data-access changes only:
  * out-edges of the support rows come from the session index (CSR for a
    forward relation, CSC for its reverse) — no |E_r| gather per relation;
  * the RGCN mean denominator is the in-degree read off the index's indptr
    (identical integer counts to cache.deg) — no lazily built _deg / _ei_all;
  * destinations are compacted once per (layer, node type) with a bool mask
    over N_t + nonzero, replacing two torch.unique sorts;
  * every relation's contribution is accumulated in v1's order (edge term,
    then self term, relation by relation), so the sums are v1's;
  * every cached layer, the last included, is a RowOverlay; the returned final
    layer is densified transiently (not retained by the cache).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.incremental.delta import _is_uniform
from src.incremental.index import ragged
from src.incremental.overlay import OverlayCache, RowOverlay, dense, gather
from src.models.model import _key


@torch.no_grad()
def delta_forward_v2(model, cache, gs, iv, index):
    enc = model.encoder
    model.eval()
    assert enc.model_type == "rgcn", "delta path is RGCN-only"
    L, hidden = enc.num_layers, enc.hidden
    N = {nt: cache.h[0][nt].size(0) for nt in enc.node_types}
    stats = {"method": "delta", "impl": "v2", "nodes_recomputed": 0,
             "edges_touched": 0, "edges_scanned": 0, "sign_flips": 0,
             "dirty_frontier": []}
    raw = lambda layer: getattr(layer, "raw", layer)

    def out_edges(et, r):
        """(dst ids, owner index into r, in-degree source) of edges leaving r."""
        s, rel, d = et
        if rel.startswith("rev_"):
            ix = index.rels[(d, rel[4:], s)]
            pos, own = ragged(ix.csc_ptr, r)
            return ix.csc_src[pos], own, ix.csr_ptr      # in-deg in reverse = out-deg fwd
        ix = index.rels[et]
        pos, own = ragged(ix.csr_ptr, r)
        return ix.csr_dst[pos], own, ix.csc_ptr

    # layer 0: Δ through the input projection (bias cancels)
    rows, dh = {}, {}
    for nt, delta in iv.feat_deltas.items():
        r = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if r.numel():
            rows[nt] = r
            dh[nt] = F.linear(delta[r], enc.proj[nt].weight)
    h0 = dict(raw(cache.h[0]))
    for nt, r in rows.items():
        h0[nt] = RowOverlay(h0[nt], r, gather(h0[nt], r) + dh[nt])
    new_h, new_z = [h0], []

    for li in range(L):
        layer = enc.layers[li]
        last = li == L - 1
        # pass 1: out-edges of the support, per relation; union of targets per type
        edges = {}
        hit = {}
        for et in enc.all_types:
            s, _, d = et
            if s in rows:
                e_dst, own, dptr = out_edges(et, rows[s])
                stats["edges_scanned"] += e_dst.numel()
                if e_dst.numel():
                    edges[et] = (e_dst, own, dptr)
                    hit.setdefault(d, torch.zeros(N[d], dtype=torch.bool))[e_dst] = True
            if d in rows:
                hit.setdefault(d, torch.zeros(N[d], dtype=torch.bool))[rows[d]] = True
        u = {nt: m.nonzero().squeeze(1) for nt, m in hit.items()}
        lut = {}
        for nt, un in u.items():
            lut[nt] = torch.empty(N[nt], dtype=torch.long)
            lut[nt][un] = torch.arange(un.numel())

        # pass 2: accumulate Δz on the union rows, in v1's (relation) order
        dz = {nt: torch.zeros(un.numel(), hidden) for nt, un in u.items()}
        for et in enc.all_types:
            s, _, d = et
            cell = layer[_key(et)]
            if et in edges:
                e_dst, own, dptr = edges[et]
                inv = lut[d][e_dst]
                deg = (dptr[u[d] + 1] - dptr[u[d]]).float().clamp(min=1e-8)
                if _is_uniform(dh[s]):            # rank-one hop: identical messages
                    cnt = torch.zeros(u[d].numel()).index_add_(
                        0, inv, torch.ones(inv.numel()))
                    msg0 = F.linear(dh[s][:1], cell.lin_msg.weight)[0]
                    dz[d] += (cnt / deg).unsqueeze(-1) * msg0
                else:
                    # W·Δh once per support row, then gathered per edge
                    # (row-wise linear commutes with the gather; v1 applied
                    # W per edge)
                    msg = F.linear(dh[s], cell.lin_msg.weight)[own]
                    num = torch.zeros(u[d].numel(), hidden).index_add_(0, inv, msg)
                    dz[d] += num / deg.unsqueeze(-1)
                stats["edges_touched"] += int(e_dst.numel())
            if d in rows:
                dz[d].index_add_(0, lut[d][rows[d]], F.linear(dh[d], cell.lin_self.weight))

        h_layer = dict(raw(cache.h[li + 1]))
        z_layer = dict(raw(cache.z[li]))
        rows_next, dh_next = {}, {}
        for nt, un in u.items():
            hb, zb = h_layer[nt], z_layer[nt]
            if last:                                   # z == h at the last layer
                h_layer[nt] = z_layer[nt] = RowOverlay(hb, un, gather(hb, un) + dz[nt])
                corr = dz[nt]
            else:
                z_old = gather(zb, un)
                h_old = gather(hb, un)
                corr = F.relu(z_old + dz[nt]) - h_old
                stats["sign_flips"] += int(((z_old > 0) != (z_old + dz[nt] > 0)).sum())
                h_layer[nt] = RowOverlay(hb, un, h_old + corr)
                z_layer[nt] = RowOverlay(zb, un, z_old + dz[nt])
            rows_next[nt], dh_next[nt] = un, corr
            stats["nodes_recomputed"] += int(un.numel())
        stats["dirty_frontier"].append({nt: int(r.numel()) for nt, r in rows_next.items()})
        new_h.append(h_layer)
        new_z.append(z_layer)
        rows, dh = rows_next, dh_next

    return ({nt: dense(e) for nt, e in new_h[-1].items()},
            OverlayCache(new_h, new_z, cache, gs, iv, index), stats)
