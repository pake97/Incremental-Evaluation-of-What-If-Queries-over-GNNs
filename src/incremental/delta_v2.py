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

Edge edits (v2 only, method="delta" on a topological Δ; the mean aggregator is
sum / in-degree, so it stays linear given the new degree):
  * kept base edges from changed sources carry W·Δh (deleted columns skipped);
    added edges carry W·h' (the source's new state), deleted edges -W·h;
  * a destination whose in-degree changes gets (num + Δnum)/deg' - num/deg,
    with num rebuilt from its base in-edges (the cache keeps h and z, not the
    per-relation sums); every other destination gets Δnum/deg.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.incremental.delta import _is_uniform
from src.incremental.index import in_sorted, ragged
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

    def fwd_of(et):
        s, rel, d = et
        return ((d, rel[4:], s), True) if rel.startswith("rev_") else (et, False)

    def out_edges(et, r):
        """(dst ids, owner index into r, in-degree source) of kept edges leaving r."""
        f, rev = fwd_of(et)
        ix = index.rels[f]
        if rev:
            pos, own = ragged(ix.csc_ptr, r)
            dst, eid, dptr = ix.csc_src[pos], ix.csc_eid[pos], ix.csr_ptr
        else:
            pos, own = ragged(ix.csr_ptr, r)
            dst, eid, dptr = ix.csr_dst[pos], ix.csr_eid[pos], ix.csc_ptr
        if f in dels:                                   # skip deleted base columns
            k = ~in_sorted(eid, dels[f])
            dst, own = dst[k], own[k]
        return dst, own, dptr

    def in_edges(et, r):
        """(src ids, owner index into r) of the BASE in-edges of rows r."""
        f, rev = fwd_of(et)
        ix = index.rels[f]
        if rev:
            pos, own = ragged(ix.csr_ptr, r)
            return ix.csr_dst[pos], own
        pos, own = ragged(ix.csc_ptr, r)
        return ix.csc_src[pos], own

    # edge edits per relation (both directions): added / deleted (src, dst) and Δdeg
    dels = {et: torch.sort(c).values for et, c in iv.del_cols.items()
            if c is not None and c.numel()}
    edits = {}
    for et in enc.all_types:
        f, rev = fwd_of(et)
        a = iv.add_edges.get(f)
        a = a if a is not None and a.numel() else None
        dl = gs.edge_index[f][:, dels[f]] if f in dels else None
        if a is None and dl is None:
            continue
        if rev:
            a = a.flip(0) if a is not None else None
            dl = dl.flip(0) if dl is not None else None
        ddeg = torch.zeros(N[et[2]], dtype=torch.long)
        if a is not None:
            ddeg.index_add_(0, a[1], torch.ones(a.size(1), dtype=torch.long))
        if dl is not None:
            ddeg.index_add_(0, dl[1], -torch.ones(dl.size(1), dtype=torch.long))
        edits[et] = (a, dl, ddeg)

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
            if et in edits:
                m = hit.setdefault(d, torch.zeros(N[d], dtype=torch.bool))
                for e in edits[et][:2]:
                    if e is not None:
                        m[e[1]] = True
        u = {nt: m.nonzero().squeeze(1) for nt, m in hit.items()}
        lut = {}
        for nt, un in u.items():
            lut[nt] = torch.empty(N[nt], dtype=torch.long)
            lut[nt][un] = torch.arange(un.numel())

        # pass 2: accumulate Δz on the union rows, in v1's (relation) order
        dz = {nt: torch.zeros(un.numel(), hidden) for nt, un in u.items()}
        h_cur, h_base = new_h[li], raw(cache.h[li])
        for et in enc.all_types:
            s, _, d = et
            cell = layer[_key(et)]
            if et in edits:
                # Δnum on u[d]: kept edges (W·Δh), added (W·h'), deleted (-W·h)
                add, dl, ddeg = edits[et]
                W = cell.lin_msg.weight
                dnum = torch.zeros(u[d].numel(), hidden)
                n_e = 0
                if et in edges:
                    e_dst, own, dptr = edges[et]
                    dnum.index_add_(0, lut[d][e_dst], F.linear(dh[s], W)[own])
                    n_e += e_dst.numel()
                for e, sign, src_h in ((add, 1.0, h_cur), (dl, -1.0, h_base)):
                    if e is not None:
                        dnum.index_add_(0, lut[d][e[1]], sign * F.linear(gather(src_h[s], e[0]), W))
                        n_e += e.size(1)
                f, rev = fwd_of(et)
                dptr = index.rels[f].csr_ptr if rev else index.rels[f].csc_ptr
                deg = (dptr[u[d] + 1] - dptr[u[d]]).float()
                dd = ddeg[u[d]]
                chg = (dd != 0).nonzero().squeeze(1)
                contrib = dnum / deg.clamp(min=1e-8).unsqueeze(-1)
                if chg.numel():
                    # rows whose in-degree changes: rebuild the base sum, renormalise
                    src, own = in_edges(et, u[d][chg])
                    num = torch.zeros(chg.numel(), hidden).index_add_(
                        0, own, F.linear(gather(h_base[s], src), W))
                    n_e += src.numel()
                    d0, d1 = deg[chg].clamp(min=1e-8), deg[chg] + dd[chg]
                    # a row left with no in-edges aggregates to exactly 0 (from
                    # scratch: empty sum); num + Δnum is only a float residual there
                    new = torch.where((d1 > 0).unsqueeze(-1),
                                      (num + dnum[chg]) / d1.clamp(min=1e-8).unsqueeze(-1),
                                      torch.zeros(()))
                    contrib[chg] = new - num / d0.unsqueeze(-1)
                dz[d] += contrib
                stats["edges_touched"] += int(n_e)
            elif et in edges:
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
