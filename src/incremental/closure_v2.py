"""Closure maintenance, v2: the same algorithm as closure.py (L-hop dirty-row
recompute, Alg. 2), with per-query cost ~ affected rows + their incident edges.

What changes is only the data access:
  * topology = base session index (CSR/CSC, src/incremental/index.py) ⊕ the
    intervention's own edit list (deleted base columns, added edges) — no
    gs.apply, no re-flip of every relation, no |E|-length masks;
  * frontier expansion slices the out-edges of dirty rows; aggregation slices
    the in-edges of recomputed rows (CSC order = base column order within a
    row, added edges after — the order from-scratch sums in);
  * source rows are compacted with a bool mask + nonzero over N_s (no sort);
    destinations need no compaction (CSC slices come grouped by row);
  * every cached layer, the last included, is a RowOverlay; the returned final
    layer is densified transiently (not retained by the cache).
Dirty sets, recomputed rows, edges aggregated and the cell math are those of
v1, so stats (nodes_recomputed, edges_touched, sign_flips) match v1.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.incremental.index import in_sorted, ragged
from src.incremental.overlay import OverlayCache, RowOverlay, dense, gather
from src.models.model import _key


@torch.no_grad()
def closure_forward_v2(model, cache, gs, iv, index,
                       feat_dirty: Dict[str, torch.Tensor],
                       struct_dirty: Dict[str, torch.Tensor], attn=None, dense_frac=None):
    """Exact h on gs ⊕ iv (iv base-relative), resumed from `cache` (whose
    state differs from gs ⊕ iv only on feat_dirty / struct_dirty rows).

    attn (hGAT, base cache only; see attn_state): attention Δ. A recomputed row
    whose own h is unchanged gets z_old + Σ_r Δagg_r from the cached softmax
    sums; only rows whose own h changed are aggregated from their in-edges.

    dense_frac: a node type whose recomputed rows reach this fraction of its rows
    is recomputed in full, from-scratch style (whole relation edge lists, no row
    slicing or source compaction); the extra rows are clean, so they come out
    unchanged. None = never."""
    enc = model.encoder
    model.eval()
    L = enc.num_layers
    N = {nt: cache.h[0][nt].size(0) for nt in enc.node_types}
    stats = {"method": "closure", "impl": "v2", "nodes_recomputed": 0,
             "edges_touched": 0, "edges_scanned": 0, "sign_flips": 0,
             "dirty_frontier": []}

    adds = {et: a for et, a in iv.add_edges.items() if a is not None and a.numel()}
    dels = {et: torch.sort(c).values for et, c in iv.del_cols.items()
            if c is not None and c.numel()}
    stats["edges_scanned"] += sum(c.numel() for c in dels.values())   # the sort

    def mask_of(rows_by_nt):
        m = {nt: torch.zeros(N[nt], dtype=torch.bool) for nt in enc.node_types}
        for nt, rows in rows_by_nt.items():
            if rows.numel():
                m[nt][rows] = True
        return m

    dirty, sdirty = mask_of(feat_dirty), mask_of(struct_dirty)

    def in_ptr(et):
        s_, r_, d_ = et
        if r_.startswith("rev_"):
            return index.rels[(d_, r_[4:], s_)].csr_ptr
        return index.rels[et].csc_ptr

    def oriented(et):
        """(fwd relation, added, deleted) edges of et as (src, dst) in et's direction."""
        s_, r_, d_ = et
        rev = r_.startswith("rev_")
        fwd = (d_, r_[4:], s_) if rev else et
        a = adds.get(fwd)
        dl = gs.edge_index[fwd][:, dels[fwd]] if fwd in dels else None
        if rev:
            a = a.flip(0) if a is not None else None
            dl = dl.flip(0) if dl is not None else None
        return fwd, rev, a, dl

    full_ei = {}

    def full_edges(et):
        """et's edges on gs ⊕ iv as (src, dst), in the order gs.apply + build_cache use."""
        if et not in full_ei:
            fwd, rev, a, _ = oriented(et)
            ei = gs.edge_index[fwd]
            if fwd in dels:
                keep = torch.ones(ei.size(1), dtype=torch.bool)
                keep[dels[fwd]] = False
                ei = ei[:, keep]
            if rev:
                ei = ei.flip(0)
            full_ei[et] = torch.cat([ei, a], 1) if a is not None else ei
        return full_ei[et]

    def _attn_part(part, nt, li, layer):
        """z on rows whose own h is unchanged: z_old + Σ_r (N'/D' - N/D), with N, D
        the base softmax numerator / denominator (shift c) and N', D' after the
        terms of changed / added / deleted in-edges are swapped."""
        P = part.numel()
        lut = torch.full((N[nt],), -1, dtype=torch.long)
        lut[part] = torch.arange(P)
        z = gather(hraw(cache.z[li], nt), part).clone()
        h_v = gather(h_prev[nt], part)                  # == base h on these rows
        for et in enc.all_types:
            s_ = et[0]
            if et[2] != nt:
                continue
            cell = layer[_key(et)]
            c, Nm, D = attn[li][et]
            fwd, rev, a, dl = oriented(et)
            ad = (cell.lin(h_v) * cell.att_dst).sum(-1)     # per part row
            groups = []                                      # (src, dst, h layer, sign)
            dN = torch.zeros(P, Nm.size(-1))
            dD = torch.zeros(P)
            hit = torch.zeros(P, dtype=torch.bool)
            dr = dirty[s_].nonzero().squeeze(1)
            if dr.numel():
                ix = index.rels[fwd]
                optr, iptr = (ix.csc_ptr, ix.csr_ptr) if rev else (ix.csr_ptr, ix.csc_ptr)
                if int((optr[dr + 1] - optr[dr]).sum()) <= int((iptr[part + 1] - iptr[part]).sum()):
                    # out-edges of the changed sources, kept if they land on a Δ row
                    pos, own = ragged(optr, dr)
                    dst, eid = (ix.csc_src[pos], ix.csc_eid[pos]) if rev \
                        else (ix.csr_dst[pos], ix.csr_eid[pos])
                    k = lut[dst] >= 0
                else:
                    # in-edges of the Δ rows, kept if they come from a changed source
                    pos, pown = ragged(iptr, part)
                    src, eid = (ix.csr_dst[pos], ix.csr_eid[pos]) if rev \
                        else (ix.csc_src[pos], ix.csc_eid[pos])
                    lut_dr = torch.full((N[s_],), -1, dtype=torch.long)
                    lut_dr[dr] = torch.arange(dr.numel())
                    own, dst = lut_dr[src], part[pown]
                    k = own >= 0
                if fwd in dels:
                    k &= ~in_sorted(eid, dels[fwd])
                own, dst = own[k], dst[k]
                stats["edges_scanned"] += pos.numel()
                if dst.numel():
                    # kept edges from changed sources: W·h and logit once per
                    # source row (old and new), one combined term per edge
                    loc = lut[dst]
                    w_ = {}
                    for H in (cache.h[li], h_prev):
                        hs = cell.lin(gather(hraw(H, s_), dr))
                        e = F.leaky_relu((hs * cell.att_src).sum(-1)[own] + ad[loc], 0.2)
                        w_[id(H)] = (hs, torch.exp(e - c[dst]))
                    (ho, wo), (hn, wn) = w_[id(cache.h[li])], w_[id(h_prev)]
                    dN.index_add_(0, loc, hn[own] * wn.unsqueeze(-1) - ho[own] * wo.unsqueeze(-1))
                    dD.index_add_(0, loc, wn - wo)
                    hit[loc] = True
                    stats["edges_touched"] += int(dst.numel())
            if a is not None:
                m = lut[a[1]] >= 0
                groups.append((a[0][m], a[1][m], h_prev, 1.0))
            if dl is not None:
                m = lut[dl[1]] >= 0
                groups.append((dl[0][m], dl[1][m], cache.h[li], -1.0))
            ddeg = torch.zeros(P, dtype=torch.long)
            for src, dst, H, sign in groups:              # added / deleted edges
                if not src.numel():
                    continue
                loc = lut[dst]
                hs = cell.lin(gather(hraw(H, s_), src))
                e = F.leaky_relu((hs * cell.att_src).sum(-1) + ad[loc], 0.2)
                w = torch.exp(e - c[dst])
                dN.index_add_(0, loc, sign * hs * w.unsqueeze(-1))
                dD.index_add_(0, loc, sign * w)
                hit[loc] = True
                stats["edges_touched"] += int(src.numel())
            if a is not None:
                ddeg.index_add_(0, lut[a[1]][lut[a[1]] >= 0],
                                torch.ones(int((lut[a[1]] >= 0).sum()), dtype=torch.long))
            if dl is not None:
                ddeg.index_add_(0, lut[dl[1]][lut[dl[1]] >= 0],
                                -torch.ones(int((lut[dl[1]] >= 0).sum()), dtype=torch.long))
            t = hit.nonzero().squeeze(1)
            if not t.numel():
                continue
            pr = part[t]
            n0, d0 = Nm[pr], D[pr]
            n1, d1 = n0 + dN[t], d0 + dD[t]
            ptr = in_ptr(et)
            deg1 = (ptr[pr + 1] - ptr[pr]) + ddeg[t]
            zero = torch.zeros(())
            old = torch.where((d0 > 0).unsqueeze(-1), n0 / d0.clamp(min=1e-30).unsqueeze(-1), zero)
            # a row left with no in-edges aggregates to exactly 0 (empty sum)
            new = torch.where((deg1 > 0).unsqueeze(-1), n1 / d1.clamp(min=1e-30).unsqueeze(-1), zero)
            z[t] += new - old
        return z
    hraw = lambda layer, nt: getattr(layer, "raw", layer)[nt]   # overlay-aware read

    # layer 0: re-project feature-dirty rows (x on gs ⊕ iv, rows only)
    h_prev = dict(getattr(cache.h[0], "raw", cache.h[0]))
    for nt, rows in feat_dirty.items():
        if rows.numel():
            x = gs.x[nt][rows] + iv.feat_deltas[nt][rows]
            h_prev[nt] = RowOverlay(hraw(cache.h[0], nt), rows, enc.proj[nt](x))
    hs, zs = [h_prev], []

    for li in range(L):
        drows = {nt: dirty[nt].nonzero().squeeze(1) for nt in enc.node_types}
        nxt = {nt: dirty[nt] | sdirty[nt] for nt in enc.node_types}
        # expand: dst of out-edges of dirty rows, forward and reverse direction
        for (s, r, d), ix in index.rels.items():
            a = adds.get((s, r, d))
            dense_s = dense_frac is not None and drows[s].numel() >= dense_frac * N[s]
            dense_d = dense_frac is not None and drows[d].numel() >= dense_frac * N[d]
            if dense_s or dense_d:                    # most rows dirty: one pass over E
                ei = gs.edge_index[(s, r, d)]
                if a is not None:
                    ei = torch.cat([ei, a], 1)
                if dense_s:
                    nxt[d][ei[1][dirty[s][ei[0]]]] = True
                if dense_d:
                    nxt[s][ei[0][dirty[d][ei[1]]]] = True
                stats["edges_scanned"] += ei.size(1)
            if drows[s].numel() and not dense_s:
                pos, _ = ragged(ix.csr_ptr, drows[s])
                nxt[d][ix.csr_dst[pos]] = True
                stats["edges_scanned"] += pos.numel()
                if a is not None:
                    nxt[d][a[1][dirty[s][a[0]]]] = True
                    stats["edges_scanned"] += a.size(1)
            if drows[d].numel() and not dense_d:
                pos, _ = ragged(ix.csc_ptr, drows[d])
                nxt[s][ix.csc_src[pos]] = True
                stats["edges_scanned"] += pos.numel()
                if a is not None:
                    nxt[s][a[0][dirty[d][a[1]]]] = True
                    stats["edges_scanned"] += a.size(1)
            # deleted edges join two struct-dirty rows (dirty at every layer),
            # so expanding through them marks nothing new — same set as v1

        layer = enc.layers[li]
        h_out = dict(getattr(cache.h[li + 1], "raw", cache.h[li + 1]))
        z_out = dict(getattr(cache.z[li], "raw", cache.z[li]))
        last = li == L - 1
        dprev = {}

        def dense_prev(nt):
            if nt not in dprev:
                dprev[nt] = dense(h_prev[nt])
            return dprev[nt]

        for nt in enc.node_types:
            rows_all = nxt[nt].nonzero().squeeze(1)
            if not rows_all.numel():
                continue
            if dense_frac is not None and attn is None and rows_all.numel() >= dense_frac * N[nt]:
                rows = rows_all = torch.arange(N[nt])
                h_dst = dense_prev(nt)
                acc = []
                for et in [et for et in enc.all_types if et[2] == nt]:
                    ei = full_edges(et)
                    stats["edges_touched"] += int(ei.size(1))
                    acc.append(layer[_key(et)](dense_prev(et[0]), h_dst, ei,
                                               torch.ones(ei.size(1)), N[nt]))
                z_rows = sum(acc) if acc else h_dst
                stats["nodes_recomputed"] += N[nt]
                if last:                              # every row new: plain tensors
                    h_out[nt] = z_out[nt] = z_rows
                else:
                    z_old = dense(hraw(cache.z[li], nt))
                    stats["sign_flips"] += int(((z_old > 0) != (z_rows > 0)).sum())
                    h_out[nt], z_out[nt] = F.relu(z_rows), z_rows
                continue
            full = dirty[nt][rows_all] if attn is not None else None
            rows = rows_all[full] if attn is not None else rows_all   # aggregated rows
            amask = torch.zeros(N[nt], dtype=torch.bool)
            amask[rows] = True
            h_dst = gather(h_prev[nt], rows)
            acc = []
            for et in [et for et in enc.all_types if et[2] == nt]:
                s, rel, d = et
                if rel.startswith("rev_"):           # in-edges in reverse = out-edges fwd
                    fwd = (d, rel[4:], s)
                    ix = index.rels[fwd]
                    pos, dloc = ragged(ix.csr_ptr, rows)
                    src, eid = ix.csr_dst[pos], ix.csr_eid
                    a_dst, a_src = (adds[fwd][0], adds[fwd][1]) if fwd in adds else (None, None)
                else:
                    fwd = et
                    ix = index.rels[fwd]
                    pos, dloc = ragged(ix.csc_ptr, rows)
                    src, eid = ix.csc_src[pos], ix.csc_eid
                    a_dst, a_src = (adds[fwd][1], adds[fwd][0]) if fwd in adds else (None, None)
                stats["edges_scanned"] += pos.numel()
                if fwd in dels:                       # drop deleted base columns
                    keep = ~in_sorted(eid[pos], dels[fwd])
                    src, dloc = src[keep], dloc[keep]
                    stats["edges_scanned"] += pos.numel() + dels[fwd].numel()
                if a_dst is not None:                 # added edges, after base ones
                    m = amask[a_dst]
                    stats["edges_scanned"] += a_dst.numel()
                    src = torch.cat([src, a_src[m]])
                    dloc = torch.cat([dloc, torch.searchsorted(rows, a_dst[m])])
                if src.numel():
                    seen = torch.zeros(N[s], dtype=torch.bool)
                    seen[src] = True
                    u = seen.nonzero().squeeze(1)
                    lut = torch.empty(N[s], dtype=torch.long)
                    lut[u] = torch.arange(u.numel())
                    h_src = gather(h_prev[s], u)
                    ei_local = torch.stack([lut[src], dloc])
                    stats["edges_touched"] += int(src.numel())
                else:
                    h_src = h_dst[:0]
                    ei_local = torch.zeros((2, 0), dtype=torch.long)
                acc.append(layer[_key(et)](h_src, h_dst, ei_local,
                                           torch.ones(ei_local.size(1)), rows.numel()))

            z_rows = sum(acc) if acc else h_dst
            if attn is not None:
                z_all = torch.empty(rows_all.numel(), z_rows.size(-1))
                z_all[full] = z_rows
                if bool((~full).any()):
                    z_all[~full] = _attn_part(rows_all[~full], nt, li, layer)
                z_rows = z_all
            rows = rows_all
            stats["nodes_recomputed"] += int(rows.numel())
            if last:                                  # z == h at the last layer
                h_out[nt] = z_out[nt] = RowOverlay(hraw(cache.h[li + 1], nt), rows, z_rows)
            else:
                h_rows = F.relu(z_rows)
                z_old = gather(hraw(cache.z[li], nt), rows)
                stats["sign_flips"] += int(((z_old > 0) != (z_rows > 0)).sum())
                h_out[nt] = RowOverlay(hraw(cache.h[li + 1], nt), rows, h_rows)
                z_out[nt] = RowOverlay(hraw(cache.z[li], nt), rows, z_rows)

        stats["dirty_frontier"].append({nt: int(nxt[nt].sum()) for nt in enc.node_types
                                        if bool(nxt[nt].any())})
        hs.append(h_out)
        zs.append(z_out)
        h_prev = h_out
        dirty = nxt

    new_cache = OverlayCache(hs, zs, cache, gs, iv, index)
    return {nt: dense(e) for nt, e in hs[-1].items()}, new_cache, stats


@torch.no_grad()
def attn_state(model, cache):
    """hGAT softmax state of the BASE cache, per layer and relation:
    (c, N, D) with c the per-dst shift (max logit, 0 without in-edges),
    N = Σ exp(e - c)·W h_src and D = Σ exp(e - c). Built once per session and
    memoized on the cache, like the index (Part A/C count it)."""
    if getattr(cache, "_attn", None) is None:
        from torch_geometric.utils import scatter
        enc = model.encoder
        ei_all = cache.ei_all(enc.fwd_types)
        st = []
        for li in range(enc.num_layers):
            layer, h, per = enc.layers[li], cache.h[li], {}
            for et in enc.all_types:
                s, _, d = et
                cell, ei = layer[_key(et)], ei_all[et]
                n_d = h[d].size(0)
                hs = cell.lin(h[s])
                e = F.leaky_relu((hs * cell.att_src).sum(-1)[ei[0]]
                                 + (cell.lin(h[d]) * cell.att_dst).sum(-1)[ei[1]], 0.2)
                c = scatter(e, ei[1], dim=0, dim_size=n_d, reduce="max") if e.numel() \
                    else torch.zeros(n_d)
                w = torch.exp(e - c[ei[1]])
                per[et] = (c, torch.zeros(n_d, hs.size(-1)).index_add_(
                    0, ei[1], hs[ei[0]] * w.unsqueeze(-1)),
                    torch.zeros(n_d).index_add_(0, ei[1], w))
            st.append(per)
        cache._attn = st
    return cache._attn


def attn_state_nbytes(cache) -> int:
    return sum(t.numel() * t.element_size() for per in getattr(cache, "_attn", None) or []
               for trip in per.values() for t in trip)
