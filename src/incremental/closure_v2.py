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
                       struct_dirty: Dict[str, torch.Tensor]):
    """Exact h on gs ⊕ iv (iv base-relative), resumed from `cache` (whose
    state differs from gs ⊕ iv only on feat_dirty / struct_dirty rows)."""
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
            if drows[s].numel():
                pos, _ = ragged(ix.csr_ptr, drows[s])
                nxt[d][ix.csr_dst[pos]] = True
                stats["edges_scanned"] += pos.numel()
                if a is not None:
                    nxt[d][a[1][dirty[s][a[0]]]] = True
                    stats["edges_scanned"] += a.size(1)
            if drows[d].numel():
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
        for nt in enc.node_types:
            rows = nxt[nt].nonzero().squeeze(1)
            if not rows.numel():
                continue
            h_dst = gather(h_prev[nt], rows)
            acc = []
            for et in enc.all_types:
                s, rel, d = et
                if d != nt:
                    continue
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
                    m = nxt[nt][a_dst]
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
