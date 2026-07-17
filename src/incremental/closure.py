"""Closure maintenance: exact incremental forward by dirty-row recomputation.

The general engine — works for both architectures (RGCN, HGAT) and both
intervention types (feature and topological). Per layer it recomputes ONLY the
rows whose inputs could have changed:

  dirty_{l+1} = struct_dirty ∪ dirty_l ∪ { v : some in-neighbor of v (in the
                edited graph, forward or reverse relation) is in dirty_l }

seeded with the feature cohort at layer 0. `struct_dirty` (endpoints of edited
edges) is re-added at every layer because edited topology changes a node's
aggregation at every layer. Recomputed rows use the model's own cell modules on
the restricted edge set — the same math as the full forward, so the result is
exact up to float summation order (covered by ATOL/RTOL).

Dirty sets are conservative supersets: recomputing a clean row reproduces its
value, so over-marking can never break exactness — only under-marking could,
and the expansion above never under-marks.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.incremental.cache import ForwardCache
from src.models.model import _key

EdgeType = Tuple[str, str, str]


@torch.no_grad()
def closure_forward(model, cache: ForwardCache, view,
                    feat_dirty: Dict[str, torch.Tensor],
                    struct_dirty: Dict[str, torch.Tensor]):
    """Exact h on `view`'s graph, starting from `cache` (computed on a graph
    that differs from `view` only by the edits summarized in feat_dirty /
    struct_dirty). Returns (h_final, new_cache, stats)."""
    enc = model.encoder
    model.eval()
    L = enc.num_layers
    N = {nt: cache.h[0][nt].size(0) for nt in enc.node_types}
    stats = {"method": "closure", "nodes_recomputed": 0, "edges_touched": 0,
             "sign_flips": 0, "dirty_frontier": []}

    # edited adjacency, forward + derived reverse
    ei_all = {}
    for (s, r, d) in enc.fwd_types:
        ei = view.edge_index_dict[(s, r, d)]
        ei_all[(s, r, d)] = ei
        ei_all[(d, "rev_" + r, s)] = ei.flip(0)

    def mask_of(rows_by_nt):
        m = {nt: torch.zeros(N[nt], dtype=torch.bool) for nt in enc.node_types}
        for nt, rows in rows_by_nt.items():
            if rows.numel():
                m[nt][rows] = True
        return m

    dirty = mask_of(feat_dirty)
    sdirty = mask_of(struct_dirty)

    # layer 0: re-project feature-dirty rows
    h_prev = dict(cache.h[0])
    for nt, rows in feat_dirty.items():
        if rows.numel():
            t = cache.h[0][nt].clone()
            t[rows] = enc.proj[nt](view.x_dict[nt][rows])
            h_prev[nt] = t
    hs = [h_prev]
    zs = []

    for li in range(L):
        # expand the dirty frontier
        nxt = {nt: dirty[nt] | sdirty[nt] for nt in enc.node_types}
        for et in enc.all_types:
            s, _, d = et
            ei = ei_all[et]
            if ei.numel() and bool(dirty[s].any()):
                m = dirty[s][ei[0]]
                if bool(m.any()):
                    nxt[d][ei[1][m]] = True

        layer = enc.layers[li]
        h_out = dict(cache.h[li + 1])
        z_out = dict(cache.z[li])
        for nt in enc.node_types:
            if not bool(nxt[nt].any()):
                continue
            rows = nxt[nt].nonzero(as_tuple=False).squeeze(1)
            lookup = torch.full((N[nt],), -1, dtype=torch.long)
            lookup[rows] = torch.arange(rows.numel())
            h_dst = h_prev[nt][rows]

            acc = []
            for et in enc.all_types:
                s, _, d = et
                if d != nt:
                    continue
                ei = ei_all[et]
                if ei.numel():
                    m = nxt[nt][ei[1]]
                    e_sel = ei[:, m]
                else:
                    e_sel = ei
                if e_sel.numel():
                    # gather only the needed src rows (cell applies lin to all
                    # of h_src, so pass the unique-src slice)
                    u, inv = torch.unique(e_sel[0], return_inverse=True)
                    h_src = h_prev[s][u]
                    ei_local = torch.stack([inv, lookup[e_sel[1]]])
                    ew = torch.ones(ei_local.size(1))
                    stats["edges_touched"] += int(ei_local.size(1))
                else:
                    h_src = h_prev[s][:0]
                    ei_local = e_sel.new_zeros((2, 0))
                    ew = torch.ones(0)
                acc.append(layer[_key(et)](h_src, h_dst, ei_local, ew, rows.numel()))

            z_rows = sum(acc) if acc else h_dst
            h_rows = F.relu(z_rows) if li < L - 1 else z_rows
            h_out[nt] = cache.h[li + 1][nt].clone()
            h_out[nt][rows] = h_rows
            z_out[nt] = cache.z[li][nt].clone()
            z_out[nt][rows] = z_rows
            stats["nodes_recomputed"] += int(rows.numel())
            # true induced sign-flips: recomputed rows whose ReLU on/off state
            # changed vs the base cache (rows that did not actually change
            # contribute z_new == z_old and no flip), comparable to the delta
            # path's count.
            if li < L - 1:
                z_old = cache.z[li][nt][rows]
                stats["sign_flips"] += int(((z_old > 0) != (z_rows > 0)).sum())

        stats["dirty_frontier"].append({nt: int(nxt[nt].sum()) for nt in enc.node_types
                                        if bool(nxt[nt].any())})
        hs.append(h_out)
        zs.append(z_out)
        h_prev = h_out
        dirty = nxt

    new_cache = ForwardCache(hs, zs, dict(view.x_dict), dict(view.edge_index_dict))
    return hs[-1], new_cache, stats
