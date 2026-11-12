"""Delta maintenance: low-rank propagation for feature interventions on RGCN.

A feature intervention is a rank-one update of one node type's feature matrix
(one column shifted over a cohort). With a fixed, normalized aggregator and
unchanged topology every operation between activations is LINEAR, so instead
of recomputing rows we propagate the *change*:

  Δh0 = Δx · W_proj^T                          (rank-one, cohort rows only)
  Δz_{l+1} = Σ_r  scatter(W_msg Δh_l over edges FROM dirty sources) / deg_r
             + W_self Δh_l                     (biases cancel)
  Δh_{l+1} = ReLU(z_{l+1} + Δz_{l+1}) − ReLU(z_{l+1})   (cached pre-activations
             make the ReLU correction exact; it is the identity wherever the
             activation sign did not flip, sparse corrections where it did)

Implementation notes (both exactness-preserving):
  * All buffers are sized to the SUPPORT (touched rows), never to N — the
    only O(N) objects are int64 lookup tables and the copy-on-write clones
    of tensors the result actually changes.
  * Rank-one shortcut: while Δh_l is the same vector on every support row
    (true at hop 0 for a uniform cohort shift), each edge carries the
    identical message, so aggregation needs per-dst edge COUNTS instead of
    per-edge matmuls: Δagg = (count/deg) ⊗ (W_msg δ). The sums are the same;
    only the association order differs (covered by ATOL/RTOL).

The saving over closure recompute: only edges LEAVING the dirty set are
touched, never the full in-neighborhood of a dirty node — a node with 10k
in-edges and one dirty in-neighbor costs one edge, not 10k.

Requires: model_type == 'rgcn' (data-independent aggregator; HGAT's attention
renormalizes over all in-edges, so it goes through closure), intervention is
feature-only, and the cache's topology equals the query graph's topology.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.incremental.cache import ForwardCache
from src.models.model import _key


def _is_uniform(vals: torch.Tensor) -> bool:
    """All support rows carry the same delta vector (rank-one, e.g. a uniform
    cohort shift). Cheap O(rows x hidden) check."""
    return vals.size(0) > 1 and bool((vals == vals[0]).all())


@torch.no_grad()
def delta_forward(model, cache: ForwardCache, feat_deltas: Dict[str, torch.Tensor]):
    """Exact h after a feature-only intervention on an RGCN, from `cache`.
    feat_deltas[nt] is the full [N_nt, d_nt] additive delta (mostly zero rows).
    Returns (h_final, new_cache, stats)."""
    enc = model.encoder
    model.eval()
    assert enc.model_type == "rgcn", "delta path is RGCN-only"
    L = enc.num_layers
    N = {nt: cache.h[0][nt].size(0) for nt in enc.node_types}
    hidden = enc.hidden
    stats = {"method": "delta", "nodes_recomputed": 0, "edges_touched": 0,
             "edges_scanned": 0, "sign_flips": 0, "dirty_frontier": []}

    ei_all = cache.ei_all(enc.fwd_types)

    def make_lookup(nt, r):
        lut = torch.full((N[nt],), -1, dtype=torch.long)
        lut[r] = torch.arange(r.numel())
        return lut

    # layer 0: Δ through the input projection (bias cancels).
    # rows[nt]: support indices; dh[nt]: [len(rows), hidden] compact deltas
    rows: Dict[str, torch.Tensor] = {}
    dh: Dict[str, torch.Tensor] = {}
    for nt, delta in feat_deltas.items():
        r = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if r.numel():
            rows[nt] = r
            dh[nt] = F.linear(delta[r], enc.proj[nt].weight)
    lookup = {nt: make_lookup(nt, r) for nt, r in rows.items()}

    new_h = [_overlay_add(cache.h[0], rows, dh)]
    new_z = []

    for li in range(L):
        layer = enc.layers[li]
        # per node type: list of (global_rows, values) contributions
        acc: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        for et in enc.all_types:
            s, _, d = et
            cell = layer[_key(et)]
            ei = ei_all[et]
            if s in rows and ei.numel():
                li_src = lookup[s][ei[0]]
                stats["edges_scanned"] += ei.size(1)
                m = li_src >= 0
                if bool(m.any()):
                    e_dst = ei[1][m]
                    e_src = li_src[m]
                    u_dst, inv = torch.unique(e_dst, return_inverse=True)
                    deg = cache.deg(et, N[d])[u_dst]
                    if _is_uniform(dh[s]):
                        # rank-one hop: identical message on every edge
                        cnt = torch.zeros(u_dst.numel()).index_add_(
                            0, inv, torch.ones(inv.numel()))
                        msg0 = F.linear(dh[s][:1], cell.lin_msg.weight)[0]
                        vals = (cnt / deg).unsqueeze(-1) * msg0
                    else:
                        msg = F.linear(dh[s][e_src], cell.lin_msg.weight)
                        num = torch.zeros(u_dst.numel(), hidden).index_add_(0, inv, msg)
                        vals = num / deg.unsqueeze(-1)
                    acc.setdefault(d, []).append((u_dst, vals))
                    stats["edges_touched"] += int(m.sum())
            if d in rows:
                acc.setdefault(d, []).append(
                    (rows[d], F.linear(dh[d], cell.lin_self.weight)))

        rows_next, dh_next, dz_compact = {}, {}, {}
        for nt, parts in acc.items():
            cat_rows = torch.cat([p[0] for p in parts])
            cat_vals = torch.cat([p[1] for p in parts])
            u, inv = torch.unique(cat_rows, return_inverse=True)
            dz = torch.zeros(u.numel(), hidden).index_add_(0, inv, cat_vals)
            if li < L - 1:
                # exact ReLU correction against cached pre-activations.
                z_old = cache.z[li][nt][u]
                corr = F.relu(z_old + dz) - cache.h[li + 1][nt][u]
                # a sign flip = an activation whose ReLU on/off state changed;
                # these are the only entries where the low-rank propagation
                # needs a nonlinear correction. Counting them measures how
                # sparse that correction is.
                stats["sign_flips"] += int(((z_old > 0) != (z_old + dz > 0)).sum())
            else:
                corr = dz
            rows_next[nt] = u
            dh_next[nt] = corr
            dz_compact[nt] = dz
            stats["nodes_recomputed"] += int(u.numel())

        stats["dirty_frontier"].append({nt: int(r.numel())
                                        for nt, r in rows_next.items()})
        new_z.append(_overlay_add(cache.z[li], rows_next, dz_compact))
        new_h.append(_overlay_add(cache.h[li + 1], rows_next, dh_next))
        rows, dh = rows_next, dh_next
        lookup = {nt: make_lookup(nt, r) for nt, r in rows.items()}

    x_new = dict(cache.x_dict)
    for nt, delta in feat_deltas.items():
        x_new[nt] = cache.x_dict[nt] + delta
    new_cache = ForwardCache(new_h, new_z, x_new, dict(cache.edge_index_dict))
    new_cache._deg = cache._deg  # topology unchanged — share the degree cache
    new_cache._ei_all = cache._ei_all  # and the reverse-index memo
    return new_h[-1], new_cache, stats


def _overlay_add(base: Dict[str, torch.Tensor], rows: Dict[str, torch.Tensor],
                 add: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """base with add summed onto the support rows, copy-on-write per node type."""
    out = dict(base)
    for nt, r in rows.items():
        if r.numel():
            t = base[nt].clone()
            t[r] += add[nt]
            out[nt] = t
    return out
