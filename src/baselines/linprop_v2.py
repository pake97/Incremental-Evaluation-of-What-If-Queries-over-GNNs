"""_linprop (Ripple, NeutronRT-RGCN) on the v2 primitives: same arithmetic,
per-query cost ~ changed rows + the edges leaving them.

Data-access changes only (the cell math is methods._linprop's, line for line):
  * kept edges from changed sources = out-edges from the session GraphIndex
    (CSR for a forward relation, CSC of its forward for a reverse), minus the
    deleted columns — no |E_r| copy of the kept edge list, no |E_r| mask;
  * those edges are re-sorted by base column, the order the reference
    aggregates them in, so every per-row sum is the reference's sum;
  * Δh, Δnum and Δz live on the affected rows only (compact buffers + a
    row lookup), not N x hidden per relation and layer;
  * the degree change is built only for edited relations.
Counters are incremented exactly as in the reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.baselines.harness import BaselineCache, Counters
from src.baselines.methods import _z_base
from src.incremental.index import in_sorted, ragged
from src.models.model import _key


def _fwd(et):
    s, r, d = et
    return ((d, r[4:], s), True) if r.startswith("rev_") else (et, False)


@torch.no_grad()
def linprop_v2(cache: BaselineCache, iv, degree_mode: str, index):
    enc = cache.enc
    L, hid = cache.L, enc.hidden
    rgcn = cache.mt == "rgcn"
    N = cache.N

    dels = {et: torch.unique(c) for et, c in iv.del_cols.items()
            if c is not None and c.numel()}
    adds = {et: a for et, a in iv.add_edges.items() if a is not None and a.numel()}

    def edits(et):
        """(added, deleted) edge lists of et in reference order, and Δdeg at dst."""
        f, rev = _fwd(et)
        base = cache.gs.edge_index[f]
        add = adds.get(f, base[:, :0])
        dl = base[:, dels[f]] if f in dels else base[:, :0]
        if rev:
            add, dl = add.flip(0), dl.flip(0)
        ddeg = None
        if degree_mode == "new" and (add.numel() or dl.numel()):
            ddeg = torch.zeros(N[et[2]])
            if add.numel():
                ddeg.index_add_(0, add[1], torch.ones(add.size(1)))
            if dl.numel():
                ddeg.index_add_(0, dl[1], -torch.ones(dl.size(1)))
        return add, dl, ddeg
    E = {et: edits(et) for et in cache.all_types}

    def kept_out(et, rows):
        """Kept base edges leaving `rows` along et, ascending base column."""
        f, rev = _fwd(et)
        ix = index.rels[f]
        if rev:
            pos, own = ragged(ix.csc_ptr, rows)
            dst, eid = ix.csc_src[pos], ix.csc_eid[pos]
        else:
            pos, own = ragged(ix.csr_ptr, rows)
            dst, eid = ix.csr_dst[pos], ix.csr_eid[pos]
        src = rows[own]
        if f in dels:
            k = ~in_sorted(eid, dels[f])
            src, dst, eid = src[k], dst[k], eid[k]
        o = torch.argsort(eid)
        return src[o], dst[o], eid[o]

    # Δh as (rows, values) per type over every recomputed row (zero and non-finite
    # rows kept: the reference's dense Δh carries them into H' and the output);
    # lut maps a row to its slot (-1 = not recomputed). A source counts as
    # changed with the reference's test, |Δh|.sum() > 0 (NaN rows are not).
    def changed(nt):
        rows, vals = dh[nt]
        nz = vals.abs().sum(-1) > 0
        return rows[nz]

    def lut_of(dh):
        out = {}
        for nt, (rows, _) in dh.items():
            lu = torch.full((N[nt],), -1, dtype=torch.long)
            lu[rows] = torch.arange(rows.numel())
            out[nt] = lu
        return out

    def at(dh, lut, nt, idx):
        """Dense Δh rows at idx (zeros where unchanged)."""
        out = torch.zeros(idx.numel(), hid)
        if nt in dh:
            slot = lut[nt][idx]
            m = slot >= 0
            out[m] = dh[nt][1][slot[m]]
        return out

    dh = {}
    for nt, delta in iv.feat_deltas.items():
        rows = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if rows.numel():
            dh[nt] = (rows, F.linear(delta[rows], enc.proj[nt].weight))

    for li in range(L):
        layer = enc.layers[li]
        H = cache.H[li]
        lut = lut_of(dh)
        contrib = []                  # (d, rows, Δz rows) per relation, in order
        for et in cache.all_types:
            s, _, d = et
            cell = layer[_key(et)]
            add, dl, ddeg = E[et]
            parts = []                # (dst, msg) in reference order
            if s in dh:
                src, dst, eid = kept_out(et, changed(s))
                if dst.numel():
                    hsrc = dh[s][1][lut[s][src]]      # W per edge, as the reference
                    if rgcn:
                        msg = cell.lin_msg(hsrc)
                    else:
                        msg = cell.lin(hsrc) * cache.att[li][et][eid].unsqueeze(-1)
                    parts.append((dst, msg))
                    Counters.add_edges(dst.numel())
            if add.numel():
                hs = H[s][add[0]] + at(dh, lut, s, add[0])
                if rgcn:
                    msg = cell.lin_msg(hs)
                else:
                    hd = H[d][add[1]] + at(dh, lut, d, add[1])
                    msg = cell.lin(hs) * _alpha(cache, li, et, add, cell, hs, hd).unsqueeze(-1)
                parts.append((add[1], msg))
                Counters.add_edges(add.size(1))
            if dl.numel():
                hs = H[s][dl[0]]
                if rgcn:
                    msg = cell.lin_msg(hs)
                else:
                    msg = cell.lin(hs) * _alpha(cache, li, et, dl, cell, hs,
                                                H[d][dl[1]]).unsqueeze(-1)
                parts.append((dl[1], -msg))
                Counters.add_edges(dl.size(1))

            aff = torch.zeros(N[d], dtype=torch.bool)
            for dst, _ in parts:
                aff[dst] = True
            if d in dh:
                aff[changed(d)] = True
            if not bool(aff.any()):
                continue
            rows = aff.nonzero(as_tuple=False).squeeze(1)
            pos = torch.empty(N[d], dtype=torch.long)
            pos[rows] = torch.arange(rows.numel())
            dnum = torch.zeros(rows.numel(), hid)
            for dst, msg in parts:
                dnum.index_add_(0, pos[dst], msg)
            if rgcn:
                deg_base = cache.deg[et][rows]
                deg_eff = deg_base + ddeg[rows] if ddeg is not None else deg_base
                deg_eff = deg_eff.clamp(min=1e-8)
                agg_new = (cache.num[li][et][rows] + dnum) / deg_eff.unsqueeze(-1)
                dz = agg_new - cache.agg[li][et][rows]
            else:
                dz = dnum
            contrib.append((d, rows, dz, F.linear(at(dh, lut, d, rows), cell.lin_self.weight)))

        # Δz per type on the union of touched rows, accumulated in relation order
        tmask = {}
        for d, rows, _, _ in contrib:
            tmask.setdefault(d, torch.zeros(N[d], dtype=torch.bool))[rows] = True
        dh_next = {}
        for nt, m in tmask.items():
            rows = m.nonzero(as_tuple=False).squeeze(1)
            pos = torch.empty(N[nt], dtype=torch.long)
            pos[rows] = torch.arange(rows.numel())
            dz = torch.zeros(rows.numel(), hid)
            for d, r, a, b in contrib:
                if d == nt:
                    dz[pos[r]] += a
                    dz[pos[r]] += b
            z_new = _z_base(cache, li, nt, rows) + dz
            h_new = F.relu(z_new) if li < L - 1 else z_new
            dh_next[nt] = (rows, h_new - cache.H[li + 1][nt][rows])
            Counters.add_verts(rows.numel())
        dh = dh_next

    out = {}
    for nt in cache.node_types:
        h = cache.H[-1][nt]
        if nt in dh:
            h = h.clone()
            h[dh[nt][0]] = h[dh[nt][0]] + dh[nt][1]
        out[nt] = h
    return out


def _alpha(cache, li, et, ei, cell, hs_rows, hd_rows):
    """methods._frozen_alpha_for on pre-gathered endpoint rows."""
    e = (cell.lin(hs_rows) * cell.att_src).sum(-1) + (cell.lin(hd_rows) * cell.att_dst).sum(-1)
    e = F.leaky_relu(e, 0.2)
    return torch.exp(e) / cache.att_sum[li][et][ei[1]].clamp(min=1e-12)
