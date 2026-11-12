"""Incremental what-if engine: dispatches to the right maintainer.

  delta   — RGCN + feature-only Δ + cache topology == query topology
            (low-rank propagation, src/incremental/delta.py)
  closure — everything else: topological Δ, HGAT, mixed Δ
            (L-hop dirty-row recompute, src/incremental/closure.py)

Both are exact; `method="auto"` picks the cheaper applicable one. Every result
carries stats (nodes recomputed, edges aggregated, per-layer frontier) — the
hardware-independent efficiency proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.data.graph_state import GraphState, Intervention
from src.data.schema import GraphSchema
from src.incremental.cache import ForwardCache
from src.incremental.closure import closure_forward
from src.incremental.closure_v2 import closure_forward_v2
from src.incremental.delta import delta_forward
from src.incremental.delta_v2 import delta_forward_v2
from src.incremental.index import get_index
from src.incremental.overlay import OverlayCache


@dataclass
class IncResult:
    h: Dict[str, torch.Tensor]
    cache: ForwardCache          # exact cached state of the resulting graph
    stats: dict


def feat_dirty_rows(iv: Intervention) -> Dict[str, torch.Tensor]:
    out = {}
    for nt, delta in iv.feat_deltas.items():
        rows = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if rows.numel():
            out[nt] = rows
    return out


def struct_dirty_rows(gs: GraphState, iv: Intervention) -> Dict[str, torch.Tensor]:
    """Endpoints of edited edges: their in-edge multiset changes in the forward
    relation (dst side) or its derived reverse (src side), at every layer."""
    acc: Dict[str, list] = {}

    def mark(nt, ids):
        acc.setdefault(nt, []).append(ids)

    for et, cols in iv.del_cols.items():
        if cols is None or not cols.numel():
            continue
        s, _, d = et
        base = gs.edge_index[et][:, cols]
        mark(s, base[0])
        mark(d, base[1])
    for et, new in iv.add_edges.items():
        if new is None or not new.numel():
            continue
        s, _, d = et
        mark(s, new[0])
        mark(d, new[1])
    return {nt: torch.unique(torch.cat(ids)) for nt, ids in acc.items()}


def struct_dirty_rows_v2(gs: GraphState, iv: Intervention) -> Dict[str, torch.Tensor]:
    """struct_dirty_rows with a bool mask over N_t instead of a sort over the
    endpoints (same rows, sorted)."""
    acc: Dict[str, torch.Tensor] = {}

    def mark(nt, ids):
        acc.setdefault(nt, torch.zeros(gs.num_nodes[nt], dtype=torch.bool))[ids] = True

    for et, cols in iv.del_cols.items():
        if cols is not None and cols.numel():
            s, _, d = et
            base = gs.edge_index[et][:, cols]
            mark(s, base[0])
            mark(d, base[1])
    for et, new in iv.add_edges.items():
        if new is not None and new.numel():
            s, _, d = et
            mark(s, new[0])
            mark(d, new[1])
    return {nt: m.nonzero().squeeze(1) for nt, m in acc.items()}


def _check_base_topology(cache, gs: GraphState, need_unedited: bool) -> None:
    """v2 reads topology from the BASE session index + the query's edit list,
    so the cache's own topology must be the base graph's (an OverlayCache
    always shares the base index; its edits matter only for the delta path)."""
    if isinstance(cache, OverlayCache):
        ok = cache._gs is gs and not (need_unedited and is_topological(cache._iv))
    else:
        ok = all(cache.edge_index_dict.get(et) is t for et, t in gs.edge_index.items())
    if not ok:
        raise ValueError("impl='v2' needs a cache on the base graph's topology")


def is_topological(iv: Intervention) -> bool:
    return (any(c is not None and c.numel() for c in iv.del_cols.values())
            or any(a is not None and a.numel() for a in iv.add_edges.values()))


@torch.no_grad()
def incremental_forward(model, gs: GraphState, cache: ForwardCache,
                        iv: Intervention, schema: Optional[GraphSchema] = None,
                        method: str = "auto", impl: str = "v1") -> IncResult:
    """Exact incremental answer to the what-if `iv`, resumed from `cache`.

    `iv` is base-relative (the full composed intervention set of the target
    state); `cache` may belong to the base graph or to any ancestor state —
    the dirty seeds must then describe the RESIDUAL edits (see reuse engine).
    This entry point covers the common case cache == base graph state.
    """
    if schema is not None:
        schema.validate(iv)
    topo = is_topological(iv)
    arch = model.cfg["model_type"]

    if method == "auto":
        method = "delta" if (arch == "rgcn" and not topo) else "closure"
    if method == "delta" and (arch != "rgcn" or topo):
        raise ValueError("delta path requires RGCN and a feature-only intervention")

    if impl == "v2":
        _check_base_topology(cache, gs, need_unedited=(method == "delta"))
        index = get_index(cache)
        if method == "delta":
            h, new_cache, stats = delta_forward_v2(model, cache, gs, iv, index)
        else:
            h, new_cache, stats = closure_forward_v2(
                model, cache, gs, iv, index,
                feat_dirty=feat_dirty_rows(iv),
                struct_dirty=struct_dirty_rows_v2(gs, iv))
        return IncResult(h, new_cache, stats)
    if impl != "v1":
        raise ValueError(f"unknown impl '{impl}'")
    if method == "delta":
        h, new_cache, stats = delta_forward(model, cache, iv.feat_deltas)
    else:
        view = gs.apply(iv)
        h, new_cache, stats = closure_forward(
            model, cache, view,
            feat_dirty=feat_dirty_rows(iv),
            struct_dirty=struct_dirty_rows(gs, iv))
        # gs.apply masks / concatenates every edited relation in full
        stats["edges_scanned"] += sum(gs.edge_index[et].size(1) for et in
                                      set(iv.del_cols) | set(iv.add_edges))
    return IncResult(h, new_cache, stats)
