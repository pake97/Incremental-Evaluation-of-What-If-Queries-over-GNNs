"""Cached forward state of one hypothetical graph.

A ForwardCache stores, for the graph it was computed on:
  * h[0..L]  — post-activation embeddings per layer (h[0] = input projection),
  * z[0..L-1] — pre-activations (input to the ReLU that produced h[l+1]),
  * the graph's own x / edge_index dicts (by reference — cheap),
  * lazily, per-relation in-degrees (needed by the delta path).

Incremental maintenance derives a *new* cache copy-on-write: only tensors
holding recomputed rows are cloned, everything untouched is shared.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.model import _key

EdgeType = Tuple[str, str, str]


class ForwardCache:
    def __init__(self, h: List[Dict[str, torch.Tensor]],
                 z: List[Dict[str, torch.Tensor]],
                 x_dict: Dict[str, torch.Tensor],
                 edge_index_dict: Dict[EdgeType, torch.Tensor]):
        self.h = h                       # len L+1
        self.z = z                       # len L
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self._deg: Dict[EdgeType, torch.Tensor] = {}
        self._ei_all: Dict[EdgeType, torch.Tensor] = {}

    @property
    def h_final(self) -> Dict[str, torch.Tensor]:
        return self.h[-1]

    def ei_all(self, fwd_types) -> Dict[EdgeType, torch.Tensor]:
        """Forward relations plus their derived reverses, built once and reused.
        Topology is fixed for the life of a cache, so the maintainers need not
        re-flip every relation on every call (an op-count reduction; exact —
        the reverse index is the same tensor either way)."""
        if not self._ei_all:
            for (s, r, d) in fwd_types:
                ei = self.edge_index_dict[(s, r, d)]
                self._ei_all[(s, r, d)] = ei
                self._ei_all[(d, "rev_" + r, s)] = ei.flip(0)
        return self._ei_all

    def deg(self, et: EdgeType, dst_size: int) -> torch.Tensor:
        """In-degree per dst node for a (possibly reverse) relation of THIS
        cache's topology — the denominator of the RGCN mean aggregator."""
        if et not in self._deg:
            s, r, d = et
            if r.startswith("rev_"):
                base = self.edge_index_dict[(d, r[4:], s)]
                dst = base[0]
            else:
                dst = self.edge_index_dict[et][1]
            ones = torch.ones(dst.numel())
            self._deg[et] = torch.zeros(dst_size).index_add_(0, dst, ones).clamp(min=1e-8)
        return self._deg[et]

    def nbytes(self) -> int:
        seen, total = set(), 0
        for layer in list(self.h) + list(self.z):
            for t in layer.values():
                if id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
        return total


@torch.no_grad()
def build_cache(model, view) -> ForwardCache:
    """Full from-scratch forward that records every layer's h and z.

    Mirrors HeteroEncoder.forward exactly; tests assert h_final matches
    model.encode on the same view.
    """
    enc = model.encoder
    model.eval()

    h = {nt: enc.proj[nt](view.x_dict[nt]) for nt in enc.node_types}

    ei_all, ew_all = {}, {}
    for (s, r, d) in enc.fwd_types:
        ei = view.edge_index_dict[(s, r, d)]
        ew = torch.ones(ei.size(1))
        ei_all[(s, r, d)] = ei
        ew_all[(s, r, d)] = ew
        rev = (d, "rev_" + r, s)
        ei_all[rev] = ei.flip(0)
        ew_all[rev] = ew

    hs, zs = [h], []
    for li, layer in enumerate(enc.layers):
        acc = {nt: [] for nt in enc.node_types}
        for et in enc.all_types:
            s, _, d = et
            acc[d].append(layer[_key(et)](h[s], h[d], ei_all[et], ew_all[et],
                                          h[d].size(0)))
        z, h_new = {}, {}
        for nt in enc.node_types:
            zt = sum(acc[nt]) if acc[nt] else h[nt]
            z[nt] = zt
            h_new[nt] = F.relu(zt) if li < enc.num_layers - 1 else zt
        zs.append(z)
        hs.append(h_new)
        h = h_new

    return ForwardCache(hs, zs, dict(view.x_dict), dict(view.edge_index_dict))
