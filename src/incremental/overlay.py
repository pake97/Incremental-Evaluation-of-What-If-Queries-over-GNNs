"""Row overlays: the v2 maintainers' copy-on-write at ROW granularity.

A RowOverlay is (base, rows, vals): the tensor `base` with the sorted `rows`
replaced by `vals`. Reads of a set of rows go through `gather` (cost ~ rows
read, never N). A dense tensor is built only by `dense()` — the v2 maintainers
call it transiently for the final layer they return (the query output, as
from-scratch returns it); the cache keeps every layer, the last included, as
overlays. Anything else that indexes a layer of an OverlayCache gets it
materialized on first access (v1 consumers: reuse, analyst, baselines).

OverlayCache is a ForwardCache whose layers may hold overlays and whose graph
is (base GraphState ⊕ the intervention) — an edge/feature overlay, materialized
only if a consumer reads x_dict / edge_index_dict.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Union

import torch

from src.incremental.cache import ForwardCache


class RowOverlay:
    def __init__(self, base, rows: torch.Tensor, vals: torch.Tensor):
        self.base, self.rows, self.vals = base, rows, vals   # rows sorted, unique

    def size(self, dim=None):
        return self.base.size() if dim is None else self.base.size(dim)

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        out = gather(self.base, idx)
        if self.rows.numel():
            pos = torch.searchsorted(self.rows, idx).clamp_(max=self.rows.numel() - 1)
            hit = self.rows[pos] == idx
            out[hit] = self.vals[pos[hit]]
        return out

    def dense(self) -> torch.Tensor:
        # walk the overlay chain: clone the dense root once, then apply the
        # overlays root-first so the newest rows win
        chain, e = [], self
        while isinstance(e, RowOverlay):
            chain.append(e)
            e = e.base
        t = e.clone()
        for o in reversed(chain):
            t[o.rows] = o.vals
        return t

    def own_bytes(self) -> int:
        return (self.rows.numel() * self.rows.element_size()
                + self.vals.numel() * self.vals.element_size())


Entry = Union[torch.Tensor, RowOverlay]


def gather(t: Entry, idx: torch.Tensor) -> torch.Tensor:
    return t.gather(idx) if isinstance(t, RowOverlay) else t[idx]


def dense(t: Entry) -> torch.Tensor:
    return t.dense() if isinstance(t, RowOverlay) else t


class _Layer(Mapping):
    """Read-only view of one layer: overlays are materialized (and memoized)
    on item access, so dict(layer) / layer[nt] behave like a dense layer."""

    def __init__(self, raw: Dict[str, Entry]):
        self.raw = raw

    def __getitem__(self, nt):
        e = self.raw[nt]
        if isinstance(e, RowOverlay):
            e = self.raw[nt] = e.dense()
        return e

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)


class OverlayCache(ForwardCache):
    def __init__(self, h: List[Dict[str, Entry]], z: List[Dict[str, Entry]],
                 parent: ForwardCache, gs, iv, index):
        self.h = [_Layer(d) if any(isinstance(e, RowOverlay) for e in d.values())
                  else d for d in h]
        self.z = [_Layer(d) if any(isinstance(e, RowOverlay) for e in d.values())
                  else d for d in z]
        self._parent, self._gs, self._iv, self._index = parent, gs, iv, index
        self._view = None
        self._deg, self._ei_all = {}, {}

    @property
    def h_final(self):
        """Dense final layer, built transiently (not memoized into the cache)."""
        return {nt: dense(e) for nt, e in _raw(self.h[-1]).items()}

    def _graph(self):
        if self._view is None:
            self._view = self._gs.apply(self._iv)
        return self._view

    @property
    def x_dict(self):
        return dict(self._graph().x_dict)

    @property
    def edge_index_dict(self):
        return dict(self._graph().edge_index_dict)

    def nbytes(self) -> int:
        """Retained bytes of THIS state on top of its parent: overlay rows +
        values, plus dense tensors it owns (not shared with the parent)."""
        shared = {id(t) for layers in (self._parent.h, self._parent.z)
                  for layer in layers for t in _raw(layer).values()}
        seen, total = set(), 0
        for layers in (self.h, self.z):
            for layer in layers:
                for e in _raw(layer).values():
                    if id(e) in seen or id(e) in shared:
                        continue
                    seen.add(id(e))
                    total += e.own_bytes() if isinstance(e, RowOverlay) \
                        else e.numel() * e.element_size()
        return total


def _raw(layer):
    return layer.raw if isinstance(layer, _Layer) else layer
