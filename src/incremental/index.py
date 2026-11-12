"""Session index for the v2 maintainers: per forward relation, a CSR (by
source) and a CSC (by destination) over the BASE graph's edges.

Built once per session (topology is fixed for the life of a base cache; a
what-if's edits are applied on top as an edit list, never written here).
Incident edges of a set of rows are then a union of indptr ranges, so their
cost is the number of incident edges, not |E_r|:

  out-edges of src rows S : csr_dst[ ranges csr_ptr[S]..csr_ptr[S+1] ]
  in-edges  of dst rows D : csc_src[ ranges csc_ptr[D]..csc_ptr[D+1] ]

Both orders come from a STABLE sort, so within one row the edges keep base
column order — the order from-scratch aggregates them in. The *_eid arrays map
back to base columns (for deleting edges by column). A reverse relation reuses
its forward relation's index with the roles swapped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

EdgeType = Tuple[str, str, str]


@dataclass
class RelIndex:
    csr_ptr: torch.Tensor    # [N_src + 1]
    csr_dst: torch.Tensor    # [E] dst of each edge, edges ordered by src
    csr_eid: torch.Tensor    # [E] base column of each edge in csr order
    csc_ptr: torch.Tensor    # [N_dst + 1]
    csc_src: torch.Tensor    # [E] src of each edge, edges ordered by dst
    csc_eid: torch.Tensor    # [E] base column of each edge in csc order


class GraphIndex:
    def __init__(self, rels: Dict[EdgeType, RelIndex], build_s: float):
        self.rels = rels
        self.build_s = build_s

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for r in self.rels.values() for t in vars(r).values())


def _ptr(keys: torch.Tensor, n: int) -> torch.Tensor:
    ptr = torch.zeros(n + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.bincount(keys, minlength=n), 0)
    return ptr


def build_index(edge_index_dict: Dict[EdgeType, torch.Tensor],
                num_nodes: Dict[str, int]) -> GraphIndex:
    t0 = time.perf_counter()
    rels = {}
    for (s, r, d), ei in edge_index_dict.items():
        src, dst = ei[0], ei[1]
        pr = torch.sort(src, stable=True).indices
        pc = torch.sort(dst, stable=True).indices
        rels[(s, r, d)] = RelIndex(_ptr(src, num_nodes[s]), dst[pr], pr,
                                   _ptr(dst, num_nodes[d]), src[pc], pc)
    return GraphIndex(rels, time.perf_counter() - t0)


def get_index(cache) -> GraphIndex:
    """The cache's session index, built on first use (callers that report
    build cost build it explicitly right after build_cache)."""
    if getattr(cache, "_index", None) is None:
        n = {nt: t.size(0) for nt, t in cache.h[0].items()}
        cache._index = build_index(cache.edge_index_dict, n)
    return cache._index


def ragged(ptr: torch.Tensor, rows: torch.Tensor):
    """Positions of all edges in the indptr ranges of `rows`, plus the
    per-edge owner (index into `rows`). Cost ~ |rows| + #edges returned."""
    starts = ptr[rows]
    cnt = ptr[rows + 1] - starts
    owner = torch.repeat_interleave(torch.arange(rows.numel()), cnt)
    if owner.numel() == 0:
        return owner, owner
    offs = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(owner.numel()) + (starts - offs)[owner]
    return pos, owner


def in_sorted(x: torch.Tensor, sorted_vals: torch.Tensor) -> torch.Tensor:
    """Elementwise membership of x in a sorted 1-D tensor (no |E| scan)."""
    if sorted_vals.numel() == 0:
        return torch.zeros_like(x, dtype=torch.bool)
    pos = torch.searchsorted(sorted_vals, x).clamp_(max=sorted_vals.numel() - 1)
    return sorted_vals[pos] == x
