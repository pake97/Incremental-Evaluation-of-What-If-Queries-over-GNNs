"""What-if intervention layer for heterogeneous graphs.

A what-if query evaluates the trained GNN on a hypothetically modified graph
``G^D = G (+) Delta``.  ``Intervention`` is the Delta: a schema-valid bulk
update (edge additions, edge deletions, feature deltas) that references ids in
the base graph.  ``GraphState.apply`` materialises the modified input dicts
without ever writing to the stored graph, so interventions are hypothetical,
reversible, and droppable by construction.  Two things make apply cheap:

1. **Copy-on-write.**  The base graph is immutable.  Applying an intervention
   returns dicts that share the *same tensor objects* for every relation and
   node type the intervention does not touch.  Only changed tensors are
   reallocated.  A transfer-edge edit therefore touches ~1 of 13 relations.

2. **Vectorised edits.**  Deletions are a boolean mask over the one affected
   relation (`keep[cols] = False`), additions are a `cat`, feature edits are an
   out-of-place add on a single node type.  No Python-level per-edge loop.

Reverse relations are *not* stored here; the encoder derives reverse messages
at forward time, so the editor only manages the forward relations and can never
desync a forward/reverse pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

EdgeType = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Hard intervention
# ---------------------------------------------------------------------------
@dataclass
class Intervention:
    """A discrete change set.

    add_edges[et]  : LongTensor [2, k]   new (src, dst) columns to append.
    del_cols[et]   : LongTensor [m]      column indices into the *base*
                                         edge_index of ``et`` to remove.
    feat_deltas[nt]: FloatTensor [N_nt, d]  additive feature delta (a node type
                                         is "edited" if any row is non-zero).
    add_attr[et]   : FloatTensor [k, d]  optional edge_attr for the added edges
                                         (defaults to the relation's mean attr).
    """

    add_edges: Dict[EdgeType, torch.Tensor] = field(default_factory=dict)
    del_cols: Dict[EdgeType, torch.Tensor] = field(default_factory=dict)
    feat_deltas: Dict[str, torch.Tensor] = field(default_factory=dict)
    add_attr: Dict[EdgeType, torch.Tensor] = field(default_factory=dict)

    # -- accounting ---------------------------------------------------------
    def n_edits(self) -> int:
        n = sum(int(t.numel()) for t in self.del_cols.values())
        n += sum(int(t.size(1)) for t in self.add_edges.values())
        n += sum(1 for d in self.feat_deltas.values()
                 if d is not None and torch.any(d.abs().sum(-1) > 1e-9))
        return n


@dataclass
class GraphView:
    """Result of an intervention: dicts ready for the GNN forward.

    Untouched entries are the base graph's own tensors (shared by reference).
    """
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[EdgeType, torch.Tensor]
    edge_attr_dict: Dict[EdgeType, Optional[torch.Tensor]]


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------
class GraphState:
    """Immutable base graph; produces edited views in O(edited relation)."""

    def __init__(self, data):
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)
        self._x = {nt: data[nt].x for nt in self.node_types}
        self._ei = {et: data[et].edge_index for et in self.edge_types}
        self._ea = {et: getattr(data[et], "edge_attr", None) for et in self.edge_types}
        self.num_nodes = {nt: int(data[nt].num_nodes) for nt in self.node_types}
        self._pair_index: Dict[EdgeType, Dict[Tuple[int, int], int]] = {}
        self._mean_attr: Dict[EdgeType, torch.Tensor] = {}

    # -- base accessors (read-only) ----------------------------------------
    @property
    def x(self):
        return self._x

    @property
    def edge_index(self):
        return self._ei

    def base_view(self) -> GraphView:
        return GraphView(self._x, self._ei, self._ea)

    # -- pair <-> column lookup (built once per relation, cached) ----------
    def cols_for_pairs(self, et: EdgeType, pairs) -> torch.Tensor:
        """Resolve (src, dst) pairs to base edge columns (first match each)."""
        if et not in self._pair_index:
            ei = self._ei[et]
            idx: Dict[Tuple[int, int], int] = {}
            for c, (s, d) in enumerate(ei.t().tolist()):
                idx.setdefault((s, d), c)
            self._pair_index[et] = idx
        lut = self._pair_index[et]
        cols = [lut[(int(s), int(d))] for s, d in pairs if (int(s), int(d)) in lut]
        return torch.tensor(cols, dtype=torch.long)

    def _mean_edge_attr(self, et: EdgeType, k: int) -> Optional[torch.Tensor]:
        if self._ea[et] is None:
            return None
        if et not in self._mean_attr:
            ea = self._ea[et]
            self._mean_attr[et] = ea.mean(0) if ea.numel() else torch.zeros(ea.size(1))
        return self._mean_attr[et].unsqueeze(0).expand(k, -1).clone()

    # -- hard apply (copy-on-write) ----------------------------------------
    def apply(self, iv: Intervention) -> GraphView:
        x = dict(self._x)           # shallow: shared tensor refs
        ei = dict(self._ei)
        ea = dict(self._ea)

        for nt, delta in iv.feat_deltas.items():
            if delta is not None and delta.abs().sum() > 0:
                x[nt] = self._x[nt] + delta

        # deletions first (mask the base), then additions (append columns)
        for et, cols in iv.del_cols.items():
            if cols is None or cols.numel() == 0:
                continue
            base = self._ei[et]
            keep = torch.ones(base.size(1), dtype=torch.bool)
            keep[cols] = False
            ei[et] = base[:, keep]
            if ea[et] is not None:
                ea[et] = ea[et][keep]

        for et, new in iv.add_edges.items():
            if new is None or new.numel() == 0:
                continue
            ei[et] = torch.cat([ei[et], new], dim=1)
            if ea[et] is not None:
                attr = iv.add_attr.get(et)
                if attr is None:
                    attr = self._mean_edge_attr(et, new.size(1))
                ea[et] = torch.cat([ea[et], attr], dim=0)

        return GraphView(x, ei, ea)


__all__ = ["GraphState", "Intervention", "GraphView", "EdgeType"]
