"""Hypothetical states: sets of interventions and their composition.

A state is the frozenset of intervention ids applied — identified by the SET,
not the path (order independence). Composition merges base-relative
interventions; it is defined only for non-conflicting sets and raises
ConflictError otherwise:

  * two interventions deleting the same base edge column conflict
    (deleting the same edge twice is ill-defined);
  * feature deltas compose by addition (commutative), edge additions by
    concatenation (aggregation is order-invariant), deletions by disjoint
    union — so composition order cannot change the result.

Interventions are registered once in a Registry and referenced by id, mirroring
how states in an exploration session reference previously defined edits.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List

import torch

from src.data.graph_state import Intervention

State = FrozenSet[int]


class ConflictError(ValueError):
    """The intervention set is not composable (order-independent) as defined."""


class Registry:
    """Append-only store of interventions; states reference them by id."""

    def __init__(self):
        self.items: List[Intervention] = []

    def add(self, iv: Intervention) -> int:
        self.items.append(iv)
        return len(self.items) - 1

    def __getitem__(self, i: int) -> Intervention:
        return self.items[i]

    def __len__(self) -> int:
        return len(self.items)


def compose(registry: Registry, state: Iterable[int]) -> Intervention:
    """Merge the state's interventions into one base-relative Intervention.
    Iteration in sorted id order makes the composition deterministic; the
    merge rules make it order-independent."""
    add_edges: Dict = {}
    del_cols: Dict = {}
    feat_deltas: Dict = {}
    add_attr: Dict = {}

    for i in sorted(state):
        iv = registry[i]
        for et, new in iv.add_edges.items():
            add_edges[et] = (torch.cat([add_edges[et], new], dim=1)
                             if et in add_edges else new)
            if et in iv.add_attr:
                prev = add_attr.get(et)
                add_attr[et] = (torch.cat([prev, iv.add_attr[et]])
                                if prev is not None else iv.add_attr[et])
        for et, cols in iv.del_cols.items():
            del_cols[et] = (torch.cat([del_cols[et], cols])
                            if et in del_cols else cols)
        for nt, delta in iv.feat_deltas.items():
            feat_deltas[nt] = (feat_deltas[nt] + delta
                               if nt in feat_deltas else delta)

    for et, cols in del_cols.items():
        if torch.unique(cols).numel() != cols.numel():
            raise ConflictError(f"state deletes the same base edge twice on {et}")

    return Intervention(add_edges=add_edges, del_cols=del_cols,
                        feat_deltas=feat_deltas, add_attr=add_attr)


def conflicts(registry: Registry, state: Iterable[int], new_id: int) -> bool:
    """Would adding `new_id` to `state` create a conflicting set?"""
    new = registry[new_id]
    for i in state:
        old = registry[i]
        for et, cols in new.del_cols.items():
            if et in old.del_cols and cols.numel() and old.del_cols[et].numel():
                combined = torch.cat([old.del_cols[et], cols])
                if torch.unique(combined).numel() != combined.numel():
                    return True
    return False
