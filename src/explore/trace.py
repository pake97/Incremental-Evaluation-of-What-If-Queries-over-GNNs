"""Synthetic exploration-DAG traces with controlled, swept parameters.

This is the honest way to show when reuse pays off: every structural property
that drives reuse benefit is an explicit, swept knob —

  branching       max children spawned from one state
  max_depth       cap on |D| per state
  backtrack_rate  P(next expansion starts from a random earlier state
                  instead of the current one)
  share_rate      P(a step re-applies an EXISTING intervention from the pool
                  instead of sampling a fresh one) — the prefix-sharing /
                  re-convergence knob: shared Δs let different paths reach the
                  same state (DAG re-convergence) and make ancestors overlap
  feature_frac    intervention mix (rest is topological)
  cohort_size /   locality of the two intervention kinds
  topo_edits

The generator records the realized structure (states, DAG edges, re-convergence
events, per-state parents) so experiments can correlate reuse benefit with the
*measured* sharing, not just the requested knob.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from src.data.graph_state import GraphState
from src.data.schema import GraphSchema
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.explore.state import Registry, State, conflicts

EdgeType = Tuple[str, str, str]


@dataclass
class TraceConfig:
    n_states: int = 50
    branching: int = 3
    max_depth: int = 6
    backtrack_rate: float = 0.3
    share_rate: float = 0.3
    feature_frac: float = 0.5
    cohort_size: int = 32
    topo_edits: int = 8
    node_type: str = ""          # cohort node type (dataset task spec label_nt)
    relation: EdgeType = ()      # topo relation (dataset task spec core_rel)
    seed: int = 42


@dataclass
class Trace:
    registry: Registry
    order: List[State]                       # distinct states in visit order
    dag_edges: List[Tuple[State, State, int]]  # (parent, child, delta id)
    parents: Dict[State, set] = field(default_factory=dict)
    reconvergences: int = 0

    def structure(self) -> dict:
        depths = [len(s) for s in self.order]
        multi_parent = sum(1 for p in self.parents.values() if len(p) > 1)
        return {
            "n_states": len(self.order),
            "n_dag_edges": len(self.dag_edges),
            "n_deltas": len(self.registry),
            "max_depth": max(depths) if depths else 0,
            "mean_depth": sum(depths) / max(len(depths), 1),
            "reconvergences": self.reconvergences,
            "multi_parent_states": multi_parent,
            "sharing": multi_parent / max(len(self.order), 1),
        }


def generate(schema: GraphSchema, gs: GraphState, cfg: TraceConfig) -> Trace:
    g = torch.Generator().manual_seed(cfg.seed)

    def rand() -> float:
        return float(torch.rand(1, generator=g))

    def new_delta() -> int:
        if rand() < cfg.feature_frac:
            iv = sample_feature_iv(schema, gs, g, cfg.node_type, cfg.cohort_size)
        else:
            iv = sample_topo_iv(schema, gs, g, cfg.relation,
                                n_add=cfg.topo_edits, n_del=cfg.topo_edits)
        return reg.add(iv)

    reg = Registry()
    root: State = frozenset()
    trace = Trace(registry=reg, order=[], dag_edges=[], parents={})
    seen = {root}
    children: Dict[State, int] = {root: 0}
    current = root

    guard = 0
    while len(trace.order) < cfg.n_states and guard < 50 * cfg.n_states:
        guard += 1
        # pick the parent state
        eligible = [s for s in seen
                    if len(s) < cfg.max_depth and children.get(s, 0) < cfg.branching]
        if not eligible:
            break
        if (current not in eligible) or rand() < cfg.backtrack_rate:
            current = eligible[int(torch.randint(0, len(eligible), (1,), generator=g))]

        # pick the delta: shared from the pool, or fresh
        did = None
        if len(reg) and rand() < cfg.share_rate:
            for _ in range(5):
                cand = int(torch.randint(0, len(reg), (1,), generator=g))
                if cand not in current and not conflicts(reg, current, cand):
                    did = cand
                    break
        if did is None:
            did = new_delta()
            if conflicts(reg, current, did):
                continue

        child = frozenset(current | {did})
        trace.dag_edges.append((current, child, did))
        trace.parents.setdefault(child, set()).add(current)
        children[current] = children.get(current, 0) + 1
        if child in seen:
            trace.reconvergences += 1
        else:
            seen.add(child)
            children[child] = 0
            trace.order.append(child)
        current = child

    return trace
